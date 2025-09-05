import os
import faiss
import numpy as np
import torch
from tqdm import tqdm
import argparse
import pandas as pd
import ir_datasets

from sentence_transformers import SentenceTransformer
from entmax import entmax_bisect

def load_index_and_metadata(index_dir):
    """
    Loads the FAISS index and associated metadata (document IDs).
    """
    index_file = os.path.join(index_dir, "faiss_hnsw_index.bin")
    metadata_file = os.path.join(index_dir, "doc_ids.npy")

    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Index or metadata not found in {index_dir}")

    print(f"[info] Loading FAISS index from {index_file}...")
    index = faiss.read_index(index_file)

    print(f"[info] Loading document IDs from {metadata_file}...")
    doc_ids = np.load(metadata_file, allow_pickle=True)
    gpu_index = faiss.index_cpu_to_all_gpus(index)

    return gpu_index, doc_ids

def search_index(index, query_embedding, top_k=10):
    """
    Searches the FAISS index with a given query embedding.
    """
    query_embedding = np.expand_dims(query_embedding, axis=0).astype(np.float32)
    distances, indices = index.search(query_embedding, k=top_k)
    return distances[0], indices[0]

def get_faiss_matches(index, doc_ids, query_vector, top_k=1000):
    """
    Retrieve top_k matches from a FAISS index, returning a list of dicts.
    """
    distances, indices = search_index(index, query_vector, top_k=top_k)
    matches = []
    for dist, idx in zip(distances, indices):
        doc_id = doc_ids[idx]
        score = float(dist)
        matches.append({"id": str(doc_id), "score": score})
    return matches

def produce_trec_run_lines(query_id: str, query_matches):
    """
    Produce TREC run lines from a list of matches.
    """
    query_matches.sort(key=lambda x: -x["score"])
    lines = []
    for i, match in enumerate(query_matches):
        line = f'{query_id} Q0 {match["id"]} {i+1} {match["score"]} dense'
        lines.append(line)
    return lines

def modify_query_with_P(query_vector, P, alpha):
    """
    Modify the query vector using the trained parameter P and entmax.
    """
    query_tensor = torch.from_numpy(query_vector).to(P.device)
    
    # Compute interaction and apply entmax
    interaction_vector = query_tensor * P
    sparse_weights = entmax_bisect(interaction_vector, alpha=alpha, dim=0)
    
    # Create binary mask and apply to query
    binary_mask = (sparse_weights > 0).float()
    modified_query_vector = query_tensor * binary_mask
    
    non_zero_dims = torch.count_nonzero(binary_mask).item()
    
    return modified_query_vector.cpu().numpy(), non_zero_dims

def embed_text(text: str, normalize_embeddings: bool, prompt=None) -> np.ndarray:
    """Embed the given text using the loaded sentence-transformer model."""
    return model.encode(text, normalize_embeddings=normalize_embeddings, prompt=prompt)

def run_queries(
    ir_dataset_queries_path: str,
    output_trec_path: str,
    top_k: int,
    normalize_embeddings: bool,
    query_prompt: str,
    sparsity_log_file: str,
    alpha: float
):
    # Load IR dataset queries
    dataset = ir_datasets.load(ir_dataset_queries_path)
    
    # Compute query embeddings
    print("Computing and modifying query vectors...")
    new_query_vectors_map = {}
    sparsity_logs = []

    for q in tqdm(dataset.queries_iter(), desc="Processing queries"):
        query_id, query_text = q
        orig_query_vector = embed_text(query_text, normalize_embeddings, query_prompt)
        
        # Modify query vector
        new_query_vector, non_zero_dims = modify_query_with_P(orig_query_vector, P, alpha)
        new_query_vectors_map[str(query_id)] = new_query_vector
        
        total_dims = orig_query_vector.shape[0]
        sparsity_logs.append(f"{query_id}\t{non_zero_dims}\t{total_dims}")

    print("Done computing and modifying query vectors.")

    # Write sparsity log if path is provided
    if sparsity_log_file:
        print(f"Writing sparsity log to {sparsity_log_file}")
        with open(sparsity_log_file, "w") as f:
            f.write("query_id\tnon_zero_dimensions\ttotal_dimensions\n")
            for log_line in sparsity_logs:
                f.write(f"{log_line}\n")

    # Execute queries against FAISS
    print("Executing queries against FAISS...")
    final_results_map = {}
    for q_id, q_vec in tqdm(new_query_vectors_map.items(), desc="Executing queries"):
        matches = get_faiss_matches(index, doc_ids, q_vec, top_k=top_k)
        final_results_map[q_id] = matches
    print("Done executing queries.")

    # Produce and write TREC run file
    full_trec_run = []
    for q_id, docs in final_results_map.items():
        t_lines = produce_trec_run_lines(q_id, docs)
        full_trec_run.extend(t_lines)
    
    print(f"Writing TREC run to {output_trec_path}")
    with open(output_trec_path, "w") as file:
        for line in full_trec_run:
            file.write(f"{line}\n")
    print("Done.")

# Argument parser setup
parser = argparse.ArgumentParser(description="Run queries with a trained sparseq parameter.")
parser.add_argument("--model", type=str, required=True, help="Model name or path.")
parser.add_argument("--p_param_path", type=str, required=True, help="Path to the trained parameter P file (.pt).")
parser.add_argument("--index-dir", type=str, required=True, help="Directory of the FAISS index.")
parser.add_argument("--ir-ds-query-path", type=str, required=True, help="ir_datasets path for queries.")
parser.add_argument("--output-trec-name", type=str, required=True, help="Output TREC run file path.")
parser.add_argument("--top-k", type=int, default=1000, help="Number of results to retrieve.")
parser.add_argument("--alpha", type=float, default=2.0, help="Alpha for entmax.")
parser.add_argument("--sparsity-log-file", type=str, default=None, help="Path to save sparsity log.")
parser.add_argument("--normalize_embeddings", action="store_true", help="Normalize embeddings.")
parser.add_argument("--query_prompt", type=str, default=None, help="Query prompt.")

args = parser.parse_args()

# Load FAISS index and model
print(f"[info] Loading FAISS index from: {args.index_dir}")
index, doc_ids = load_index_and_metadata(args.index_dir)

print(f"[info] Loading query embedding model: {args.model}")
model = SentenceTransformer(args.model)

# Load trained parameter P
print(f"[info] Loading trained parameter P from: {args.p_param_path}")
P = torch.load(args.p_param_path, map_location=model.device)

if __name__ == "__main__":
    run_queries(
        ir_dataset_queries_path=args.ir_ds_query_path,
        output_trec_path=args.output_trec_name,
        top_k=args.top_k,
        normalize_embeddings=args.normalize_embeddings,
        query_prompt=args.query_prompt,
        sparsity_log_file=args.sparsity_log_file,
        alpha=args.alpha,
    )
