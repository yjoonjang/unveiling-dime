import os
import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import ir_datasets

from sentence_transformers import SentenceTransformer
from entmax import entmax_bisect

class QueryAdapter(nn.Module):
    def __init__(self, embedding_dim, hidden_dim_ratio=0.5):
        super().__init__()
        hidden_dim = int(embedding_dim * hidden_dim_ratio)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)

def load_index_and_metadata(index_dir):
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
    query_embedding = np.expand_dims(query_embedding, axis=0).astype(np.float32)
    distances, indices = index.search(query_embedding, k=top_k)
    return distances[0], indices[0]

def get_faiss_matches(index, doc_ids, query_vector, top_k=1000):
    distances, indices = search_index(index, query_vector, top_k=top_k)
    matches = [{"id": str(doc_ids[idx]), "score": float(dist)} for dist, idx in zip(distances, indices)]
    return matches

def produce_trec_run_lines(query_id: str, query_matches):
    query_matches.sort(key=lambda x: -x["score"], reverse=True)
    return [f'{query_id} Q0 {m["id"]} {i+1} {m["score"]} dense' for i, m in enumerate(query_matches)]

def modify_query_with_adapter(query_vector, adapter, alpha):
    query_tensor = torch.from_numpy(query_vector).to(next(adapter.parameters()).device)
    adapter_output = adapter(query_tensor)
    interaction_vector = query_tensor * adapter_output
    sparse_weights = entmax_bisect(interaction_vector, alpha=alpha, dim=0)
    binary_mask = (sparse_weights > 0).float()
    modified_query_vector = query_tensor * binary_mask
    non_zero_dims = torch.count_nonzero(binary_mask).item()
    return modified_query_vector.cpu().numpy(), non_zero_dims

def embed_text(text: str, normalize_embeddings: bool, prompt=None) -> np.ndarray:
    return model.encode(text, normalize_embeddings=normalize_embeddings, prompt=prompt)

def run_queries(ir_ds_path, out_trec_path, top_k, normalize, prompt, log_file, alpha):
    dataset = ir_datasets.load(ir_ds_path)
    print("Computing and modifying query vectors...")
    new_query_vectors_map = {}
    sparsity_logs = []

    for query_id, query_text in tqdm(dataset.queries_iter(), desc="Processing queries"):
        orig_vec = embed_text(query_text, normalize, prompt)
        new_vec, nz_dims = modify_query_with_adapter(orig_vec, adapter, alpha)
        new_query_vectors_map[str(query_id)] = new_vec
        sparsity_logs.append(f"{query_id}\t{nz_dims}\t{orig_vec.shape[0]}")

    if log_file:
        with open(log_file, "w") as f:
            f.write("query_id\tnon_zero_dimensions\ttotal_dimensions\n")
            f.writelines(f"{line}\n" for line in sparsity_logs)

    print("Executing queries against FAISS...")
    final_results_map = {
        q_id: get_faiss_matches(index, doc_ids, q_vec, top_k=top_k)
        for q_id, q_vec in tqdm(new_query_vectors_map.items(), desc="Executing queries")
    }

    full_trec_run = [line for q_id, docs in final_results_map.items() for line in produce_trec_run_lines(q_id, docs)]
    with open(out_trec_path, "w") as file:
        file.writelines(f"{line}\n" for line in full_trec_run)
    print("Done.")

parser = argparse.ArgumentParser(description="Run queries with a trained Sparseq Adapter.")
parser.add_argument("--model", required=True)
parser.add_argument("--adapter_path", required=True)
parser.add_argument("--index-dir", required=True)
parser.add_argument("--ir-ds-query-path", required=True)
parser.add_argument("--output-trec-name", required=True)
parser.add_argument("--top-k", type=int, default=1000)
parser.add_argument("--alpha", type=float, default=2.0)
parser.add_argument("--sparsity-log-file", default=None)
parser.add_argument("--normalize_embeddings", action="store_true")
parser.add_argument("--query_prompt", default=None)

args = parser.parse_args()

index, doc_ids = load_index_and_metadata(args.index_dir)
model = SentenceTransformer(args.model)
embedding_dim = model.get_sentence_embedding_dimension()
adapter = QueryAdapter(embedding_dim).to(model.device)
adapter.load_state_dict(torch.load(args.adapter_path, map_location=model.device))
adapter.eval()

if __name__ == "__main__":
    run_queries(
        ir_ds_path=args.ir_ds_query_path,
        out_trec_path=args.output_trec_name,
        top_k=args.top_k,
        normalize=args.normalize_embeddings,
        prompt=args.query_prompt,
        log_file=args.sparsity_log_file,
        alpha=args.alpha,
    )
