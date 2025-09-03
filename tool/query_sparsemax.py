import os
import faiss
import numpy as np
import torch
from tqdm import tqdm
import argparse
import pandas as pd
import ir_datasets

from sentence_transformers import SentenceTransformer, util
from entmax import sparsemax

def load_index_and_metadata(index_dir):
    """
    Loads the FAISS index and associated metadata (document IDs).
    :param index_dir: Directory containing the FAISS index and metadata files.
    :return: (index, doc_ids)
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
        doc_vec = index.reconstruct(int(idx))
        score = float(dist)
        matches.append({"id": str(doc_id), "score": score, "values": doc_vec})
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

def normalize(arr, t_min, t_max):
    diff = t_max - t_min
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max == arr_min:
        return np.full_like(arr, (t_min + t_max) / 2)
    norm_arr = (((arr - arr_min) * diff) / (arr_max - arr_min)) + t_min
    return norm_arr

def softmax(scores, temperature=1.0):
    """
    Compute softmax over an array of scores with optional temperature.
    """
    scaled_scores = scores / temperature
    exps = np.exp(scaled_scores - np.max(scaled_scores))
    return exps / np.sum(exps)

def compute_centroid(query_matches, k=None, weighted=False, attention_type="linear", temperature=1.0):
    """
    Compute the centroid of the top-k match embeddings.
    """
    if k is not None and k > 0:
        query_matches = query_matches[:k] if len(query_matches) > k else query_matches

    embeddings = []
    scores = []
    for match in query_matches:
        embeddings.append(match["values"])
        scores.append(match["score"])

    embeddings_array = np.array(embeddings, dtype=float)

    if weighted:
        if attention_type == "linear":
            weights = normalize(np.array(scores), 0, 1)
        else:  # "softmax"
            weights = softmax(np.array(scores), temperature)
        centroid = np.average(embeddings_array, axis=0, weights=weights)
    else:
        centroid = np.mean(embeddings_array, axis=0)

    return centroid

def sparsemax_zero_out_dims_in_query(centroid, query_vector):
    """
    Apply sparsemax to the dimension-wise product between centroid and query_vector,
    then use the resulting binary mask to zero out dimensions in query_vector.
    Returns the modified query vector and the number of non-zero dimensions.
    """
    # Compute dimension-wise importance scores
    itx_vec = np.multiply(centroid, query_vector)
    
    # Convert to torch tensor for sparsemax
    importance_tensor = torch.tensor(itx_vec, dtype=torch.float32)
    
    # Apply sparsemax to get sparse attention weights
    sparse_weights = sparsemax(importance_tensor, dim=0)
    
    # Convert back to numpy and create binary mask
    sparse_weights_np = sparse_weights.detach().numpy()
    binary_mask = (sparse_weights_np > 0).astype(np.float32)
    
    # Apply binary mask to original query vector
    modified_query_vector = query_vector * binary_mask
    
    non_zero_dims = np.count_nonzero(binary_mask)

    return modified_query_vector, non_zero_dims

def embed_text(text: str, normalize_embeddings: bool, prompt=None) -> np.ndarray:
    """Embed the given text using the loaded sentence-transformer model."""
    return model.encode(text, normalize_embeddings=normalize_embeddings, prompt=prompt)

def run_queries(
    ir_dataset_queries_path: str,
    output_trec_path: str,
    top_k: int,
    weighted: bool,
    prf_k: int,
    reuse_step1_retrieval: bool,
    metric: str,
    attention_type: str,
    temperature: float,
    llm_docs: str,
    normalize_embeddings: bool,
    query_prompt: str,
    doc_prompt: str,
    sparsity_log_file: str
):
    # Load IR dataset queries
    try:
        dataset = ir_datasets.load(ir_dataset_queries_path)
    except Exception:
        print(f"No dataset found for {ir_dataset_queries_path} on https://ir-datasets.com/")
        raise
    
    # Compute query embeddings
    print("Computing query vectors...")
    query_vectors_map = {}
    for q in tqdm(dataset.queries_iter(), desc="Computing query vectors"):
        query_id = q[0]
        query_text = q[1]
        query_vector = embed_text(query_text, normalize_embeddings, query_prompt)
        query_vectors_map[str(query_id)] = query_vector
    print("Done computing query vectors")

    if llm_docs is None:
        # Step 1: Retrieve using original query
        print("Executing first-stage queries against FAISS...")
        step1_results_map = {}
        for q_id, q_vec in tqdm(query_vectors_map.items(), desc="Executing queries"):
            matches = get_faiss_matches(index, doc_ids, q_vec, top_k=top_k)
            step1_results_map[q_id] = matches
        print("Done executing first-stage queries.")

    if prf_k <= 0:
        final_results_map = step1_results_map
        # Produce TREC run lines
        full_trec_run = []
        for q_id, docs in final_results_map.items():
            t_lines = produce_trec_run_lines(q_id, docs)
            full_trec_run.extend(t_lines)    
    else: 
        sparsity_logs = []
        if llm_docs is not None:
            llm_doc_df = pd.read_csv(llm_docs, header=None, names=['qid', 'query_text','doc_text'])
            llm_docs_map = {}
            for _, row in llm_doc_df.iterrows():
                query_id = row['qid']
                llm_doc = row['doc_text']
                llm_docs_map[str(query_id)] = embed_text(llm_doc, normalize_embeddings, doc_prompt)
            print("Done computing llm doc vectors")
            print("Computing new query vectors (LLM + sparsemax zero-out)...")
            new_query_vectors_map = {}
            for q_id, _ in query_vectors_map.items():
                orig_query_vector = query_vectors_map[q_id]
                new_query_vector, non_zero_dims = sparsemax_zero_out_dims_in_query(
                    llm_docs_map[q_id],
                    orig_query_vector
                )
                total_dims = orig_query_vector.shape[0]
                sparsity_logs.append(f"{q_id}\t{non_zero_dims}\t{total_dims}")
                new_query_vectors_map[q_id] = new_query_vector
        else:
            # Compute new query vectors by PRF + sparsemax zero-out
            print("Computing new query vectors (PRF + sparsemax zero-out)...")
            new_query_vectors_map = {}
            for q_id, q_result in step1_results_map.items():
                centroid = compute_centroid(
                    q_result,
                    k=prf_k,
                    weighted=weighted,
                    attention_type=attention_type,
                    temperature=temperature
                )
                orig_query_vector = query_vectors_map[q_id]
                new_query_vector, non_zero_dims = sparsemax_zero_out_dims_in_query(
                    centroid,
                    orig_query_vector
                )
                total_dims = orig_query_vector.shape[0]
                sparsity_logs.append(f"{q_id}\t{non_zero_dims}\t{total_dims}")
                new_query_vectors_map[q_id] = new_query_vector
            
        if sparsity_log_file:
            print(f"Writing sparsity log to {sparsity_log_file}")
            with open(sparsity_log_file, "w") as f:
                f.write("query_id\tnon_zero_dimensions\ttotal_dimensions\n")
                for log_line in sparsity_logs:
                    f.write(f"{log_line}\n")
        
        # Re-ranking or second-stage retrieval
        if reuse_step1_retrieval:
            print("Re-ranking step1 retrieval results using the new query vectors...")
            final_results_map = {}
            for q_id, docs in step1_results_map.items():
                new_qvec = new_query_vectors_map[q_id]
                final_docs = []
                for match in docs:
                    doc_vec = match["values"]
                    new_score = float(np.dot(new_qvec, doc_vec))
                    final_docs.append({"id": match["id"], "score": new_score})

                final_docs.sort(key=lambda x: -x["score"])
                final_docs = final_docs[:top_k]
                final_results_map[q_id] = final_docs

            # Produce TREC run lines
            full_trec_run = []
            for q_id, docs in final_results_map.items():
                t_lines = produce_trec_run_lines(q_id, docs)
                full_trec_run.extend(t_lines)
        else:
            # Second-stage FAISS query with sparsemax-modified query vector
            print("Executing second-stage queries (with sparsemax query vectors) against FAISS...")
            final_query_results_map = {}
            for q_id, q_vec in tqdm(new_query_vectors_map.items(), desc="Executing new queries"):
                matches = get_faiss_matches(index, doc_ids, q_vec, top_k=top_k)
                final_query_results_map[q_id] = matches
            print("Done executing second-stage queries.")

            # Produce TREC run lines
            full_trec_run = []
            for q_id, q_result in final_query_results_map.items():
                t_lines = produce_trec_run_lines(q_id, q_result)
                full_trec_run.extend(t_lines)

    # Write TREC file
    print(f"Writing TREC run to {output_trec_path}")
    with open(output_trec_path, "w") as file:
        for line in full_trec_run:
            file.write(f"{line}\n")
    print("Done.")

# Parser setup (기존과 동일하되 alpha 관련 파라미터 제거)
parser = argparse.ArgumentParser(
    description="Script for PRF + sparsemax dimension selection using a local FAISS index",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--model", type=str, required=True, help="Model name or path")
parser.add_argument("--index-dir", type=str, required=True, help="Directory containing the FAISS index and metadata")
parser.add_argument("--ir-ds-query-path", type=str, required=True, help="Path to the dataset file on ir-datasets")
parser.add_argument("--output-trec-name", type=str, required=True, help="Path to save the output TREC run")
parser.add_argument("--top-k", type=int, default=1000, help="Number of results to retrieve")
parser.add_argument("--weighted", action="store_true", help="Use weighted average for centroid computation")
parser.add_argument("--prf-k", type=int, default=False, help="Number of top documents to use for PRF")
parser.add_argument("--reuse-step1-retrieval", action="store_true", help="Reuse the retrieval from step1 and only re-rank")
parser.add_argument("--metric", type=str, default="dot", choices=["dot", "cosine"], help="Metric for computing new scores")
parser.add_argument("--attention-type", type=str, default="linear", choices=["linear", "softmax"], help="Type of attention weighting")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the softmax attention")
parser.add_argument("--llm-docs", type=str, default=None, help="Path to Pandas file containing LLM-generated docs")
parser.add_argument("--normalize_embeddings", action="store_true", help="If specified, normalize the embeddings.")
parser.add_argument("--query_prompt", type=str, default=None, help="Query prompt.")
parser.add_argument("--doc_prompt", type=str, default=None, help="Doc prompt.")
parser.add_argument("--sparsity-log-file", type=str, default=None, help="Path to save the sparsity log")

args = parser.parse_args()

print(f"[info] Loading FAISS index from: {args.index_dir}")
index, doc_ids = load_index_and_metadata(args.index_dir)

print(f"[info] Loading query embedding model: {args.model}")
model = SentenceTransformer(args.model)

if __name__ == "__main__":
    run_queries(
        ir_dataset_queries_path=args.ir_ds_query_path,
        output_trec_path=args.output_trec_name,
        top_k=args.top_k,
        weighted=args.weighted,
        prf_k=args.prf_k,
        reuse_step1_retrieval=args.reuse_step1_retrieval,
        metric=args.metric,
        attention_type=args.attention_type,
        temperature=args.temperature,
        llm_docs=args.llm_docs,
        normalize_embeddings=args.normalize_embeddings,
        query_prompt=args.query_prompt,
        doc_prompt=args.doc_prompt,
        sparsity_log_file=args.sparsity_log_file
    )