#!/usr/bin/env bash

MODEL_NAMES=(
  "BAAI/bge-m3"
  "mixedbread-ai/mxbai-embed-large-v1"
  "intfloat/multilingual-e5-large"
)

declare -A QUERY_PARAMS
QUERY_PARAMS["intfloat/multilingual-e5-large"]="--normalize_embeddings --query_prompt \"query: \""
QUERY_PARAMS["mixedbread-ai/mxbai-embed-large-v1"]="--normalize_embeddings  --query_prompt \"Represent this sentence for searching relevant passages: \""
QUERY_PARAMS["BAAI/bge-m3"]="--normalize_embeddings"

DATASET_IDS=(
  "msmarco-passage"
)

QUERY_IDS=(
  "msmarco-passage/trec-dl-2019/judged"
  "msmarco-passage/trec-dl-2020/judged"
  "msmarco-passage/trec-dl-hard"
)

ALPHAS=(
  "1.5" 
  "1.6" 
#   "1.7" 
#   "1.8" 
#   "1.9" 
  "2.0" 
)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  SAFE_MODEL_NAME=${MODEL_NAME//\//_}

  echo "Running Adapter retrieval for model: ${MODEL_NAME}"
  echo "---------------------------------------------------"

  for DATASET_ID in "${DATASET_IDS[@]}"; do
    for QUERY_ID in "${QUERY_IDS[@]}"; do
      SAFE_DATASET_ID=${DATASET_ID//\//_}
      SAFE_QUERY_NAME=${QUERY_ID//\//_}
      
      TREC_DIR="runs_sparseq_adapter/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/${SAFE_QUERY_NAME}/"
      mkdir -p "${TREC_DIR}/logs"

      INDEX_DIR="output/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/index/"
      ADAPTER_PATH="trained_adapters/${SAFE_MODEL_NAME}_adapter.pt"
      
      if [ ! -f "${ADAPTER_PATH}" ]; then
        echo "Error: Trained adapter file not found at ${ADAPTER_PATH}"
        echo "Please run train_sparseq/train_models_adapter.sh first."
        continue 2 # Continue outer loops
      fi
      
      PARAMS="${QUERY_PARAMS[$MODEL_NAME]}"

      # --- Baseline is not run here, assuming it exists from other runs ---

      # --- Queries with trained Adapter ---
      for ALPHA in "${ALPHAS[@]}"; do
        TREC_FILE="${TREC_DIR}/sparseq_adapter_alpha${ALPHA}.trec"
        LOG_FILE="${TREC_DIR}/logs/sparseq_adapter_alpha${ALPHA}.log"
        
        echo "Running sparseq_adapter query with alpha: ${ALPHA}"
        
        eval python tool/query_sparseq_adapter.py \
          --model "${MODEL_NAME}" \
          --adapter_path "${ADAPTER_PATH}" \
          --index-dir "${INDEX_DIR}" \
          --ir-ds-query-path "${QUERY_ID}" \
          --output-trec-name "${TREC_FILE}" \
          --sparsity-log-file "${LOG_FILE}" \
          --alpha ${ALPHA} \
          ${PARAMS}
      done
    done
  done
done
