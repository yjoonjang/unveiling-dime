#!/usr/bin/env bash
BATCH_SIZE=800000

# Define an array of model names
MODEL_NAMES=(
#   "BAAI/bge-m3"
  "mixedbread-ai/mxbai-embed-large-v1"
#   "intfloat/multilingual-e5-large"
)

# --- Model-specific parameters for generation and querying ---
declare -A GENERATE_EMBEDDINGS_PARAMS
GENERATE_EMBEDDINGS_PARAMS["intfloat/multilingual-e5-large"]="--normalize_embeddings --doc_prompt \"passage: \""
GENERATE_EMBEDDINGS_PARAMS["mixedbread-ai/mxbai-embed-large-v1"]="--normalize_embeddings"
GENERATE_EMBEDDINGS_PARAMS["BAAI/bge-m3"]="--normalize_embeddings"

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
  "1.7"
  "1.8"
  "1.9"
  "2.0"
  "2.1"
)

# Loop over each model and dataset
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  SAFE_MODEL_NAME=${MODEL_NAME//\//_}

  # This script assumes parameter P has been trained separately by train_sparseq/train_models.sh
  echo "Running retrieval for model: ${MODEL_NAME}"
  echo "---------------------------------------------------"

  for DATASET_ID in "${DATASET_IDS[@]}"; do
    # Embedding generation and indexing steps are assumed to be done.
    # If not, the original logic from run_marco.sh can be re-inserted here.

    for QUERY_ID in "${QUERY_IDS[@]}"; do
      SAFE_DATASET_ID=${DATASET_ID//\//_}
      SAFE_QUERY_NAME=${QUERY_ID//\//_}
      
      TREC_DIR="runs_sparseq_parameter/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/${SAFE_QUERY_NAME}/"
      mkdir -p "${TREC_DIR}/logs"

      INDEX_DIR="output/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/index/"
      P_PARAM_PATH="trained_params/${SAFE_MODEL_NAME}_P.pt"
      
      if [ ! -f "${P_PARAM_PATH}" ]; then
        echo "Error: Trained parameter file not found at ${P_PARAM_PATH}"
        echo "Please run train_sparseq/train_models.sh first."
        continue
      fi
      
      PARAMS="${QUERY_PARAMS[$MODEL_NAME]}"

      # --- Baseline query ---
      BASELINE_TREC_FILE="${TREC_DIR}/baseline.trec"
      echo "Running baseline query..."
      eval python tool/query_sparseq_parameter.py \
        --ir-ds-query-path "${QUERY_ID}" \
        --output-trec-name "${BASELINE_TREC_FILE}" \
        --index-dir "${INDEX_DIR}" \
        --p_param_path "${P_PARAM_PATH}" \
        --alpha 1 \
        --model "${MODEL_NAME}" \
        ${PARAMS}

      # --- Queries with trained P and sparsemax ---
      for ALPHA in "${ALPHAS[@]}"; do
        TREC_FILE="${TREC_DIR}/sparseq_parameter_alpha${ALPHA}.trec"
        LOG_FILE="${TREC_DIR}/logs/sparseq_parameter_alpha${ALPHA}.log"
        
        echo "Running sparseq_parameter query with alpha: ${ALPHA}"
        
        eval python tool/query_sparseq_parameter.py \
          --ir-ds-query-path "${QUERY_ID}" \
          --output-trec-name "${TREC_FILE}" \
          --sparsity-log-file "${LOG_FILE}" \
          --index-dir "${INDEX_DIR}" \
          --p_param_path "${P_PARAM_PATH}" \
          --alpha ${ALPHA} \
          --model "${MODEL_NAME}" \
          ${PARAMS}
      done
    done
  done
done
