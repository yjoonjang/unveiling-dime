#!/usr/bin/env bash
BATCH_SIZE=800000

# Define an array of model names
MODEL_NAMES=(
  # M3
  BAAI/bge-m3
  # mxbai
  mixedbread-ai/mxbai-embed-large-v1
  # E5  
  "intfloat/multilingual-e5-large"
  # Snowflake
#   "Snowflake/snowflake-arctic-embed-l-v2.0"
#   # ANCE
#   "sentence-transformers/msmarco-roberta-base-ance-firstp"
#   # Contriever
#   "facebook/contriever-msmarco"
#   # TAS-B
#   "sentence-transformers/msmarco-distilbert-base-tas-b"
)

# Define model-specific parameters
declare -A GENERATE_EMBEDDINGS_PARAMS
GENERATE_EMBEDDINGS_PARAMS["intfloat/multilingual-e5-large"]="--normalize_embeddings --doc_prompt \"passage: \""
GENERATE_EMBEDDINGS_PARAMS["mixedbread-ai/mxbai-embed-large-v1"]="--normalize_embeddings"
GENERATE_EMBEDDINGS_PARAMS["nomic-ai/nomic-embed-text-v1.5"]="--normalize_embeddings"
GENERATE_EMBEDDINGS_PARAMS["BAAI/bge-multilingual-gemma2"]="--normalize_embeddings"
GENERATE_EMBEDDINGS_PARAMS["BAAI/bge-m3"]="--normalize_embeddings"

declare -A QUERY_PARAMS
QUERY_PARAMS["intfloat/multilingual-e5-large"]="--normalize_embeddings --query_prompt \"query: \" --doc_prompt \"passage: \""
QUERY_PARAMS["mixedbread-ai/mxbai-embed-large-v1"]="--normalize_embeddings  --query_prompt \"Represent this sentence for searching relevant passages: \""
QUERY_PARAMS["Snowflake/snowflake-arctic-embed-l-v2.0"]="--query_prompt \"query: \""
QUERY_PARAMS["nomic-ai/nomic-embed-text-v1.5"]="--normalize_embeddings"
QUERY_PARAMS["BAAI/bge-multilingual-gemma2"]="--normalize_embeddings --query-prompt /home/amallia/repro-dime/bge-gemma2.prompt"
QUERY_PARAMS["BAAI/bge-m3"]="--normalize_embeddings"

DATASET_IDS=(
  "msmarco-passage"
)

QUERY_IDS=(
  "msmarco-passage/trec-dl-2019/judged"
  "msmarco-passage/trec-dl-2020/judged"
  "msmarco-passage/trec-dl-hard"
)

# Only PRF_KS needed (no ZERO_DIMS for sparsemax)
PRF_KS=(
  "1"
  "2"
  "5"
  "10"
)

# Loop over each model and dataset
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  for DATASET_ID in "${DATASET_IDS[@]}"; do

    PARAMS="${GENERATE_EMBEDDINGS_PARAMS[$MODEL_NAME]}"

    echo "Running embeddings generation with:"
    echo "  Model Name: ${MODEL_NAME}"
    echo "  Dataset ID: ${DATASET_ID}"
    echo "  Parameters: ${PARAMS}"

    if [[ "$*" != *"-skip-embed"* ]]; then
      eval python tool/generate_embeddings.py \
        --model_name "${MODEL_NAME}" \
        --dataset_id "${DATASET_ID}" \
        --batch_size "${BATCH_SIZE}" \
        --flush_size 1000000 \
        --output_dir "output" \
        ${PARAMS}
    else
      echo "Skipping embeddings generation (-skip-embed flag detected)"
    fi

    echo "---------------------------------------------------"

    echo "Running FAISS indexing with:"
    echo "  Model Name: ${MODEL_NAME}"
    echo "  Dataset ID: ${DATASET_ID}"
    SAFE_MODEL_NAME=${MODEL_NAME//\//_}
    SAFE_DATASET_ID=${DATASET_ID//\//_}

    INPUT_DIR="output/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/"
    OUTPUT_DIR="output/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/index/"

    if [[ "$*" != *"-skip-index"* ]]; then
      eval python tool/index_embeddings.py \
        --input_dir "${INPUT_DIR}" \
        --output_dir "${OUTPUT_DIR}"
    else
      echo "Skipping FAISS indexing (-skip-index flag detected)"
    fi
    echo "---------------------------------------------------"
    
    for QUERY_ID in "${QUERY_IDS[@]}"; do
      SAFE_QUERY_NAME=${QUERY_ID//\//_}
      
      TREC_DIR="runs_sparsemax_test/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/${SAFE_QUERY_NAME}/"
      
      mkdir -p "${TREC_DIR}"

      # Baseline query (no PRF)
      TREC_FILE="${TREC_DIR}/baseline.trec"
      PARAMS="${QUERY_PARAMS[$MODEL_NAME]}"

      echo "Running baseline query with:"
      echo "  PRF_K: 0 (no PRF)"
      echo "  Parameters: ${PARAMS}"

      eval python tool/query_sparsemax.py \
        --ir-ds-query-path "${QUERY_ID}" \
        --output-trec-name "${TREC_FILE}" \
        --index-dir "${OUTPUT_DIR}" \
        --prf-k 0 \
        --top-k 10 \
        --model ${MODEL_NAME} \
        ${PARAMS}

      # PRF + Sparsemax queries
      for PRF_K in "${PRF_KS[@]}"; do

        TREC_FILE="${TREC_DIR}/sparsemax_@${PRF_K}.trec"
        LOG_FILE="${TREC_DIR}/sparsemax_@${PRF_K}.log"
        echo "Running sparsemax query with:"
        echo "  PRF_K: ${PRF_K}"
        echo "  Parameters: ${PARAMS}"
        
        eval python tool/query_sparsemax.py \
          --ir-ds-query-path "${QUERY_ID}" \
          --output-trec-name "${TREC_FILE}" \
          --sparsity-log-file "${LOG_FILE}" \
          --index-dir "${OUTPUT_DIR}" \
          --prf-k ${PRF_K} \
          --top-k 10 \
          --model ${MODEL_NAME} \
          ${PARAMS}
      done
    done
  done
done