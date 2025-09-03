#!/usr/bin/env bash

# Define an array of model names (matching run_marco_sparseq.sh)
MODEL_NAMES=(
  # M3
#   BAAI/bge-m3
  # mxbai
#   mixedbread-ai/mxbai-embed-large-v1
#   # E5  
  "intfloat/multilingual-e5-large"
  # Commented out models (matching run_marco_sparseq.sh)
  # "Snowflake/snowflake-arctic-embed-l-v2.0"
  # "sentence-transformers/msmarco-roberta-base-ance-firstp"
  # "facebook/contriever-msmarco"
  # "sentence-transformers/msmarco-distilbert-base-tas-b"
)

DATASET_IDS=(
  "msmarco-passage"
)

QUERY_IDS=(
  "msmarco-passage/trec-dl-2019/judged"
  "msmarco-passage/trec-dl-2020/judged"
  "msmarco-passage/trec-dl-hard"
)

# PRF_K values for sparsemax (matching run_marco_sparseq.sh)
PRF_KS=(
  "1"
  "2"
  "5"
  "10"
)

# Loop over each model and dataset
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  for DATASET_ID in "${DATASET_IDS[@]}"; do 
    for QUERY_ID in "${QUERY_IDS[@]}"; do
      SAFE_MODEL_NAME=${MODEL_NAME//\//_}
      SAFE_DATASET_ID=${DATASET_ID//\//_}
      SAFE_QUERY_NAME=${QUERY_ID//\//_}
      
      # Create qrels directory and file
      QRELS_DIR="qrels/${DATASET_ID}/${SAFE_QUERY_NAME}"
      mkdir -p ${QRELS_DIR}
      
      QRELS_FILE="${QRELS_DIR}/qrels.tsv"
      
      # Generate qrels file if it doesn't exist
      if [ ! -f "${QRELS_FILE}" ]; then
        echo "Generating qrels for: ${QUERY_ID}"
        ir_datasets export ${QUERY_ID} qrels --format trec > "${QRELS_FILE}"
      fi
      
      # Evaluate baseline (no PRF)
      TREC_DIR="runs_sparsemax_test/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/${SAFE_QUERY_NAME}"
      BASELINE_FILE="${TREC_DIR}/baseline.trec"
      
      if [ -f "${BASELINE_FILE}" ]; then
        echo "Running eval for: ${BASELINE_FILE}"
        trec_eval -m ndcg_cut.10 "${QRELS_FILE}" "${BASELINE_FILE}"
        echo ""
      else
        echo "Warning: Baseline file not found: ${BASELINE_FILE}"
      fi
      
      # Evaluate sparsemax results for each PRF_K
      for PRF_K in "${PRF_KS[@]}"; do
        SPARSEMAX_FILE="${TREC_DIR}/sparsemax_@${PRF_K}.trec"
        LOG_FILE="${TREC_DIR}/sparsemax_@${PRF_K}.log"
        
        if [ -f "${SPARSEMAX_FILE}" ]; then
          echo "Running eval for: ${SPARSEMAX_FILE}"
          trec_eval -m ndcg_cut.10 "${QRELS_FILE}" "${SPARSEMAX_FILE}"
          echo ""
        else
          echo "Warning: Sparsemax file not found: ${SPARSEMAX_FILE}"
        fi
      done
    done
  done
done

echo "Evaluation completed!"