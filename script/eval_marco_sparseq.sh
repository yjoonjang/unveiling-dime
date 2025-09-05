#!/usr/bin/env bash

MODEL_NAMES=(
#   "intfloat/multilingual-e5-large"
"BAAI/bge-m3"
  # Add other models as needed
)

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

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  for DATASET_ID in "${DATASET_IDS[@]}"; do 
    for QUERY_ID in "${QUERY_IDS[@]}"; do
      SAFE_MODEL_NAME=${MODEL_NAME//\//_}
      SAFE_DATASET_ID=${DATASET_ID//\//_}
      SAFE_QUERY_NAME=${QUERY_ID//\//_}
      
      QRELS_DIR="qrels/${DATASET_ID}/${SAFE_QUERY_NAME}"
      mkdir -p ${QRELS_DIR}
      QRELS_FILE="${QRELS_DIR}/qrels.tsv"
      
      if [ ! -f "${QRELS_FILE}" ]; then
        echo "Generating qrels for: ${QUERY_ID}"
        ir_datasets export ${QUERY_ID} qrels --format trec > "${QRELS_FILE}"
      fi
      
      TREC_DIR="runs_sparseq_parameter/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/${SAFE_QUERY_NAME}"
      
      # Evaluate baseline
      BASELINE_FILE="${TREC_DIR}/baseline.trec"
      if [ -f "${BASELINE_FILE}" ]; then
        echo "Running eval for: ${BASELINE_FILE}"
        trec_eval -m ndcg_cut.10 "${QRELS_FILE}" "${BASELINE_FILE}"
        echo ""
      else
        echo "Warning: Baseline file not found: ${BASELINE_FILE}"
      fi
      
      # Evaluate sparseq results for each ALPHA
      for ALPHA in "${ALPHAS[@]}"; do
        SPARSEQ_FILE="${TREC_DIR}/sparseq_parameter_alpha${ALPHA}.trec"
        LOG_FILE="${TREC_DIR}/logs/sparseq_parameter_alpha${ALPHA}.log"
        
        if [ -f "${SPARSEQ_FILE}" ]; then
          echo "Running eval for: ${SPARSEQ_FILE}"
          trec_eval -m ndcg_cut.10 "${QRELS_FILE}" "${SPARSEQ_FILE}"
          
          if [ -f "${LOG_FILE}" ]; then
            AVG_DIMS=$(awk 'NR > 1 {sum += $2} END {if (NR > 1) printf "%.2f", sum/(NR-1); else print "0"}' "${LOG_FILE}")
            echo "  Average active dimensions: ${AVG_DIMS}"
          fi
          echo ""
        else
          echo "Warning: Sparseq_parameter file not found: ${SPARSEQ_FILE}"
        fi
      done
    done
  done
done

echo "Evaluation completed!"
