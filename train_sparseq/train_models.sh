#!/usr/bin/env bash

# Define an array of model names to train
MODEL_NAMES=(
#   "BAAI/bge-m3"
  "mixedbread-ai/mxbai-embed-large-v1"
#   "intfloat/multilingual-e5-large"
)

# --- Model-specific parameters for training P ---
declare -A TRAIN_QUERY_PROMPTS
TRAIN_QUERY_PROMPTS["intfloat/multilingual-e5-large"]="query: "
TRAIN_QUERY_PROMPTS["mixedbread-ai/mxbai-embed-large-v1"]="Represent this sentence for searching relevant passages: "
TRAIN_QUERY_PROMPTS["BAAI/bge-m3"]=""

declare -A TRAIN_DOC_PROMPTS
TRAIN_DOC_PROMPTS["intfloat/multilingual-e5-large"]="passage: "
TRAIN_DOC_PROMPTS["mixedbread-ai/mxbai-embed-large-v1"]=""
TRAIN_DOC_PROMPTS["BAAI/bge-m3"]=""

# This script should be run from the project root directory
cd "$(dirname "$0")/.." # Move to project root from script location

# Loop over each model to train P
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  echo "==================================================="
  echo "Running training for parameter P for model: ${MODEL_NAME}"
  echo "==================================================="

  # Construct prompt arguments
  QUERY_PROMPT_ARG=""
  if [ -n "${TRAIN_QUERY_PROMPTS[$MODEL_NAME]}" ]; then
    QUERY_PROMPT_ARG="--query_prompt \"${TRAIN_QUERY_PROMPTS[$MODEL_NAME]}\""
  fi
  
  DOC_PROMPT_ARG=""
  if [ -n "${TRAIN_DOC_PROMPTS[$MODEL_NAME]}" ]; then
    DOC_PROMPT_ARG="--doc_prompt \"${TRAIN_DOC_PROMPTS[$MODEL_NAME]}\""
  fi

  eval python train_sparseq/train.py \
    --model_name "${MODEL_NAME}" \
    --output_dir "trained_params" \
    --dataset_path "/data_x/yjoonjang/SPARSEQ/DATA/miracl_train" \
    --epochs 1 \
    --batch_size 32 \
    --lr 1e-4 \
    --device "cuda:5" \
    ${QUERY_PROMPT_ARG} \
    ${DOC_PROMPT_ARG}
    
  echo "---------------------------------------------------"
done

echo "All training finished."
