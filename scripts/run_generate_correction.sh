#!/bin/bash

export PYTHONPATH="../../chem-diff:${PYTHONPATH:-}"
MODEL_NAME=$1
# dir of MODEL_NAME

DIFFUSION_STEPS=${2:-2000}

GEN=$3


OUT_DIR=${4}

if [ -z "$OUT_DIR" ]; then
    OUT_DIR=${MODEL_NAME}
fi

BATCH_SIZE=${5:-100}
TOP_P=${6:-0.9}
CLAMP=${7:-no_clamp}
SEQ_LEN=${8:-10}
SEED=${9:-10708}

python -u src/train_infer/generate_smiles_correction.py --model_name_or_path ${MODEL_NAME} \
--batch_size ${BATCH_SIZE} \
--gen_path ${GEN} \
--top_p ${TOP_P} \
--seed ${SEED} \
--out_dir ${OUT_DIR} --diffusion_steps ${DIFFUSION_STEPS} --clamp ${CLAMP} --sequence_len ${SEQ_LEN}
