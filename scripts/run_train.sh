#!/bin/bash
set -u

export PYTHONPATH="../../chem-diff:${PYTHONPATH:-}"

DSET=${1:-moses2}
GPU=${2:-0}
AUGMENT_PROB=${3:-0}  # New variable at position 3
INIT_PRETRAINED_MODEL=${4:-"False"}
USE_PRETRAINED_EMBEDDINGS=${5:-"False"}
FREEZE_EMBEDDINGS=${6:-"False"}

LR_ANNEAL_STEPS=${7:-25001}
LR=${8:-0.0001}
DIFFUSION_STEPS=${9:-2000}
NOISE_SCHEDULE=${10:-sqrt}
BATCH_SIZE=${11:-64}
SEQ_LEN=${12:-100}

CHECKPOINT_PATH=${13:-"ckpts/${DSET}_AUGMENT_PROB-${AUGMENT_PROB}"}
TRAIN_TXT_PATH=${14:-data/gdb13/gdb13.smi}
VAL_TXT_PATH=${15:-"no"}
IN_CHANNELS=${16:-128}
WEIGHT_DECAY=${17:-0.0}
SEED=${18:-10708}
DROPOUT=${19:-0.1}
NUM_HEADS=${20:-4}
CONFIG_NAME=${21:-"seyonec/SMILES_tokenized_PubChem_shard00_160k"}

NOTES=${22:-"Pre-trained models, pre-trained embeddings, embeddings not frozen"}

mkdir -p ${CHECKPOINT_PATH}

ARGS=(--checkpoint_path ${CHECKPOINT_PATH}
    --save_interval 50000 --lr ${LR}
    --batch_size ${BATCH_SIZE}
    --diffusion_steps ${DIFFUSION_STEPS}
    --noise_schedule ${NOISE_SCHEDULE}
    --sequence_len ${SEQ_LEN} --seed ${SEED}
    --dropout ${DROPOUT} --in_channel ${IN_CHANNELS}
    --out_channel ${IN_CHANNELS}
    --weight_decay ${WEIGHT_DECAY}
    --predict_xstart True
    --train_txt_path ${TRAIN_TXT_PATH}
    --dataset ${DSET}
    --val_txt_path ${VAL_TXT_PATH}
    --num_heads ${NUM_HEADS}
    --config_name ${CONFIG_NAME}
    --init_pretrained ${INIT_PRETRAINED_MODEL}
    --freeze_embeddings ${FREEZE_EMBEDDINGS}
    --use_pretrained_embeddings ${USE_PRETRAINED_EMBEDDINGS}
    --augment_prob ${AUGMENT_PROB}  # Add the variable to ARGS
    --notes \""${NOTES}"\")

if [ ${LR_ANNEAL_STEPS} -eq 0 ]; then
    LR_ANNEAL_STEPS=100
    DEBUG=true
else
    DEBUG=false
fi

ARGS+=(--lr_anneal_steps $LR_ANNEAL_STEPS)

if [ $DEBUG = true ]; then
    ARGS+=(--debug)
fi

export CUDA_VISIBLE_DEVICES=$GPU && python -u src/train_infer/train_chem.py "${ARGS[@]}"
