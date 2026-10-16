#!/usr/bin/env bash
set -euo pipefail

# Optional arguments: <gpu_id> <model_name> <report_to> <do_save>
GPU_ID="${1:-${GPU_ID:-0}}"
MODEL="${2:-${MODEL:-Llama-2-7b-chat-hf}}"
REPORTTO="${3:-${REPORTTO:-wandb}}"
DO_SAVE="${4:-${DO_SAVE:-True}}"
WANDB_PROJECT="BalDRO"
TRAINER="SatImp"
PRETRAINED_PATH="open-unlearning/tofu_Llama-2-7b-chat-hf_full"

SPLITS=(
    "forget05 holdout05 retain95"
    "forget01 holdout01 retain99"
    "forget10 holdout10 retain90"
)
LR_SET=("1e-5" "2e-5" "5e-5")
BZ_SET=("4 8")
EPOCH_SET=(10)
ALPHA_SET=(1.0 0.1 0.01)
BETA1_SET=(5.0 6.0)
BETA2_SET=(1.0)

for SPLIT_CONFIG in "${SPLITS[@]}"; do
    for LR in "${LR_SET[@]}"; do
        for BZ_CONFIG in "${BZ_SET[@]}"; do
            for EPOCHS in "${EPOCH_SET[@]}"; do
                for ALPHA in "${ALPHA_SET[@]}"; do
                    for BETA1 in "${BETA1_SET[@]}"; do
                        for BETA2 in "${BETA2_SET[@]}"; do
                            read -r FORGET_SPLIT HOLDOUT_SPLIT RETAIN_SPLIT <<< "${SPLIT_CONFIG}"
                            read -r BATCH_SIZE GRAD_ACC <<< "${BZ_CONFIG}"

                            RUN_DATE=$(date "+%Y%m%d")
                            TIMESTAMP=$(date "+%H%M%S")
                            RUN_SUFFIX="lr${LR}_b${BATCH_SIZE}_ga${GRAD_ACC}_a${ALPHA}_1b${BETA1}_2b${BETA2}_e${EPOCHS}_${RUN_DATE}_${TIMESTAMP}"
                            TASK_NAME="unlearn_tofu_${MODEL}_${FORGET_SPLIT}_${TRAINER}_${RUN_SUFFIX}"
                            OUTPUT_DIR="./saves/unlearn/tofu/${FORGET_SPLIT}/${MODEL}/${TRAINER}/${RUN_SUFFIX}"
                            mkdir -p "${OUTPUT_DIR}/logs"

                            echo "[START] ${TASK_NAME}"
                            echo "GPU_ID=${GPU_ID} MODEL=${MODEL} SPLIT=${FORGET_SPLIT}/${HOLDOUT_SPLIT}/${RETAIN_SPLIT}"
                            export CUDA_VISIBLE_DEVICES="${GPU_ID}"
                            export WANDB_PROJECT="${WANDB_PROJECT}"
                            python src/train.py --config-name=unlearn.yaml \
                                experiment=unlearn/tofu/default \
                                trainer="${TRAINER}" \
                                model="${MODEL}" \
                                model.model_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
                                model.tokenizer_args.pretrained_model_name_or_path="${PRETRAINED_PATH}" \
                                forget_split="${FORGET_SPLIT}" \
                                holdout_split="${HOLDOUT_SPLIT}" \
                                retain_split="${RETAIN_SPLIT}" \
                                task_name="${TASK_NAME}" \
                                paths.output_dir="${OUTPUT_DIR}" \
                                do_save="${DO_SAVE}" \
                                eval.tofu.retain_logs_path="./saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json" \
                                trainer.args.ddp_find_unused_parameters=true \
                                trainer.args.gradient_checkpointing=true \
                                trainer.args.report_to="${REPORTTO}" \
                                trainer.args.run_name="${TASK_NAME}" \
                                trainer.args.logging_steps=1 \
                                trainer.args.learning_rate="${LR}" \
                                trainer.args.per_device_train_batch_size="${BATCH_SIZE}" \
                                trainer.args.gradient_accumulation_steps="${GRAD_ACC}" \
                                trainer.args.num_train_epochs="${EPOCHS}" \
                                trainer.args.eval_strategy=epoch \
                                trainer.args.eval_on_start=False \
                                trainer.method_args.beta1="${BETA1}" \
                                trainer.method_args.beta2="${BETA2}" \
                                trainer.method_args.alpha="${ALPHA}" \
                                trainer.method_args.gamma=0.1 \
                                trainer.method_args.retain_loss_type=NLL
                            echo "[END] ${TASK_NAME}"
                        done
                    done
                done
            done
        done
    done
done
