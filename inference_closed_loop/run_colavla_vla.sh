#!/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

cd "$BASE_DIR" || exit 1

conda activate colavla

export PYTHONPATH="$(pwd):${PYTHONPATH}"

NUSCENES_PATH="${NUSCENES_PATH:-$BASE_DIR/neuro-ncap/data/nuscenes}"
MODEL_NAME='colavla'
MODEL_CONFIG_PATH=$1
MODEL_CHECKPOINT_PATH=$2
MODLE_TYPE='vla'

RENDERING_FOLDER="$BASE_DIR/neurad-studio"
RENDERING_CHECKPOINTS_PATH='checkpoints'
NCAP_FOLDER="$BASE_DIR/neuro-ncap"
RUNS=${3:-10}

TIME_NOW=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/colavla_evaluation_${TIME_NOW}.log"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

if [ "$PWD" != "$BASE_DIR" ]; then
    log_message "Please run this script from the ColaVLA folder: $BASE_DIR"
    log_message "Current directory: $PWD"
    exit 1
fi

if [ ! -d "$RENDERING_FOLDER" ]; then
    log_message "Rendering folder not found: $RENDERING_FOLDER"
    exit 1
fi

if [ ! -d "$NCAP_FOLDER" ]; then
    log_message "NCAP folder not found: $NCAP_FOLDER"
    exit 1
fi

if [ ! -f "$MODEL_CONFIG_PATH" ]; then
    log_message "Model config not found: $MODEL_CONFIG_PATH"
    exit 1
fi

log_message "============================================================"
log_message "🚗 ColaVLA NeuroNCAP Evaluation"
log_message "============================================================"
log_message "Base directory: $BASE_DIR"
log_message "Model config: $MODEL_CONFIG_PATH"
log_message "Model checkpoint: $MODEL_CHECKPOINT_PATH"
log_message "Conda environment: $CONDA_ENV_PATH"
log_message "============================================================"

for SCENARIO in "stationary" "frontal" "side"; do
    SCENARIO_LOG_DIR="$LOG_DIR/${SCENARIO}_${TIME_NOW}"
    mkdir -p "$SCENARIO_LOG_DIR"
    log_message "Created scenario log directory: $SCENARIO_LOG_DIR"

    array_file="ncap_slurm_array_${SCENARIO}"
    id_to_seq="$NCAP_FOLDER/scripts/arrays/${array_file}.txt"

    if [ ! -f "$id_to_seq" ]; then
        log_message "Warning: Array file not found: $id_to_seq"
        log_message "Skipping scenario: $SCENARIO"
        continue
    fi

    if [ "$SCENARIO" == "stationary" ]; then
        num_scenarios=10
    elif [ "$SCENARIO" == "frontal" ]; then
        num_scenarios=5
    else
        num_scenarios=5
    fi

    log_message "Running scenario: $SCENARIO ($num_scenarios sequences)"

    for i in $(seq 1 "$num_scenarios"); do
        sequence=$(awk -v ArrayTaskID="$i" '$1==ArrayTaskID {print $2}' "$id_to_seq")
        if [ -z "$sequence" ]; then
            log_message "Warning: Undefined sequence for task $i in $SCENARIO"
            continue
        fi

        log_message "Running sequence: $sequence"

        export BASE_DIR="$BASE_DIR"
        export NUSCENES_PATH="$NUSCENES_PATH"
        export MODEL_NAME="$MODEL_NAME"
        export MODEL_CONFIG_PATH="$MODEL_CONFIG_PATH"
        export MODEL_CHECKPOINT_PATH="$MODEL_CHECKPOINT_PATH"
        export RENDERING_FOLDER="$RENDERING_FOLDER"
        export RENDERING_CHECKPOINTS_PATH="$RENDERING_CHECKPOINTS_PATH"
        export NCAP_FOLDER="$NCAP_FOLDER"
        export CONDA_ENV_PATH="$CONDA_ENV_PATH"
        export TIME_NOW="$TIME_NOW"

        log_message "Starting evaluation for sequence: $sequence"

        sequence_log_file="$SCENARIO_LOG_DIR/sequence_${sequence}.log"

        bash "$BASE_DIR/inference_closed_loop/run_local_colavla.sh" "$sequence" "$SCENARIO" "$MODLE_TYPE" --scenario-category="$SCENARIO" --runs "$RUNS" 2>&1 | tee -a "$LOG_FILE" | tee "$sequence_log_file"

        exit_status=${PIPESTATUS[0]}
        if [ "$exit_status" -eq 0 ]; then
            log_message "✅ Sequence $sequence completed successfully"
            log_message "Sequence log saved to: $sequence_log_file"
        else
            log_message "❌ Sequence $sequence failed with exit status $exit_status"
            log_message "Sequence log saved to: $sequence_log_file"
        fi
    done
done

log_message "Evaluation completed!"
log_message "Log file saved to: $LOG_FILE"
