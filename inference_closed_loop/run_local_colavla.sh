#!/bin/bash

seq=${1:?"No sequence specified"}
output_name=${2:?"No output name given (for logging)"}
model_type=${3:?"No model type given"}

SHOULD_START_RENDERER=true
SHOULD_START_MODEL=true

for arg in ${@:4}; do
  if [[ $arg == "--spoof-renderer" || $arg == "--spoof_renderer" ]]; then
    SHOULD_START_RENDERER=false
  fi
  if [[ $arg == "--spoof-model" || $arg == "--spoof_model" ]]; then
    SHOULD_START_MODEL=false
  fi
done

find_free_port() {
  python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

renderer_port=$(find_free_port)
model_port=$(find_free_port)

echo "============================================================"
echo "🚗 ColaVLA Local Runner"
echo "============================================================"
echo "Sequence: $seq"
echo "Output name: $output_name"
echo "Renderer port: $renderer_port"
echo "Model port: $model_port"
echo "============================================================"

echo "Activating conda environment..."
source /nfs/dataset-ofs-voyager-research/pqh/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV_PATH

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_MODE=eager
export TORCH_COMPILE=0
export TORCH_LOGS="dynamo"

if [ $SHOULD_START_RENDERER == true ]; then
  echo "Starting NeuRAD service..."

  cd $RENDERING_FOLDER
  python nerfstudio/scripts/closed_loop/main.py \
    --port $renderer_port \
    --load-config $RENDERING_CHECKPOINTS_PATH/$seq/config.yml \
    --adjust_pose \
    $RENDERER_ARGS \
    &

  RENDERER_PID=$!
  echo "Renderer service started with PID $RENDERER_PID"
  cd $BASE_DIR
fi

if [ $SHOULD_START_MODEL == true ]; then
  echo "Starting ColaVLA service..."

  ENABLE_VIS=${ENABLE_VIS:-"true"}
  VIS_OUTPUT_DIR=${VIS_OUTPUT_DIR:-"output/$TIME_NOW"}
  VIS_SAVE_IMAGES=${VIS_SAVE_IMAGES:-"true"}
  VIS_SAVE_TRAJECTORIES=${VIS_SAVE_TRAJECTORIES:-"true"}
  VIS_SAVE_CALIBRATION=${VIS_SAVE_CALIBRATION:-"true"}
  VIS_CREATE_OVERLAYS=${VIS_CREATE_OVERLAYS:-"false"}

  VIS_ARGS=""
  if [ "$ENABLE_VIS" == "true" ]; then
    VIS_ARGS="--enable-vis --vis-output-dir $VIS_OUTPUT_DIR --vis-scenario $output_name --vis-sequence $seq"
    VIS_ARGS="$VIS_ARGS --vis-save-images $VIS_SAVE_IMAGES --vis-save-trajectories $VIS_SAVE_TRAJECTORIES"
    VIS_ARGS="$VIS_ARGS --vis-save-calibration $VIS_SAVE_CALIBRATION --vis-create-overlays $VIS_CREATE_OVERLAYS"
    echo "Visualization enabled:"
    echo "  Output dir: $VIS_OUTPUT_DIR/$output_name-$seq"
    echo "  Save images: $VIS_SAVE_IMAGES"
    echo "  Save trajectories: $VIS_SAVE_TRAJECTORIES"
    echo "  Save calibration: $VIS_SAVE_CALIBRATION"
    echo "  Create overlays: $VIS_CREATE_OVERLAYS"
  fi

  python inference_closed_loop/server.py \
    --config_path $MODEL_CONFIG_PATH \
    --checkpoint_path $MODEL_CHECKPOINT_PATH \
    --port $model_port \
    --device cuda:0 \
    $MODEL_ARGS \
    --model_type $model_type \
    $VIS_ARGS \
    > server.log 2>&1 &

  MODEL_PID=$!
  echo "Model service started with PID $MODEL_PID"

  echo "Waiting for server to start (model loading may take several minutes)..."
  max_wait_minutes=30
  wait_interval=10

  for i in $(seq 1 $((max_wait_minutes * 60 / wait_interval))); do
    if curl -s http://localhost:$model_port/alive > /dev/null 2>&1; then
      echo "✅ Server is responding on port $model_port after $((i * wait_interval)) seconds"
      break
    fi

    if [ $((i % 6)) -eq 0 ]; then
      minutes_waited=$((i * wait_interval / 60))
      echo "⏳ Still waiting for server... ($minutes_waited minutes elapsed)"
    else
      echo "Waiting for server... ($i/$((max_wait_minutes * 60 / wait_interval)))"
    fi

    sleep $wait_interval
  done

  if ! curl -s http://localhost:$model_port/alive > /dev/null 2>&1; then
    echo "❌ Server failed to start within $max_wait_minutes minutes"
    echo "Server log (last 50 lines):"
    tail -50 server.log
    echo ""
    echo "Full server log saved to: server.log"
    exit 1
  fi
fi

sleep 5

echo "Running neuro-ncap evaluation..."

cd $NCAP_FOLDER

python main.py \
  --engine.renderer.port $renderer_port \
  --engine.model.port $model_port \
  --engine.dataset.data_root $NUSCENES_PATH \
  --engine.dataset.version v1.0-trainval \
  --engine.dataset.sequence $seq \
  --engine.logger.log-dir output/$TIME_NOW/$output_name-$seq \
  ${@:4}

echo "Evaluation completed for sequence $seq"

echo "Cleaning up background processes..."
if [ $SHOULD_START_RENDERER == true ] && [ ! -z $RENDERER_PID ]; then
  echo "Killing renderer service (PID: $RENDERER_PID)"
  kill $RENDERER_PID 2>/dev/null || true
fi

if [ $SHOULD_START_MODEL == true ] && [ ! -z $MODEL_PID ]; then
  echo "Killing model service (PID: $MODEL_PID)"
  kill $MODEL_PID 2>/dev/null || true
fi

cd $BASE_DIR

echo "Cleanup completed!"
