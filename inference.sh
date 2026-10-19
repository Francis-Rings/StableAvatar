#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TOKENIZERS_PARALLELISM=false
export MODEL_NAME="$SCRIPT_DIR/checkpoints/Wan2.1-Fun-V1.1-1.3B-InP"

CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config_path="deepspeed_config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path="$SCRIPT_DIR/checkpoints/StableAvatar-1.3B/transformer3d-square.pt" \
  --pretrained_wav2vec_path="$SCRIPT_DIR/checkpoints/wav2vec2-base-960h" \
  --validation_reference_path="$SCRIPT_DIR/examples/case-1/reference.png" \
  --validation_driven_audio_path="$SCRIPT_DIR/examples/case-1/audio.wav" \
  --output_dir="$SCRIPT_DIR/output_infer" \
  --validation_prompts="A middle-aged woman with short light brown hair, wearing pearl earrings and a blue blazer, is speaking passionately in front of a blurred background resembling a government building. Her mouth is open mid-phrase, her expression is engaged and energetic, and the lighting is bright and even, suggesting a television interview or live broadcast. The scene gives the impression she is singing with conviction and purpose." \
  --seed=42 \
  --ulysses_degree=1 \
  --ring_degree=1 \
  --motion_frame=25 \
  --sample_steps=50 \
  --width=512 \
  --height=512 \
  --overlap_window_length=5 \
  --clip_sample_n_frames=81 \
  --GPU_memory_mode="model_full_load" \
  --sample_text_guide_scale=3.0 \
  --sample_audio_guide_scale=5.0
