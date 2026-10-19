#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export MODEL_NAME="path/StableAvatar/checkpoints/Wan2.1-Fun-V1.1-1.3B-InP"

accelerate launch --config_file accelerate_config/accelerate_config_machine_1B_multiple.yaml  --machine_rank ${NODE_RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} train_1B_rec_vec.py \
  --config_path="deepspeed_config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_wav2vec_path="path/StableAvatar/checkpoints/wav2vec2-base-960h" \
  --validation_reference_path="path/StableAvatar/validation_singing/reference.png" \
  --validation_driven_audio_path="path/StableAvatar/validation_singing/audio.wav" \
  --train_data_rec_dir="path/StableAvatar/talking_face_data/video_rec_path.txt" \
  --train_data_vec_dir="path/StableAvatar/talking_face_data/video_vec_path.txt"  \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=10000 \
  --checkpointing_steps=500 \
  --validation_steps=500 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="path/StableAvatar/output_1B_rec_vec_dir_64" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --uniform_sampling \
  --motion_sub_loss \
  --low_vram \
  --train_mode="i2v" \
  --resume_from_checkpoint="latest"