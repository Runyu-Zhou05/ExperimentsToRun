CUDA_LAUNCH_BLOCKING=1 PYTHONUNBUFFERED=1 BATCH_SIZE_PER_DEVICE=4 MODEL_SIZE=7 accelerate launch --num-processes=4 --gpu_ids 0,1,2,3 vlmbias_grpo_lora.py
