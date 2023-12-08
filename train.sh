export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="./output"

accelerate launch train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="laion/dalle-3-dataset" \
 --image_column="image" \
 --caption_column="caption" \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --validation_image "./palette1.png" "./palette2.png" "./palette3.png"   \
 --validation_prompt "tree house on the beach" "cat warrior" "screaming dogs illustration" \
 --validation_steps=50 \
 --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
 --train_batch_size=16 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42 
