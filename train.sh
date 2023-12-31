export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="./output"

accelerate launch train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="GabrielVidal/dalle-3-palette" \
 --image_column="image" \
 --caption_column="text" \
 --conditioning_image_column="palette" \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --validation_image "./palette1.png" "./palette2.png" "./palette3.png"   \
 --validation_prompt "tree house on the beach" "cat warrior" "screaming dogs illustration" \
 --validation_steps=50 \
 --num_validation_images=2 \
 --train_batch_size=16 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --tracker_project_name="controlnet-palette" \
 --seed=42 \
 --push_to_hub
