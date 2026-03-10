EXP_NAME="***/your_exp_name"

CUDA_VISIBLE_DEVICES=4,5,6,7 \
deepspeed --master_port=24995 train_hf_ivl_seq.py \
    --version="***/InternVL3-2B" \
    --dataset_dir='./dataset' \
    --dataset="sem_seg||refer_seg||reason_seg||vqa" \
    --sample_rates="1,1,1,1" \
    --batch_size 5 \
    --grad_accumulation_steps 8 \
    --gradient_checkpointing \
    --exp_name="${EXP_NAME}" \
    --model_max_length 512 \
    --explanatory -1 \
    --lora_r 128 \
    --lora_alpha 256 \
    --epochs 1 \
    --lr 1e-4 \
    --vision_lr 1e-4 \
    --optimize_vision \
    --use_llm_lora \
    --use_vision_lora 