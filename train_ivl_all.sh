# EXP_NAME="adprbisusspao/lisa-ivl3-2b_bi2cbe_addsoa8p_vlorati_sr"
EXP_NAME="adprus_ab/ivl3-2b_iadpnrd_vlora_seq_cbs_bi"
# EXP_NAME="adprbisus_nr3_122mlp/ivl3-8b_ss2_2_vlora_ra6-20-8-1_cbs"

CUDA_VISIBLE_DEVICES=4,5,6,7 \
deepspeed --master_port=24995 train_hf_ivl_seq.py \
    --version="/data/LLM/InternVL3-2B" \
    --dataset_dir='/data/anqi/vlmseg/LISA/dataset' \
    --dataset="sem_seg||refer_seg||reason_seg||vqa" \
    --refer_seg_data="refclef||refcoco||refcoco+||refcocog||grefcoco" \
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
    --use_vision_lora \
    # --high_res \
    # --eval_only \
    # --steps_per_epoch 5