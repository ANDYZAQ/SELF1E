# # List all val datasets
val_datasets=(
    # "ReasonSeg|test"
    # "refcoco|unc|val"
    # "refcoco|unc|testA"
    # "refcoco|unc|testB"
    # "refcoco+|unc|val"
    # "refcoco+|unc|testA"
    # "refcoco+|unc|testB"
    # "refcocog|umd|test"
    # "refcocog|umd|val"
    # "grefcoco|unc|val"
    "grefcoco|unc|testA"
    # "grefcoco|unc|testB"
)

for val_dataset in "${val_datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  deepspeed --master_port=25999 train_hf_ivl_seq.py \
    --version="/bask/projects/j/jiaoj-generative/anqi/weights/InternVL3-8B" \
    --dataset_dir='/bask/homes/a/axz464/code/LISA/dataset' \
    --exp_name="adprbisus_nr3_122mlp/ivl3-8b_ss2_2_vlora_ra6-20-6-110_cbs" \
    --eval_only \
    --auto_resume \
    --val_dataset="${val_dataset}" 
done

# CUDA_VISIBLE_DEVICES=0,1 \
# deepspeed --master_port=25999 train_hf_ivl.py \
#   --version="/bask/projects/j/jiaoj-generative/anqi/weights/InternVL3-2B" \
#   --dataset_dir='/bask/homes/a/axz464/code/LISA/dataset' \
#   --exp_name="adprbisus/lisa-ivl3-2b_bi2cbe_vlorati_sr" \
#   --eval_only \
#   --eval_all_datasets \
#   # --eval_visualize \
