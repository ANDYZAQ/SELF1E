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
    --version="***/InternVL3-2B" \
    --dataset_dir='./dataset' \
    --exp_name="***/your_exp_name" \
    --eval_only \
    --auto_resume \
    --val_dataset="${val_dataset}" 
done

