CHECKPOINT="//data/LLM/InternVL3-2B"
# WEIGHT=""
# WEIGHT="//home/anqi/code/LISA/runs/adprus_ab/ivl3-2b_iadprn_vlora_seq_cbs_bisca/ckpt_model"
WEIGHT="/data/xiaokang/LISA/runs/SegWeights/ft_1epoch/ckpt_model"

# DATASET="vqa-vqav2-val"
# DATASET="vqa-textvqa-val"
# DATASET="vqa-vizwiz-val"
# DATASET="vqa-okvqa-val"
# DATASET="mme"


# GPUS=4 bash evaluate.sh ${CHECKPOINT} ${DATASET} --dynamic --weight ${WEIGHT} 

# DATASETS=("vqa-vqav2-val" "vqa-textvqa-val" "vqa-vizwiz-val" "vqa-okvqa-val" "mme")
# DATASETS=("vqa-vqav2-val" "vqa-textvqa-val" "vqa-vizwiz-val" "vqa-okvqa-val" "mme" "mmbench-dev-en" "mmbench-dev-cn" "mmbench-test-en" "mmbench-test-cn" "pope" "vqa-gqa-testdev")
# DATASETS=("vqa-vqav2-val" "vqa-textvqa-val" "vqa-vizwiz-val" "vqa-okvqa-val" "pope" "vqa-gqa-testdev")
DATASETS=("mme")

export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_PORT=63666

for DATASET in ${DATASETS[@]}; do
    GPUS=4 bash evaluate.sh ${CHECKPOINT} ${DATASET} --dynamic #--weight ${WEIGHT} 
done