# Place the base model weight path here
CHECKPOINT="***/InternVL3-2B"
# Place your weight path here
WEIGHT=""

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