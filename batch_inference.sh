RUN_NAME=$1
STRUCTURE=$2

MODEL_DIR=$3
DATA_DIR=$4

GT_DIR="${DATA_DIR}/groundTruth/"

for i in 1 2 3 4
do
    RUN="${RUN_NAME}_sp${i}"
    python3 pretrain.py --run_name=$RUN --nodes=$STRUCTURE --split="split${i}" --data_dir=$DATA_DIR
    python3 finetune.py --run_name=$RUN --nodes=$STRUCTURE --split="split${i}" --data_dir=$DATA_DIR --load_iter=10000
    python3 eval.py --run_name=$RUN --nodes=$STRUCTURE --split="split${i}" --load_iter=120 --data_dir=$DATA_DIR
    echo " ------- finished inference of ${RUN_NAME} split ${i} --------"
done

for i in 1 2 3 4
do
    FULL_RUN_NAME="${MODEL_DIR}/${RUN_NAME}_sp${i}/segmentation"
    python3 full_eval.py  --ground_truth_dir=$GT_DIR --recog_dir=$FULL_RUN_NAME --exclude_bg
    echo " ------- finished evaluation of ${RUN_NAME} split ${i} --------"
done