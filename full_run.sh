RUN_NAME=$1
STRUCTURE=$2
MODEL_DIR=$3
DATA_DIR=$4
i=$5 # split number

GT_DIR="${DATA_DIR}/groundTruth/"

RUN="${RUN_NAME}_sp${i}"
# python3 pretrain.py --run_name=$RUN --nodes=$STRUCTURE --split="split${i}" --data_dir=$DATA_DIR --max_iter=10000
# python3 finetune.py --run_name=$RUN --nodes=$STRUCTURE --split="split${i}" --data_dir=$DATA_DIR --load_iter=10000
python3 eval.py --run_name=$RUN --nodes=$STRUCTURE --split="split${i}" --load_iter=120 --data_dir=$DATA_DIR
echo " ------- finished inference of ${RUN_NAME} split ${i} --------"


RECOG="${MODEL_DIR}/${RUN_NAME}_sp${i}/segmentation"
python3 full_eval.py  --ground_truth_dir=$GT_DIR --recog_dir=$RECOG --exclude_bg
echo " ------- finished evaluation of ${RUN_NAME} split ${i} --------"