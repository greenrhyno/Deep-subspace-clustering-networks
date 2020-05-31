import numpy as np
from breakfast_data_io import get_breakfast_data, print_args
import argparse
import os
from dsc_model_fc import DSCModelFull, thrC, post_proC, err_rate

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, help="Identifier for Experiment", required=True)
parser.add_argument('--load_iter', type=int, required=True)
parser.add_argument('--nodes', type=str, required=True)
parser.add_argument('--data_dir', type=str, help="Data Root Directory", default='/home/pegasus/mnt/raptor/zijia/unsup_pl/dataset/Hollywood')
parser.add_argument('--batch_size', type=int, default=5000)
parser.add_argument('--split', help="Name of split file (without extension)", required=True)
args = parser.parse_args()

SPLIT = args.split
DATA_BASE_PATH = args.data_dir
RUN_NAME = args.run_name
RES_DIR = '/home/pegasus/mnt/raptor/ryan/DSC_results'
MODEL_DIR = os.path.join(RES_DIR, args.run_name)
BATCH_SIZE = args.batch_size
INPUT_DIM = 64
SS_DIM = 12
N_CLASS = 48 # how many class we sample
N_HIDDEN = [ int(n) for n in args.nodes.split(',') ] # num nodes per layer of encoder (mirrored in decoder)

SAVE_DIR =  os.path.join(MODEL_DIR, 'finetune')
SAVE_LABEL_DIR = os.path.join(MODEL_DIR, 'segmentation')
LOGS_DIR = os.path.join(SAVE_DIR, 'logs')

print('########################\nFINE TUNE DEEP SUBSPACE CLUSTERING\n')
print('Architecture {}'.format(N_HIDDEN))
print_args(args)

model_path = os.path.join(MODEL_DIR, 'finetune', 'checkpoint_ep{}.ckpt'.format(str(args.load_iter).zfill(5)))
# logs_path = os.path.join(save_dir, 'logs')
# if not os.path.exists(logs_path):
# 	os.makedirs(logs_path)

alpha = 0.04
reg1 = 1.0
reg2 = 30.0

print('#######################')
print('Initializing Model')
print('#######################')
DSC = DSCModelFull(
    input_dim = INPUT_DIM, 
    n_hidden = N_HIDDEN, 
    reg_const1 = reg1, 
    reg_const2 = reg2,
    batch_size = BATCH_SIZE, 
    save_dir = None,
    logs_path= None
)
DSC.restore(model_path)

# def write_predicted_labels(labels, video_name):
#     with open(os.path.join(SAVE_LABEL_DIR, video_name), 'w') as f:
#         f.write( ' '.join( [index2label[l] for l in labels] ) + '\n' )

print('Gathering data...')
# gather Breakfast data features
test_data = get_breakfast_data(DATA_BASE_PATH, SPLIT + '.test')
names = test_data['video_names']
features = test_data['features']
labels = test_data['groundtruth']
label2index = test_data['label2index']
all_features = np.concatenate(features)
all_labels = np.concatenate(labels)

print('#######################')
print('Beginning Evaluation')
print('#######################')

all_acc= []

# shuffle features
perm = np.arange(len(all_features))
np.random.shuffle(perm)
all_features = all_features[perm]
all_labels = all_labels[perm]
num_batches = len(all_features) // BATCH_SIZE

for batch_idx in range(0, len(all_features), BATCH_SIZE):
    try:
        batch_features = all_features[batch_idx : batch_idx + BATCH_SIZE ]
        if len(batch_features) != BATCH_SIZE:
            continue
        batch_labels = all_labels[batch_idx : batch_idx + BATCH_SIZE ]
        C = DSC.fit(batch_features)
        C = thrC(C,alpha)			
        y_x, CKSym_x = post_proC(C, N_CLASS, SS_DIM , 8)		
        missrate_x = err_rate(batch_labels, y_x)			
        acc = 1 - missrate_x
        all_acc.append(acc)
        print("Batch #{}/{} -- Accuracy: {:0.3%}".format(batch_idx, num_batches, acc))
    except:
        print("ERROR Skipping Batch #{}".format(batch_idx))
    
all_acc = np.array(all_acc)
avg_acc = np.mean(all_acc)
med_acc = np.median(all_acc)

print('######## Finished #########')
print('Final Accuracy -- Mean: {} Median: {}'.format(avg_acc, med_acc))