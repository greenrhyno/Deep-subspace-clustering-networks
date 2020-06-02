import numpy as np
from breakfast_data_io import get_breakfast_data, print_args
import argparse
import os
from dsc_model_fc import DSCModelFull, thrC, post_proC, err_rate

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, help="Identifier for Experiment", required=True)
parser.add_argument('--load_iter', type=int, required=True)
parser.add_argument('--nodes', type=str, required=True)
parser.add_argument('--data_dir', type=str, help="Data Root Directory", required=True)
parser.add_argument('--batch_size', type=int, default=5000)
parser.add_argument('--no_save', dest='save_labels', action='store_false')
parser.set_defaults(save_labels=True)
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

if not os.path.exists(SAVE_LABEL_DIR):
	os.makedirs(SAVE_LABEL_DIR)

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

print('Gathering data...')
# gather Breakfast data features
test_data = get_breakfast_data(DATA_BASE_PATH, SPLIT + '.test')
names = test_data['video_names']
features = test_data['features']
labels = test_data['groundtruth']
index2label = test_data['index2label']
all_features = np.concatenate(features)
all_labels = np.concatenate(labels)

def write_predicted_labels(labels, video_name):
    with open(os.path.join(SAVE_LABEL_DIR, video_name), 'w') as f:
        f.write( ' '.join( [index2label[int(l)] for l in labels] ) + '\n' )

print('#######################')
print('Beginning Evaluation')
print('#######################')

all_acc= []
num_batches = len(all_features) // BATCH_SIZE + 1

all_output_labels = np.array([])
batch_idx = 0 
last = False
while batch_idx < len(all_features):
    batch_features = all_features[ batch_idx : batch_idx + BATCH_SIZE ]
    batch_labels = all_labels[batch_idx : batch_idx + BATCH_SIZE ]
    if len(batch_features) < BATCH_SIZE:
        last = True
        partial_batch_len = len(batch_features)
        batch_features = np.concatenate([batch_features, all_features[0 : BATCH_SIZE - partial_batch_len]])
        batch_labels = np.concatenate([batch_labels, all_labels[0 : BATCH_SIZE - partial_batch_len]])
    C = DSC.fit(batch_features)
    C = thrC(C,alpha)			
    y_x, CKSym_x = post_proC(C, N_CLASS, SS_DIM, 8)		
    missrate_x, mapped_output = err_rate(batch_labels, y_x)
    acc = 1 - missrate_x
    all_acc.append(acc)
    print("Batch #{}/{} -- Batch Accuracy: {:0.3%}".format(batch_idx // BATCH_SIZE, num_batches, acc))
    if last:
        all_output_labels = np.concatenate([all_output_labels, mapped_output[ : partial_batch_len]])
    else:
        all_output_labels = np.concatenate([all_output_labels, mapped_output])
    batch_idx = batch_idx + BATCH_SIZE # increment

all_acc = np.array(all_acc)
avg_acc = np.mean(all_acc)
med_acc = np.median(all_acc)
print('Final Accuracy -- Mean: {} Median: {}'.format(avg_acc, med_acc))

assert(len(all_output_labels) == len(all_features))

if args.save_labels:
    print("\nWriting segmentation labels to {}...".format(SAVE_LABEL_DIR))
    current_idx = 0
    for idx, video_name in enumerate(names):
        vid_length = len(features[idx])
        next_start_idx = current_idx + vid_length
        # if (next_start_idx <= len(all_output_labels)):
        write_predicted_labels(all_output_labels[current_idx : next_start_idx], video_name)
        current_idx = next_start_idx


print('######## Finished #########')
