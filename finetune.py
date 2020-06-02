import numpy as np
from breakfast_data_io import get_breakfast_data, print_args
import argparse
import os
from dsc_model_fc import DSCModelFull, thrC, post_proC, err_rate
from progress.bar import Bar
# import ipdb

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, help="Identifier for Experiment", required=True)
parser.add_argument('--data_dir', type=str, help="Data Root Directory", required=True)
parser.add_argument('--load_iter', type=int, required=True)
parser.add_argument('--nodes', type=str, required=True)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--resume_finetune', dest='resume_finetune', action='store_true')
parser.set_defaults(resume_finetune=False)
parser.add_argument('--evaluate', dest='evaluate', action='store_true')
parser.set_defaults(evaluate=False)
# parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--max_iter', type=int, default=120)
parser.add_argument('--batch_size', type=int, default=5000)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--split', help="Name of split file (without extension)", required=True)
args = parser.parse_args()

SPLIT = args.split
DATA_BASE_PATH = args.data_dir
RUN_NAME = args.run_name
RES_DIR = '/home/pegasus/mnt/raptor/ryan/DSC_results'
MODEL_DIR = os.path.join(RES_DIR, args.run_name)
BATCH_SIZE = args.batch_size
MAX_ITER = args.max_iter
LR = args.lr
INPUT_DIM = 64
SS_DIM = 12
N_CLASS = 48 # how many class we sample
N_HIDDEN = [ int(n) for n in args.nodes.split(',') ] # num nodes per layer of encoder (mirrored in decoder)

SAVE_DIR =  os.path.join(MODEL_DIR, 'finetune')
LOGS_DIR = os.path.join(SAVE_DIR, 'logs')

#TODO - use gpu

print('########################\nFINE TUNE DEEP SUBSPACE CLUSTERING\n')
print('Architecture {}'.format(N_HIDDEN))
print_args(args)

model_path = os.path.join(MODEL_DIR, 'finetune' if args.resume_finetune else 'pretrain', 'checkpoint_ep{}.ckpt'.format(str(args.load_iter).zfill(5)))
save_dir = os.path.join(MODEL_DIR, 'finetune')
logs_path = os.path.join(save_dir, 'logs')

# for p in [pretrain_dir, logs_path]:
if not os.path.exists(logs_path):
	os.makedirs(logs_path)

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
    save_dir = SAVE_DIR,
    logs_path= logs_path
)
DSC.restore(model_path)

print('Gathering data...')
# gather Breakfast data features
train_data = get_breakfast_data(DATA_BASE_PATH, SPLIT + '.train')
features = train_data['features']
labels = train_data['groundtruth']
all_features = np.concatenate(features)
all_labels = np.concatenate(labels)

print('#######################')
print('Beginning Training')
print('#######################')

# ipdb.set_trace() # TODO - remove

# def next_batch(data, labels, _index_in_epoch , batch_size , _epochs_completed):
#     _num_examples = data.shape[0]
#     start = _index_in_epoch
#     _index_in_epoch += batch_size
#     if _index_in_epoch > _num_examples:
#         # Finished epoch
#         _epochs_completed += 1
#         # Shuffle the data
#         perm = np.arange(_num_examples)
#         np.random.shuffle(perm)
#         data = data[perm]
#         labels = labels[perm]
#         # Start next epoch
#         start = 0
#         _index_in_epoch = batch_size
#         assert batch_size <= _num_examples
#     end = _index_in_epoch
#     return data[start:end], labels[start:end], _index_in_epoch, _epochs_completed

start_iter = 0 if not args.resume_finetune else args.load_iter + 1
acc_= []

for iter_ft in range(start_iter, MAX_ITER + 1):
    # shuffle features
    perm = np.arange(len(all_features))
    np.random.shuffle(perm)
    all_features = all_features[perm]
    all_labels = all_labels[perm]
    costs_l1 = []
    costs_l2 = []
    for batch_idx in Bar('Iteration #{}'.format(iter_ft)).iter(range(0, len(all_features), BATCH_SIZE)):
        batch_features = all_features[batch_idx : batch_idx + BATCH_SIZE ]
        if len(batch_features) != BATCH_SIZE:
            continue
        batch_labels = all_labels[batch_idx : batch_idx + BATCH_SIZE ]
        C,l1_cost,l2_cost = DSC.finetune_fit(batch_features, LR)
        costs_l1.append(l1_cost)
        costs_l2.append(l2_cost)
        # print(batch_idx, l1_cost, l2_cost)
        if (args.evaluate and batch_idx == 0): # TODO - change to non debug value
            print("epoch: %.1d" % iter_ft, "cost: %.8f" % (l1_cost/float(len(batch_features))))
            C = thrC(C,alpha)			
            y_x, CKSym_x = post_proC(C, N_CLASS, SS_DIM , 8)		
            # ipdb.set_trace()	
            missrate_x = err_rate(batch_labels, y_x)			
            acc = 1 - missrate_x
            print ("Epoch: %d" % iter_ft,"acc: %.4f" % acc)
    print("Iteration #{} avg costs l1: {} l2: {}".format(iter_ft, np.mean(costs_l1), np.mean(costs_l2)))
    if iter_ft % args.save_interval == 0:
        print('SAVING MODEL epoch {}'.format(iter_ft))
        DSC.save_model(iter_ft)

