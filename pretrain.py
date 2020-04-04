import numpy as np
from breakfast_data_io import get_breakfast_data, print_args
import argparse
import os
from dsc_model_fc import DSCAutoEncoder


parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, help="Identifier for Experiment", required=True)
parser.add_argument('--nodes', type=str, required=True)
# parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--model_path', default='/home/pegasus/mnt/raptor/ryan/DSC_results')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--max_iter', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=250000)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--split', help="Name of split file (without extension)", default='split1')
args = parser.parse_args()

SPLIT = args.split
DATA_BASE_PATH = '/home/pegasus/mnt/raptor/ryan/breakfast_data_fisher_idt'
RUN_NAME = args.run_name
MODEL_DIR = os.path.join(args.model_path, args.run_name)
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
SAVE_INTERVAL = args.save_interval
INPUT_DIM = 64
N_CLASS = 48 # how many class we sample
N_HIDDEN = [ int(n) for n in args.nodes.split(',') ] # num nodes per layer of encoder (mirrored in decoder)

# TODO - make tensorflow use gpu

print('########################\nPRE TRAIN DEEP SUBSPACE CLUSTERING\n')
print('Architecture {}'.format(N_HIDDEN))
print_args(args)

SAVE_DIR =  os.path.join(MODEL_DIR, 'pretrain')
LOGS_DIR = os.path.join(SAVE_DIR, 'logs')
if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)

print('Gathering data...')
# gather Breakfast data features
train_data = get_breakfast_data(DATA_BASE_PATH, SPLIT + '.train')
features = train_data['features']
features = np.concatenate(features)
# shuffle features
perm = np.arange(len(features))
np.random.shuffle(perm)
features = features[perm]
del train_data, perm

print('Initializing Model...')
AE = DSCAutoEncoder(
    input_dim = INPUT_DIM, 
    n_hidden = N_HIDDEN, 
	learning_rate = LEARNING_RATE,
	save_dir = SAVE_DIR,
    batch_size = BATCH_SIZE, 
    logs_path= LOGS_DIR
)

def next_batch(data, _index_in_epoch ,batch_size , _epochs_completed):
    _num_examples = data.shape[0]
    start = _index_in_epoch
    _index_in_epoch += batch_size
    if _index_in_epoch > _num_examples:
        # Finished epoch
        _epochs_completed += 1
        # Shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        data = data[perm]
        # Start next epoch
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples
    end = _index_in_epoch
    return data[start:end], _index_in_epoch, _epochs_completed

print('Beginning Training...')

it = 0
display_step = 300
_index_in_epoch = 0
_epochs= 0

# train the network
while _epochs < args.max_iter:
    old_ep = _epochs
    batch_x,  _index_in_epoch, _epochs =  next_batch(features, _index_in_epoch , BATCH_SIZE , _epochs)
    if old_ep < _epochs and _epochs % SAVE_INTERVAL == 0:
        AE.save_model(_epochs)
    cost = AE.partial_fit(batch_x)
    it += 1
    avg_cost = cost/(BATCH_SIZE)
    if old_ep < _epochs and _epochs % 5 == 0:
            print("Epoch: {} Index: {}/{} Loss: {}".format(_epochs, _index_in_epoch, len(features), avg_cost )) 

print('Finished')
