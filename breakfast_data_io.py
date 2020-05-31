import numpy as np
from os.path import join as pjoin

###############################################
# Utility functions for reading Breakfast data
###############################################

def print_args(args):
    print('\n'.join(['{}: {}'.format(k,v) for k,v in vars(args).items()]))

def get_breakfast_data(base_path, split):
    print('Reading data...')
    label2index, _ = read_action_idx_mapping(pjoin(base_path, "mapping.txt"))
    video_list = read_nl_file(pjoin(base_path, split))
    data = dict()
    data['video_names'] = video_list
    data['label2index'] = label2index
    data['features'] = read_features(base_path, video_list)
    data['groundtruth'] = read_groundtruth(base_path, video_list, label2index)
    assert( len(data['features']) == len(data['groundtruth']) )
    print('Read features and groundtruth files for {} videos'.format(len(data['features'])))
    return data

def read_nl_file(filename):
    with open(filename, 'r') as f:
        l = f.read().split('\n')[0:-1]
    return l

def read_action_idx_mapping(mapping_file):
    label2index = dict()
    index2label = dict()
    with open(mapping_file, 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            label2index[line.split()[1]] = int(line.split()[0])
            index2label[int(line.split()[0])] = line.split()[1]
    return label2index, index2label


def read_features(base_path, video_list):
    features = []
    # read features for each video
    for video in video_list:
        # video features
        features.append(np.load(base_path + '/features/' + video + '.npy').T)

    return features


def read_groundtruth(base_path, video_list, label2index):
    # read groundtruth for each video
    groundtruth = []
    for video in video_list:
        # transcript
        with open(base_path + '/groundTruth/' + video + '.txt') as f:
            groundtruth.append([label2index[line] for line in f.read().split('\n')[0:-1]])
    return groundtruth