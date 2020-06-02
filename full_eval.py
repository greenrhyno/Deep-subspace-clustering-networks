#!/usr/bin/python

import argparse
import glob
import re
import numpy as np
from collections import defaultdict
import ipdb

def recog_file(filename, ground_truth_path, stats, exclude_bg=False):

    # read ground truth
    gt_file = ground_truth_path + re.sub('.*/','/',filename) + '.txt'
    with open(gt_file, 'r') as f:
        ground_truth = f.read().split('\n')[0:-1]
        f.close()
    # read recognized sequence
    with open(filename, 'r') as f:
        recognized = f.read().split(' ')
        # recognized =  # framelevel recognition is in 6-th line of file
        f.close()

    # print('recognized', len(recognized), 'groundtruth', len(ground_truth))

    n_frame_errors = 0
    for i in range(len(recognized)):
        if ground_truth[i] == "SIL" and exclude_bg:
            continue
        if not recognized[i] == ground_truth[i]:
            n_frame_errors += 1

    ground_truth = np.array(ground_truth)
    recognized = np.array(recognized)

    unique = set(np.unique(ground_truth)).union(set(np.unique(recognized)))
    for i in unique:
        if exclude_bg and i == "SIL":
            continue

        recog_mask = recognized == i
        gt_mask = ground_truth == i
        union = np.logical_or(recog_mask, gt_mask).sum()
        intersect = np.logical_and(recog_mask, gt_mask).sum() # num of correct prediction

        stats[i][0] = stats[i][0] + intersect
        stats[i][1] = stats[i][1] + recog_mask.sum()
        stats[i][2] = stats[i][2] + gt_mask.sum()
        stats[i][3] = stats[i][3] + union

    return n_frame_errors, len(recognized)


### MAIN #######################################################################

### arguments ###
### --recog_dir: the directory where the recognition files from inferency.py are placed
### --ground_truth_dir: the directory where the framelevel ground truth can be found
parser = argparse.ArgumentParser()
parser.add_argument('--recog_dir', required=True)
parser.add_argument('--ground_truth_dir', required=True)
parser.add_argument('--exclude_bg', action="store_true")
args = parser.parse_args()

filelist = glob.glob(args.recog_dir + '/*')
# filelist = [ f for f in filelist if "action" not in f ]

print('RECOG', args.recog_dir, '\nGT', args.ground_truth_dir)

print('Evaluate %d video files...' % len(filelist))

n_frames = 0
n_errors = 0
stats = defaultdict(lambda : [0, 0, 0, 0])
# loop over all recognition files and evaluate the frame error
for filename in filelist:
    errors, frames = recog_file(filename, args.ground_truth_dir, stats, args.exclude_bg)
    n_errors += errors
    n_frames += frames


P = np.array([ s[0] / s[1] for s in stats.values() ])
R = np.array([ s[0] / s[2] for s in stats.values() ])
P[np.isnan(P)] = 0
R[np.isnan(R)] = 0
F = (2*P*R)/(P+R+1e-6)
Jaccard = [ s[0] / s[3] for s in stats.values() ]
print('frame accuracy: %f' % (1.0 - float(n_errors) / n_frames))
print('number total frames: %d' % (n_frames))
print("P %.4f R %.4f F1 %.4f" % (np.mean(P), np.mean(R), np.mean(F)))
print("Jaccard %.4f" % np.mean(Jaccard) )

