import argparse
import glob
import json
import math
import os
import pandas as pd
from collections import defaultdict
from collections import Counter
import pickle
import numpy as np


def read_pose(kp_file):
    with open(kp_file) as kf:
        value = json.loads(kf.read())
        kps = value['people'][0]['pose_keypoints_2d']
        x = kps[0::3]
        y = kps[1::3]
        return np.stack((x, y), axis=1)


def calc_pose_flow(prev, next):
    result = np.zeros_like(prev)
    for kpi in range(prev.shape[0]):
        if np.count_nonzero(prev[kpi]) == 0 or np.count_nonzero(next[kpi]) == 0:
            result[kpi, 0] = 0.0
            result[kpi, 1] = 0.0
            continue

        ang = math.atan2(next[kpi, 1] - prev[kpi, 1], next[kpi, 0] - prev[kpi, 0])
        mag = np.linalg.norm(next[kpi] - prev[kpi])

        result[kpi, 0] = ang
        result[kpi, 1] = mag

    return result


def impute_missing_keypoints(poses):
    """Replace missing keypoints (on the origin) by values from neighbouring frames."""
    # 1. Collect missing keypoints
    missing_keypoints = defaultdict(list)  # frame index -> keypoint indices that are missing
    for i in range(poses.shape[0]):
        for kpi in range(poses.shape[1]):
            if np.count_nonzero(poses[i, kpi]) == 0:  # Missing keypoint at (0, 0)
                missing_keypoints[i].append(kpi)
    # 2. Impute them
    for i in missing_keypoints.keys():
        missing = missing_keypoints[i]
        for kpi in missing:
            # Possible replacements
            candidates = poses[:, kpi]
            min_dist = np.inf
            replacement = -1
            for f in range(candidates.shape[0]):
                if f != i and np.count_nonzero(candidates[f]) > 0:
                    distance = abs(f - i)
                    if distance < min_dist:
                        min_dist = distance
                        replacement = f
            # Replace
            if replacement > -1:
                poses[i, kpi] = poses[replacement, kpi]
    # 3. We have imputed as many keypoints as possible with the closest non-missing temporal neighbours
    return poses


def normalize(poses):
    """Normalize each pose in the array to account for camera position. We normalize
    by dividing keypoints by a factor such that the length of the neck becomes 1."""
    for i in range(poses.shape[0]):
        upper_neck = poses[i, 17]
        head_top = poses[i, 18]
        neck_length = np.linalg.norm(upper_neck - head_top)
        poses[i] /= neck_length
        assert math.isclose(np.linalg.norm(upper_neck - head_top), 1)
    return poses

def reshapeKP(kp):
    x = []
    y = []
    for i in range(len(kp)):
        if(i % 2 == 0):
            x.append(kp[i])
        else:
            y.append(kp[i])
    return np.stack((x, y), axis=1)

def main(args):
    
    with open(args.input_dir+"/readyToRun/"+'X.data', 'rb') as f:
        X = pickle.load(f)
    with open(args.input_dir+"/readyToRun/"+'Y.data', 'rb') as f:
        Y = pickle.load(f)
    with open(args.input_dir+"/readyToRun/"+'Y_meaning.data', 'rb') as f:
        y_meaning = pickle.load(f)
        
    output_dir = args.input_dir+'/kpflow2/'
    os.makedirs(output_dir, exist_ok=True)

    input_dir_index = 0
    total = len(X)

    
    for ind in X:
        print(f'{input_dir_index}/{total}')
        input_dir_index += 1

        with open(args.input_dir+'/keypoints/'+str(ind)+'.pkl', 'rb') as f:
            keypoints = pickle.load(f)

        # 1. Collect all keypoint files and pre-process them
        poses = []
        for i in range(len(keypoints)):
            poses.append(reshapeKP(keypoints[i]))
        poses = np.stack(poses)

        #poses = impute_missing_keypoints(poses)
        #poses = normalize(poses)

        # import matplotlib.pyplot as plt
        # for i in range(poses.shape[0]):
        #     plt.figure()
        #     plt.scatter(poses[i, :, 0], -poses[i, :, 1])
        #     plt.show()
        # break

        # 2. Compute pose flow
        prev = poses[0]
        flows = []
        for i in range(1, poses.shape[0]):
            next = poses[i]
            flow = calc_pose_flow(prev, next)
            flows.append(flow)
            prev = next

        np.save(os.path.join(output_dir, 'flow_{:d}'.format(ind)), flows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    args = parser.parse_args()
    main(args)
