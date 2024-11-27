# -*- coding:utf-8 -*-
# Copyright THU-CSTL

import os
import sys
import argparse
import numpy as np
from compute_eer import compute_eer
from tuneThreshold import *
from tqdm import tqdm


def compute_trial_config_self(trials, scale, output):
    f = open(trials, 'r')
    lines = f.readlines()
    f.close()

    target_scores = []
    nontarget_scores = []
    for line in lines:
        line = line.strip().split()
        if line[0] == '1':
            target_scores.append(float(line[3]))
        else:
            nontarget_scores.append(float(line[3]))

    target_size = len(target_scores)
    nontarget_size = len(nontarget_scores)
    target_scores = sorted(target_scores, reverse=False)
    nontarget_scores = sorted(nontarget_scores, reverse=True)
    # print(target_size, nontarget_size)

    f = open(output, 'w')
    f.write("tar_pos nontar_pos EER minDCF\n")
    for nontarget_scale in tqdm(range(1, scale+1)):
        nontarget_pos = int(nontarget_size * nontarget_scale/scale)
        nontarget_scores_pos = nontarget_scores[:nontarget_pos]
        nontarget_label_pos = [0] * nontarget_pos
        for target_scale in range(1, scale+1):
            target_pos = int(target_size * target_scale/scale)
            # print(nontarget_pos, target_pos)
            target_scores_pos = target_scores[:target_pos]
            target_label_pos = [1] * target_pos
            all_scores = nontarget_scores_pos + target_scores_pos
            all_labels = nontarget_label_pos + target_label_pos

            eer, th = compute_eer(all_scores, all_labels)
            c_miss = 1
            c_fa = 1
            fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
            mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, c_miss, c_fa)
            f.write("{} {} {:.5f} {:.5f}\n".format(target_pos, nontarget_pos, eer*100, mindcf))
    f.close()


def compute_trial_config_cmp(reference, test, scale, output):
    f_r = open(reference, 'r')
    lines_r = f_r.readlines()
    f_r.close()

    f_t = open(test, 'r')
    lines_t = f_t.readlines()
    f_t.close()

    assert len(lines_r) == len(lines_t)

    target_scores = {}
    nontarget_scores = {}
    for line in lines_r:
        line = line.strip().split()
        key = line[1] + '-' + line[2]
        if line[0] == '1':
            target_scores[key] = [float(line[3])]
        else:
            nontarget_scores[key] = [float(line[3])]

    for line in lines_t:
        line = line.strip().split()
        key = line[1] + '-' + line[2]
        if line[0] == '1':
            target_scores[key].append(float(line[3]))
        else:
            nontarget_scores[key].append(float(line[3]))

    target_scores = sorted(target_scores.items(), key = lambda kv:(kv[1], kv[0]), reverse=False)
    target_size = len(target_scores)
    nontarget_scores = sorted(nontarget_scores.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    nontarget_size = len(nontarget_scores)

    f = open(output, 'w')
    f.write("tar_pos nontar_pos EER minDCF\n")
    for nontarget_scale in tqdm(range(1, scale+1)):
        nontarget_pos = int(nontarget_size * nontarget_scale/scale)
        nontarget_scores_pos = [ i[1][1] for i in nontarget_scores[:nontarget_pos] ]
        nontarget_label_pos = [0] * nontarget_pos
        for target_scale in range(1, scale+1):
            target_pos = int(target_size * target_scale/scale)
            target_scores_pos = [ i[1][1] for i in target_scores[:target_pos] ]
            target_label_pos = [1] * target_pos
            all_scores = nontarget_scores_pos + target_scores_pos
            all_labels = nontarget_label_pos + target_label_pos
       
            eer, th = compute_eer(all_scores, all_labels)
            c_miss = 1
            c_fa = 1
            fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
            mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, c_miss, c_fa)
            f.write("{} {} {:.5f} {:.5f}\n".format(target_pos, nontarget_pos, eer*100, mindcf))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='mode of trial config, 0:self | 1:comp', type=int, default=0)
    parser.add_argument('--input_scores_ref', help='path of the reference scores', type=str, default="")
    parser.add_argument('--input_scores_test', help='path of the test scores', type=str, default="score.foo")
    parser.add_argument('--scale', help='the scale of trial config', type=int, default=20)
    parser.add_argument('--output', help='path of the output', type=str, default="log")
    args = parser.parse_args()

    if args.mode == 0:
        compute_trial_config_self(args.input_scores_test, args.scale, args.output)
    else:
        compute_trial_config_cmp(args.input_scores_ref, args.input_scores_test, args.scale, args.output)

