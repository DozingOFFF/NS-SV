#!/usr/bin/env python
# encoding: utf-8

from tqdm import tqdm
import numpy as np
from .compute_eer import compute_eer
from .tuneThreshold import *
from numba import jit


dict_domain = {
    0:'advertisement',
    1:'drama',
    2:'entertainment',
    3:'interview',
    4:'live_broadcast',
    5:'movie',
    6:'play',
    7:'recitation',
    8:'singing',
    9:'speech',
    10:'vlog'

}


def cosine_score(trials, scores, index_mapping, eval_vectors, apply_metric=True):
    all_scores = []
    all_labels = []
    target_scores = []
    nontarget_scores = []
    
    f = open(scores, 'w')
    for item in trials:
        all_labels.append(int(item[0]))
        enroll_vector = eval_vectors[index_mapping[item[1]]]
        test_vector = eval_vectors[index_mapping[item[2]]]
        dim = len(enroll_vector)
        score = enroll_vector.dot(test_vector.T)
        norm = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
        score = dim * score / norm
        all_scores.append(score)
        f.write(item[0] + ' ' + item[1] + ' ' + item[2] + ' ' + str(score) + '\n')
    f.close()

    if apply_metric:
        eer, th = compute_eer(all_scores, all_labels)

        c_miss = 1
        c_fa = 1
        fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
        mindcf_easy, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, c_miss, c_fa)
        mindcf_hard, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.001, c_miss, c_fa)
        return eer, th, mindcf_easy, mindcf_hard


def PLDA_score(trials, scores, index_mapping, eval_vectors, plda_analyzer, apply_metric=True):
    all_scores = []
    all_labels = []
    target_scores = []
    nontarget_scores= []

    f = open(scores, 'w')
    for item in trials:
        all_labels.append(int(item[0]))
        enroll_vector = eval_vectors[index_mapping[item[1]]]
        test_vector = eval_vectors[index_mapping[item[2]]]
        score = plda_analyzer.NLScore(enroll_vector, test_vector)
        all_scores.append(score)
        f.write(item[0] + ' ' + item[1] + ' ' + item[2] + ' ' + str(score) + '\n')
    f.close()

    if apply_metric:
        eer, th = compute_eer(all_scores, all_labels)

        c_miss = 1
        c_fa = 1
        fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
        mindcf_easy, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, c_miss, c_fa)
        mindcf_hard, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.001, c_miss, c_fa)
        return eer, th, mindcf_easy, mindcf_hard
