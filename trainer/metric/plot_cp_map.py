# -*- coding:utf-8 -*-
# Copyright THU-CSTL

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_cp_map(data, metric, scale, savedir):
    assert metric == 'eer' or metric == 'dcf'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    data = np.loadtxt(data, dtype=float, skiprows=1)

    fig = plt.figure(figsize=(20,15))
    np.random.seed(123456789)
    d = []
    for i in range (0,int(len(data)/scale)):
        row = data[scale*i:scale*i+scale]
        d.append(row)

    if metric == 'eer':
        plt.imshow(d,cmap='jet',aspect='auto',origin='lower',extent=(0, 100, 0, 100),vmin=0,vmax=100)
    else:
        plt.imshow(d,cmap='jet',aspect='auto',origin='lower',extent=(0, 100, 0, 100),vmin=0,vmax=1)

    plt.xlabel('Percentage of the lowest target scores (%)', fontsize=40)
    plt.xticks(fontsize=40)
    plt.ylabel('Percentage of the highest nontarget scores (%)', fontsize=40)
    plt.yticks(fontsize=40)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=40)

    if metric == 'eer':
        cb.set_ticks([0, 20, 40, 60, 80, 100])
        cb.set_label('EER(%)',fontsize=40)
        plt.tight_layout()
        plt.savefig(savedir + os.sep + 'cpmap_eer.png', dpi=300)
    else:
        cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cb.set_label('minDCF',fontsize=40)
        plt.tight_layout()
        plt.savefig(savedir + os.sep + 'cpmap_dcf.png', dpi=300)


def plot_delta_cp_map(ref, test, metric, scale, savedir):
    assert metric == 'eer' or metric == 'dcf'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    ref_data = np.loadtxt(ref, dtype=float, skiprows=1)
    test_data = np.loadtxt(test, dtype=float, skiprows=1)
    data = (ref_data - test_data)/ref_data

    print("Win: {:.3f}%".format(len([i for i in data if i>0])/len(data)*100))
    print("Tie: {:.3f}%".format(len([i for i in data if np.abs(i)<=1e-6])/len(data)*100))
    print("Lose: {:.3f}%".format(len([i for i in data if i<0])/len(data)*100))

    fig = plt.figure(figsize=(20,15))
    np.random.seed(123456789)
    d = []
    for i in range (0,int(len(data)/scale)):
        row = data[scale*i:scale*i+scale]
        d.append(row)

    plt.imshow(d,cmap='jet',aspect='auto',origin='lower',extent=(0, 100, 0, 100),vmin=-0.5,vmax=0.5)
    plt.xlabel('Percentage of the lowest target scores (%)', fontsize=40)
    plt.xticks(fontsize=40)
    plt.ylabel('Percentage of the highest nontarget scores (%)', fontsize=40)
    plt.yticks(fontsize=40)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=40)
    cb.set_ticks([-0.50, -0.25, 0.00, 0.25, 0.50])

    if metric == 'eer':
        cb.set_label('RCR with EER',fontsize=40)
        plt.tight_layout()
        plt.savefig(savedir + os.sep + 'delta_cpmap_eer.png', dpi=300)
    else:
        cb.set_label('RCR with minDCF',fontsize=40)
        plt.tight_layout()
        plt.savefig(savedir + os.sep + 'delta_cpmap_dcf.png', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='mode of cp-map, 0:cp-map | 1:delta cp-map', type=int, default=0)
    parser.add_argument('--metric', help='metric of cp-map, 0:EER | 1:minDCF', type=str, default="eer")
    parser.add_argument('--input_configs_ref', help='path of the reference trial configs', type=str, default="")
    parser.add_argument('--input_configs_test', help='path of the test trial configs', type=str, default="configs.foo")
    parser.add_argument('--scale', help='the scale of trial config', type=int, default=20)
    parser.add_argument('--savedir', help='save dir', type=str, default="cpmap/")
    args = parser.parse_args()

    if args.mode == 0:
        plot_cp_map(args.input_configs_test, args.metric, args.scale, args.savedir)
    else:
        plot_delta_cp_map(args.input_configs_ref, args.input_configs_test, args.metric, args.scale, args.savedir)

