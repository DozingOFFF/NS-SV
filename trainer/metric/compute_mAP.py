# -*- coding:utf-8 -*-
# Copyright THU-CSTL

import os
import sys
import argparse

# Load metadata
def load_meta_map(metadata_dir):
    enroll_meta_path = os.path.join(metadata_dir, 'enroll.meta')
    enroll_meta = open(enroll_meta_path, 'r')
    enroll_meta_map = {}
    for line in enroll_meta.readlines():
        lst = line.strip().split(' ')
        utt_id = os.path.basename(lst[0]).rstrip('.wav')
        enroll_meta_map[utt_id] = lst[1]
    enroll_meta.close()

    test_meta_path = os.path.join(metadata_dir, 'test.meta')
    test_meta = open(test_meta_path, 'r')
    test_meta_map = {}
    test_lst = []
    for line in test_meta.readlines():
        lst = line.strip().split(' ')
        utt_id = os.path.basename(lst[0]).rstrip('.wav')
        test_meta_map[utt_id] = lst[1]
        test_lst.append(utt_id)
    test_meta.close()

    return enroll_meta_map, test_meta_map, test_lst


def transfer_sr_format(input_sr_path, metadata_dir, output_sr_path):
    enroll_meta_map, test_meta_map, test_lst = load_meta_map(metadata_dir)

    f = open(input_sr_path, 'r')
    input_lines = f.readlines()
    f.close()

    f = open(output_sr_path, 'w')
    for line in input_lines:
        lst = line.strip().split(' ')
        enroll_utt_id = os.path.basename(lst[0]).rstrip('.wav')
        enroll_meta_id = enroll_meta_map[enroll_utt_id]
        f.write("{}".format(enroll_meta_id))
        for i in range(1, len(lst)):
            test_utt_id = os.path.basename(lst[i]).rstrip('.wav')
            if test_utt_id in test_lst:
                test_meta_id = test_meta_map[test_utt_id]
            else:
                test_meta_id = test_utt_id
            f.write(" {}".format(test_meta_id))
        f.write('\n')
    f.close()


# Compute mAP
def cal_mAP(input_sr_path):
    f = open(input_sr_path, 'r')
    lines = f.readlines()
    f.close()

    avg_mAP = 0.0
    num_spk = 0
    for line in lines:
        num_spk += 1
        lst = line.strip().split(' ')
        spk_id = lst[0].split('-')[0]
        spk_mAP = 0.0
        target_num = 0
        for i in range(1, len(lst)):
            utt_id = lst[i].split('-')[0]
            if utt_id == spk_id:
                target_num += 1
            spk_mAP += target_num/i
        spk_mAP /= len(lst)-1
        # print("{} {}".format(spk_id, str(spk_mAP)))
        avg_mAP += spk_mAP

    avg_mAP /= num_spk
    print("mAP = %.3f"%avg_mAP)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_sr_path', help='path of raw sr result', type=str, default="sr.topN")
    parser.add_argument('--metadata_dir', help='metadata dir', type=str, default="Task2_dev/metadata")
    parser.add_argument('--output_sr_path', help='path of meta sr result', type=str, default="sr.topN.meta")
    args = parser.parse_args()

    # generate final sr result
    transfer_sr_format(args.input_sr_path, args.metadata_dir, args.output_sr_path)
    # compute mAP
    cal_mAP(args.output_sr_path)

