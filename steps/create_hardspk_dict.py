import json
import os
import sys
import random
import pickle

import torch
from torch.nn import functional as F
import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from argparse import ArgumentParser
import numpy as np
from trainer.module import Enroll_Model
from torch.utils.data import Dataset
from trainer.dataset_loader import loadWAV

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def add_model_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--nnet_type', type=str, default="ResNet34")
    parser.add_argument('--pooling_type', type=str, default="ASP")
    parser.add_argument('--pretrain', type=str, default='../source/resnet34_avg.ckpt')
    parser.add_argument('--train_list', type=str, default='')
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--embedding_dim', type=int, default=256)

    return parser


class Train_Dataset(Dataset):
    def __init__(self, train_list, train_path, max_frames):
        self.train_path = train_path
        self.max_frames = max_frames
        # Load data & labels
        self.data_list = []
        self.data_label = []
        lines = open(train_list).read().splitlines()

        for index, line in enumerate(lines):
            speaker_label = int(line.split()[2])
            file_name = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)
        print("Number of Training data is: {}".format(self.__len__()))


    def __getitem__(self, index):

        enroll_index = index
        enroll_audio = loadWAV(os.path.join(self.train_path, self.data_list[enroll_index]), self.max_frames)
        return torch.FloatTensor(enroll_audio), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


def spk_emb(model, args):
    model.cuda()
    spk_emb_dict = {}
    lines = open(args.train_list).read().splitlines()
    data_list = []
    data_label = []
    for index, line in enumerate(lines):
        speaker_label = line.split()[0]
        file_name = os.path.join(args.train_path, line.split()[1])
        data_label.append(speaker_label)
        data_list.append(file_name)
    data_label = np.array(data_label)
    data_list = np.array(data_list)
    spks = list(set(data_label))


    for spk in tqdm.tqdm(spks):
        if len(data_list[data_label==spk]) >= 10:
            sample_utts = np.random.choice(data_list[data_label==spk],10, False)
        else:
            sample_utts = data_list[data_label==spk]

        audios = []
        for utt in sample_utts:
            audio = loadWAV(os.path.join(args.train_path, utt), args.max_frames)
            audios.append(audio)
        spk_embs = model(torch.FloatTensor(np.array(audios)).cuda())
        spk_embs = F.normalize(spk_embs , p=2, dim=1)
        spk_emb_dict[spk] = torch.mean(spk_embs, dim=0).cpu().numpy()

    return spk_emb_dict


def hard_spk_dict(spk_emb_dict):
    hard_spk_dict = {}
    spks = np.array(list(spk_emb_dict.keys()))
    embs = np.array(list(spk_emb_dict.values()))
    for i in range(len(spks)):
        cos_distance = F.cosine_similarity(torch.FloatTensor(embs[i]).unsqueeze(0), torch.FloatTensor(embs)).cpu().numpy()
        cos_distance[i] = 10
        # hard = spks[cos_distance>=0.3]
        hard = spks[np.argsort(-cos_distance)][1:]
        # assert len(hard) != 0
        assert spks[i] not in hard
        hard_spk_dict[spks[i]] = hard.tolist()
    return hard_spk_dict




if __name__ == '__main__':


    parser = ArgumentParser()
    parser = add_model_specific_args(parser)
    args = parser.parse_args()


    model = Enroll_Model(**vars(args))
    state_dict = torch.load(args.pretrain, map_location='cpu')["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        spk_emb_dict = spk_emb(model, args)
        print('finish emb extracting')
        hard_spk = hard_spk_dict(spk_emb_dict)
    with open('../source/voxceleb2-hardspksort.json', 'w') as f:
        json.dump(hard_spk, f)