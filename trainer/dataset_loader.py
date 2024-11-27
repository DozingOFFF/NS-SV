#! /usr/bin/python
# -*- encoding: utf-8 -*-
import json

import torch
import numpy as np
import random
import os
import glob
import pickle

from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset
from torch.utils.data import Sampler, DistributedSampler
from collections import Counter, OrderedDict
import torch.distributed as dist
from typing import TypeVar, Optional, Iterator
import math


def loadWAV(filename, max_frames, evalmode=False, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    sample_rate, audio  = wavfile.read(filename)

    audiosize = audio.shape[0]

    # padding
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize-max_audio, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])

    feats = []
    if evalmode and num_eval == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
    feat = np.stack(feats, axis=0).astype(float)
    return feat


def norm_wav(wav):
    #  norm wav value to [-1.0, 1.0]
    norm = max(np.absolute(wav))
    if norm > 1e-5:
        wav = wav / norm
    return wav, norm


class AugmentWAV(object):
    def __init__(self, musan_path, rir_path, max_frames):
        self.max_frames = max_frames
        self.max_audio = max_frames * 160 + 240
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise':[0,15], 'speech':[13,20], 'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,7], 'music':[1,1]}
        self.noiselist = {}
        augment_files = glob.glob (os.path.join (musan_path, '*/*/*.wav'))
        for file in augment_files:
            if file.split ('/')[-3] not in self.noiselist:
                self.noiselist[file.split ('/')[-3]] = []
            self.noiselist[file.split ('/')[-3]].append (file)
        self.rir_files = glob.glob (os.path.join (rir_path, '*/*/*.wav'))

    def additive_noise(self, noisecat, audio):
        clean_db = 10 * np.log10(np.mean(audio ** 2)+1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2)+1e-4)
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
        audio = np.sum(np.concatenate(noises,axis=0), axis=0, keepdims=True) + audio
        audio, _ = norm_wav(audio)
        return audio

    def reverberate(self, audio):
        rir_file = random.choice(self.rir_files)
        fs, rir = wavfile.read(rir_file)
        rir = np.expand_dims(rir.astype(float), 0)
        rir = rir / np.sqrt(np.sum(rir**2))
        if rir.ndim == audio.ndim:
            audio = signal.convolve(audio, rir, mode='full')[:,:self.max_audio]
        audio, _ = norm_wav(audio)
        return audio


class Train_Dataset(Dataset):
    def __init__(self, train_list, train_path, musan_path, rir_path, max_frames):
        self.train_path = train_path
        self.max_frames = max_frames
        # Load data & labels
        self.data_list = []
        self.data_label = []
        lines = open(train_list).read().splitlines()
        dictkeys = list (set ([x.split ()[0] for x in lines]))
        dictkeys.sort ()
        dictkeys = {key: ii for ii, key in enumerate (dictkeys)}
        self.label_name = {}
        for index, line in enumerate(lines):
            # speaker_label = int(line.split()[2])
            speaker_label = int(dictkeys[line.split()[0]])
            file_name = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)
            speaker = line.split()[0]
            if speaker not in self.label_name:
                self.label_name[speaker] = speaker_label
                self.label_name[speaker_label] = speaker
        print("Number of Training data is: {}".format(self.__len__()))

        self.augment_wav = AugmentWAV(musan_path, rir_path, max_frames=max_frames)


    def audio_aug(self, test_audio):
        # 0.8的几率data aug
        augtype = random.randint(1, 5)
        if augtype == 1:  # Reverberation
            test_audio = self.augment_wav.reverberate(test_audio)
        elif augtype == 2:  # Babble
            test_audio = self.augment_wav.additive_noise('speech', test_audio)
        elif augtype == 3:  # Music
            test_audio = self.augment_wav.additive_noise('music', test_audio)
        elif augtype == 4:  # Noise
            test_audio = self.augment_wav.additive_noise('noise', test_audio)
        elif augtype == 5:  # Television noise
            test_audio = self.augment_wav.additive_noise('speech', test_audio)
            test_audio = self.augment_wav.additive_noise('music', test_audio)
        return test_audio


    def __getitem__(self, index):

        enroll_index, test_index, is_match = index
        enroll_audio = loadWAV(os.path.join(self.train_path, self.data_list[enroll_index]), self.max_frames)

        # test
        test_audio = loadWAV(os.path.join(self.train_path, self.data_list[test_index]), self.max_frames)

        # data aug
        if random.uniform(0, 1) < self.aug_prob:
            test_audio = self.audio_aug(test_audio)

        return torch.FloatTensor(enroll_audio), torch.FloatTensor(test_audio), is_match

    def __len__(self):
        return len(self.data_list)




class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, hardspk_path, drop_last = True, pos_prob=0.5, start_epoch=0, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_prob = pos_prob
        self.drop_last = drop_last
        self.samespk_dict = self._samespk_indices(self.dataset.data_label)
        self.epoch = start_epoch
        self.it = start_epoch * self.__len__()
        with open(hardspk_path, 'r') as file:
            self.hardspk_dict = json.load(file)

    def __iter__(self):
        enroll_indices = np.arange(0, len(self.dataset))
        self.enroll_indices = self._shuffle_indices(enroll_indices)
        batch = []
        for i in range(len(enroll_indices)):
            batch.append (self.return_indices(i))
            if len (batch) == self.batch_size:
                yield batch
                self.it += 1
                batch = []
        if len (batch) > 0 and not self.drop_last:
            yield batch
            self.it += 1
        self.epoch += 1

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


    def return_indices(self, i):
        return [self.enroll_indices[i], self._pos_random_sample(self.enroll_indices[i]) if i%self.batch_size < self.batch_size*self.pos_prob
                    else self._neg_random_sample(self.enroll_indices[i]), 1 if i%self.batch_size < self.batch_size*self.pos_prob else 0]

    def _shuffle_indices(self, indices):
        np.random.shuffle(indices)
        return indices

    def _samespk_indices(self, data_label):
        samespk_dict = {}
        numspk_dict = Counter(data_label)
        sorted_numspk_keys = OrderedDict(sorted(numspk_dict.items(), key=lambda x: data_label.index(x[0])))
        sum = 0
        for k in sorted_numspk_keys:
            samespk_dict[k] = [sum, sum + numspk_dict[k] - 1]
            sum += numspk_dict[k]
        return samespk_dict

    def _pos_random_sample(self, index):
        # 随机获取与之来源于同一个说话人的语音
        start, end = self.samespk_dict[self.dataset.data_label[index]]
        return np.random.randint(start, end+1)

    # def _neg_random_sample(self, index):
    #     # 随机获取与之来源于不同说话人的语音
    #     start, end = self.samespk_dict[self.dataset.data_label[index]]
    #     # neg_arr = np.setdiff1d(np.arange(0, len(self.dataset)), np.arange(start, end+1))
    #     random_negs = np.random.randint(0, len(self.dataset), size=5)
    #     filtered_negs = random_negs[(random_negs < start) | (random_negs > end)]
    #     return np.random.choice(filtered_negs)

    def _neg_random_sample(self, index):
        # 随机获取与之来源于不同说话人的语音
        # 分段，前十个epoch0.5-0.1递减，后十个epoch0.2-0.01递减
        spk_list = self.hardspk_dict[self.dataset.label_name[self.dataset.data_label[index]]]
        if self.epoch < 10:
            p = 0.5 - self.epoch * 0.03
            if random.random() >= p:
                spk_list = spk_list[:int(p*len(spk_list))]
            else:
                spk_list = spk_list[int(p*len(spk_list)):]
        else:
            p = 0.39 - min(self.epoch, 20) * 0.019
            if random.random() >= 0.1:
                spk_list = spk_list[:int(p*len(spk_list))]
            else:
                spk_list = spk_list[int(p*len(spk_list)):]

        spk_index = self.dataset.label_name[random.choice(spk_list)]
        start, end = self.samespk_dict[spk_index]
        return np.random.randint(start, end+1)




class BalancedDistributedSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, batch_size, hardspk_path, num_replicas: Optional[int] = None,
                 pos_prob = 0.5, rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True, start_epoch=0, **kwargs) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = start_epoch
        self.drop_last = drop_last
        self.pos_prob = pos_prob

        with open(hardspk_path, 'r') as file:
            self.hardspk_dict = json.load(file)

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.samespk_dict = self._samespk_indices(self.dataset.data_label)

        print('rank:', self.rank)

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # 确保每个epoch在多张卡上得到的indices一样
            print('dataloader seed:', self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        # 获取单卡上的数据索引
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples


        self.enroll_indices = indices
        batch = []
        for i in range(len(self.enroll_indices)):
            batch.append(self.return_indices(i))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def return_indices(self, i):
        return [self.enroll_indices[i],
                self._pos_random_sample(self.enroll_indices[i]) if i % self.batch_size < self.batch_size * self.pos_prob
                else self._neg_random_sample(self.enroll_indices[i]),
                1 if i % self.batch_size < self.batch_size * self.pos_prob else 0]


    def _shuffle_indices(self, indices):
        np.random.shuffle(indices)
        return indices

    def _samespk_indices(self, data_label):
        samespk_dict = {}
        numspk_dict = Counter(data_label)
        sorted_numspk_keys = OrderedDict(sorted(numspk_dict.items(), key=lambda x: data_label.index(x[0])))
        sum = 0
        for k in sorted_numspk_keys:
            samespk_dict[k] = [sum, sum + numspk_dict[k]-1]
            sum += numspk_dict[k]
        return samespk_dict

    def _pos_random_sample(self, index):
        # 随机获取与之来源于同一个说话人的语音
        start, end = self.samespk_dict[self.dataset.data_label[index]]
        return np.random.randint(start, end + 1)

    def _neg_random_sample(self, index):
        # 随机获取与之来源于不同说话人的语音
        # 分段，前十个epoch0.5-0.1递减，后十个epoch0.2-0.01递减
        spk_list = self.hardspk_dict[self.dataset.label_name[self.dataset.data_label[index]]]

        if self.epoch < 10:
            p = 0.5 - self.epoch * 0.04
            if random.random() >= p:
                spk_list = spk_list[:int(p * len(spk_list))]
            else:
                spk_list = spk_list[int(p * len(spk_list)):]
        else:
            p = 0.39 - min(self.epoch, 20) * 0.019
            if random.random() >= 0.1:
                spk_list = spk_list[:int(p * len(spk_list))]
            else:
                spk_list = spk_list[int(p * len(spk_list)):]

        spk_index = self.dataset.label_name[random.choice(spk_list)]
        start, end = self.samespk_dict[spk_index]
        return np.random.randint(start, end + 1)



class Test_Dataset(Dataset):
    def __init__(self, test_path, label, enroll_list, test_list, eval_frames, num_eval=0, **kwargs):
        # load data list
        self.test_path = test_path
        self.label = label
        self.enroll_list = enroll_list
        self.test_list = test_list
        self.max_frames = eval_frames
        self.num_eval   = num_eval
        self.label = label

    def __getitem__(self, index):
        try:
            enroll_audio = loadWAV(self.enroll_list[index], self.max_frames, evalmode=True, num_eval=self.num_eval)
            test_audio = loadWAV(self.test_list[index], self.max_frames, evalmode=True, num_eval=self.num_eval)
        except:
            enroll_audio = loadWAV(os.path.join(self.test_path, self.enroll_list[index]), self.max_frames, evalmode=True, num_eval=self.num_eval)
            test_audio = loadWAV(os.path.join(self.test_path, self.test_list[index]), self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(enroll_audio), torch.FloatTensor(test_audio), self.label[index]

    def __len__(self):
        return len(self.enroll_list)



#########################################################################################################


class MixBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, hardspk_path, drop_last = True, max_pos_prob=0.5, min_pos_prob=0.2, start_epoch=0, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_prob = max_pos_prob
        self.max_pos_prob = max_pos_prob
        self.min_pos_prob = min_pos_prob
        self.drop_last = drop_last
        self.samespk_dict = self._samespk_indices(self.dataset.data_label)
        self.epoch = start_epoch
        self.it = start_epoch * self.__len__()
        with open(hardspk_path, 'r') as file:
            self.hardspk_dict = json.load(file)

    def __iter__(self):
        enroll_indices = np.arange(0, len(self.dataset))
        self.enroll_indices = self._shuffle_indices(enroll_indices)
        batch = []
        self.pos_prob_decrease()
        for i in range(len(enroll_indices)):
            batch.append (self.return_indices(i))
            if len (batch) == self.batch_size:
                yield batch
                self.it += 1
                self.pos_prob_decrease()
                batch = []
        if len (batch) > 0 and not self.drop_last:
            yield batch
            self.it += 1
        self.epoch += 1

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


    def pos_prob_decrease(self):
        if self.epoch >= 5 and self.epoch <10:
            self.pos_prob = self.min_pos_prob + (self.max_pos_prob - self.min_pos_prob) * (1 - (self.it-5*self.__len__()) / (5*self.__len__()))
        elif self.epoch >= 10:
            self.pos_prob = self.min_pos_prob

    def return_indices(self, i):
        return [self.enroll_indices[i],
                self._pos_random_sample(self.enroll_indices[i]) if i%self.batch_size < self.batch_size*self.pos_prob
                else self._neg_random_sample(self.enroll_indices[i]),
                self._neg_random_sample(self.enroll_indices[i]),
                1 if i%self.batch_size < self.batch_size*self.pos_prob else 0]

    def _shuffle_indices(self, indices):
        np.random.shuffle(indices)
        return indices

    def _samespk_indices(self, data_label):
        samespk_dict = {}
        numspk_dict = Counter(data_label)
        sorted_numspk_keys = OrderedDict(sorted(numspk_dict.items(), key=lambda x: data_label.index(x[0])))
        sum = 0
        for k in sorted_numspk_keys:
            samespk_dict[k] = [sum, sum + numspk_dict[k] - 1]
            sum += numspk_dict[k]
        return samespk_dict

    def _pos_random_sample(self, index):
        # 随机获取与之来源于同一个说话人的语音
        start, end = self.samespk_dict[self.dataset.data_label[index]]
        return np.random.randint(start, end+1)

    def _neg_random_sample(self, index):
        # 随机获取与之来源于不同说话人的语音
        hardspk_list = self.hardspk_dict[self.dataset.label_name[self.dataset.data_label[index]]]
        hardspk_index = self.dataset.label_name[random.choice(hardspk_list)]
        start, end = self.samespk_dict[hardspk_index]
        return np.random.randint(start, end+1)





class Mix_Train_Dataset(Dataset):
    def __init__(self, train_list, train_path, musan_path, rir_path, max_frames, aug_prob):
        self.train_path = train_path
        self.max_frames = max_frames
        self.max_audio = max_frames * 160 + 240
        self.aug_prob = aug_prob
        # Load data & labels
        self.data_list = []
        self.data_label = []
        lines = open(train_list).read().splitlines()
        # dictkeys = list (set ([x.split ()[0] for x in lines]))
        # dictkeys.sort ()
        # dictkeys = {key: ii for ii, key in enumerate (dictkeys)}
        self.label_name = {}
        for index, line in enumerate(lines):
            speaker_label = int(line.split()[2])
            file_name = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)
            speaker = line.split()[0]
            if speaker not in self.label_name:
                self.label_name[speaker] = speaker_label
                self.label_name[speaker_label] = speaker
        print("Number of Training data is: {}".format(self.__len__()))

        self.augment_wav = AugmentWAV(musan_path, rir_path, max_frames=max_frames)

    def partly_mix_audio(self, mix_audio_1, mix_audio_2):
        # mix_audio_1和mix_audio_2都为2s
        # 重合比例 [0, 0.5]
        overlap_ratio = random.uniform(0.1, 0.9)  # 总长度overlap的比例
        overlap_length = int(self.max_audio * overlap_ratio)
        # 开始进行overlap的位置
        start_frame = random.randint(0, self.max_audio - overlap_length)

        # 能量比例
        mix_audio_db_1 = 10 * np.log10(np.mean(mix_audio_1 ** 2) + 1e-4)
        mix_audio_db_2 = 10 * np.log10(np.mean(mix_audio_2 ** 2) + 1e-4)
        snr = random.uniform(-3, 3)
        mix_audio_2 *= np.sqrt(10 ** ((mix_audio_db_1 - mix_audio_db_2 - snr) / 10))

        mixed_audio = np.zeros((1, self.max_audio))
        mixed_audio[:, 0:start_frame + overlap_length] = mix_audio_1[:, 0:start_frame + overlap_length]
        mixed_audio[:, start_frame:] += mix_audio_2[:, 0:self.max_audio - start_frame]
        mixed_audio, _ = norm_wav(mixed_audio)
        return mixed_audio

    def completely_mix_audio(self, mix_audio_1, mix_audio_2):
        # 能量比例
        mix_audio_db_1 = 10 * np.log10(np.mean(mix_audio_1 ** 2) + 1e-4)
        mix_audio_db_2 = 10 * np.log10(np.mean(mix_audio_2 ** 2) + 1e-4)
        snr = random.uniform(-3, 3)
        # mix_audio_2 *= np.sqrt((mix_audio_db_1 / (mix_audio_db_2 * (10 ** (snr / 10)) + 1e-4)))
        mix_audio_2 *= np.sqrt(10 ** ((mix_audio_db_1 - mix_audio_db_2 - snr) / 10))

        mixed_audio = mix_audio_1 + mix_audio_2
        mixed_audio, _ = norm_wav(mixed_audio)
        return mixed_audio

    def concat_audio(self, mix_audio_1, mix_audio_2):
        # 计算两个语音信号的能量
        energy_audio_1 = np.sum(mix_audio_1 ** 2)
        energy_audio_2 = np.sum(mix_audio_2 ** 2)
        # 根据能量调整第二个语音的幅度
        mix_audio_2 = np.sqrt(energy_audio_1 / energy_audio_2) * mix_audio_2
        mixed_audio = np.zeros((1, self.max_audio * 2))
        mixed_audio[:, 0:self.max_audio] = mix_audio_1
        mixed_audio[:, self.max_audio:] = mix_audio_2
        start_frame = random.randint(int(self.max_audio * 0.2), int(self.max_audio * 0.8))
        mixed_audio = mixed_audio[:, start_frame:start_frame + self.max_audio]
        mixed_audio, _ = norm_wav(mixed_audio)
        return mixed_audio

    def front_prob_mix(self, mix_audio_1, mix_audio_2):
        front_prob = random.uniform(0, 1)
        if front_prob < 0.5:
            tmp = mix_audio_1
            mix_audio_1 = mix_audio_2
            mix_audio_2 = tmp

        mix_way_prob = random.randint(0, 2)
        if mix_way_prob == 0:
            mixed_audio = self.partly_mix_audio(mix_audio_1, mix_audio_2)
        elif mix_way_prob == 1:
            mixed_audio = self.completely_mix_audio(mix_audio_1, mix_audio_2)
        elif mix_way_prob == 2:
            mixed_audio = self.concat_audio(mix_audio_1, mix_audio_2)

        return mixed_audio

    def audio_aug(self, test_audio):
        # 0.8的几率data aug
        augtype = random.randint(1, 5)
        if augtype == 1:  # Reverberation
            test_audio = self.augment_wav.reverberate(test_audio)
        elif augtype == 2:  # Babble
            test_audio = self.augment_wav.additive_noise('speech', test_audio)
        elif augtype == 3:  # Music
            test_audio = self.augment_wav.additive_noise('music', test_audio)
        elif augtype == 4:  # Noise
            test_audio = self.augment_wav.additive_noise('noise', test_audio)
        elif augtype == 5:  # Television noise
            test_audio = self.augment_wav.additive_noise('speech', test_audio)
            test_audio = self.augment_wav.additive_noise('music', test_audio)
        return test_audio



    def __getitem__(self, index):

        enroll_index, test_index, mix_index, is_match = index
        enroll_audio = loadWAV(os.path.join(self.train_path, self.data_list[enroll_index]), self.max_frames)

        # test
        test_audio = loadWAV(os.path.join(self.train_path, self.data_list[test_index]), self.max_frames)
        mix_audio = loadWAV(os.path.join(self.train_path, self.data_list[mix_index]), self.max_frames)

        # test mix
        mix_prob = random.uniform(0, 1)
        if mix_prob < 0.75:
            test_audio = self.front_prob_mix(test_audio, mix_audio)

        if random.uniform(0, 1) < self.aug_prob:
            test_audio = self.audio_aug(test_audio)

        return torch.FloatTensor(enroll_audio), torch.FloatTensor(test_audio), is_match

    def __len__(self):
        return len(self.data_list)

