#!/usr/bin/env python
# encoding: utf-8

import os
import fire
import importlib
import torch
import torchaudio
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from scipy.io import wavfile
import numpy as np
import random
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


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



class Enroll_Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Network information Report
        print("---------Enroll_Model---------")
        print("Network Type: ", self.hparams.nnet_type)
        print("Pooling Type: ", self.hparams.pooling_type)
        print("Embedding Dim: ", self.hparams.embedding_dim)

        #########################
        ### Network Structure ###
        #########################

        # 1. Acoustic Feature
        sr = self.hparams.sample_rate
        print('sample rate: ', sr)
        self.mel_trans = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=512, 
                                                     win_length=sr * 25 // 1000, hop_length=sr * 10 // 1000, 
                                                     window_fn=torch.hamming_window, n_mels=self.hparams.n_mels)
                )
        self.instancenorm = nn.InstanceNorm1d(self.hparams.n_mels)

        # 2. Speaker_Encoder
        Speaker_Encoder = importlib.import_module('nnet.' + self.hparams.nnet_type + '_pooling').__getattribute__('Speaker_Encoder')
        self.speaker_encoder = Speaker_Encoder(**dict(self.hparams))


    def forward(self, x):
        x = self.extract_speaker_embedding(x)
        return x

    def extract_speaker_embedding(self, data):
        x = data.reshape(-1, data.size()[-1])
        x = self.mel_trans(x) + 1e-6
        x = x.log()
        x = self.instancenorm(x)
        x = self.speaker_encoder(x)
        return x

    def load_state_dict(self, state_dict):
        self_state = self.state_dict ()
        for name, param in state_dict.items ():
            origname = name
            if name not in self_state:
                print ("%s is not in the model." % origname)
                continue
            if self_state[name].size () != state_dict[origname].size ():
                print ("Wrong parameter length: %s, model: %s, loaded: %s" % (
                origname, self_state[name].size (), state_dict[origname].size ()))
                continue
            self_state[name].copy_ (param)


class Test_Encoder(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Network information Report
        print("---------Test_Encoder---------")
        print("Network Type: ", self.hparams.nnet_type)

        #########################
        ### Network Structure ###
        #########################

        # 1. Acoustic Feature
        sr = self.hparams.sample_rate
        print('sample rate: ', sr)
        self.mel_trans = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=512, 
                                                     win_length=sr * 25 // 1000, hop_length=sr * 10 // 1000, 
                                                     window_fn=torch.hamming_window, n_mels=self.hparams.n_mels)
                )
        self.instancenorm = nn.InstanceNorm1d(self.hparams.n_mels)

        # 2. Speaker_Encoder
        Speaker_Encoder = importlib.import_module('nnet.'+self.hparams.nnet_type).__getattribute__('Speaker_Encoder')
        self.speaker_encoder = Speaker_Encoder(**dict(self.hparams))


    def forward(self, x):
        x = self.extract_speaker_embedding(x)
        return x

    def extract_speaker_embedding(self, data):
        x = data.reshape(-1, data.size()[-1])
        x = self.mel_trans(x) + 1e-6
        x = x.log()
        x = self.instancenorm(x)
        x = self.speaker_encoder(x)
        return x

    def load_state_dict(self, state_dict):
        self_state = self.state_dict ()
        for name, param in state_dict.items ():
            origname = name
            if name not in self_state:
                print ("%s is not in the model." % origname)
                continue
            if self_state[name].size () != state_dict[origname].size ():
                print ("Wrong parameter length: %s, model: %s, loaded: %s" % (
                origname, self_state[name].size (), state_dict[origname].size ()))
                continue
            self_state[name].copy_ (param)



class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        assert len(inputs.size()) == 2, 'The number of dimensions of inputs tensor must be 2!'
        # reflect padding to match lengths of in/out
        inputs = inputs.unsqueeze(1)
        inputs = F.pad(inputs, (1, 0), 'reflect')
        inputs = F.conv1d(inputs, self.flipped_filter).squeeze(1)
        return inputs

