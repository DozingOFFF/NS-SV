from argparse import ArgumentParser

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
import torch.distributed as dist

from tqdm import tqdm

import importlib

from trainer.dataset_loader import *
from trainer.metric.compute_eer import compute_eer
from trainer.metric.tuneThreshold import *

from trainer.module import Enroll_Model, Test_Encoder
from trainer.nnet.TransformerEncoder import TransformerEncoder
import soundfile


class trainModel(LightningModule):
    def __init__(self, **kwargs):
        super(trainModel,self).__init__()
        self.save_hyperparameters()

        # load trials and data list
        if os.path.exists(self.hparams.trials_path):
            self.trials = np.loadtxt(self.hparams.trials_path, dtype=str)

        ## 1. enrollmodel
        self.enrollmodel = Enroll_Model(**dict(self.hparams))
        state_dict = torch.load(self.hparams.pretrain, map_location='cpu')["state_dict"]
        self.enrollmodel.load_state_dict(state_dict)

        ## 2. scoringmodel
        self.testencoder = Test_Encoder(**dict(self.hparams)) # resnet34 (no_pooling)
        self.testencoder.load_state_dict (state_dict)

        # linear
        self.scoring_linear1 = nn.Linear(2560, self.hparams.hidden_dim)
        self.scoring_linear2 = nn.Linear(self.hparams.embedding_dim, self.hparams.hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),nn.ReLU(),
                                 nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim), nn.ReLU(),
                                  nn.Linear(self.hparams.hidden_dim, 1))

        # Transformer
        self.transformer_model = TransformerEncoder(**dict(self.hparams))


        # 3. Loss / Classifier

        self.BCEloss = nn.BCELoss()



    def training_epoch_end(self, outputs):
        if (self.current_epoch + 1) % self.hparams.eval_interval == 0:
            self.eval()
            self.evaluate()
        print('\nlr:', self.optimizers().param_groups[0]['lr'])

    

    def forward(self, enroll_utt, test_utt):
        # enroll model
        with torch.no_grad():
            enroll_emb = self.enrollmodel(enroll_utt)  # [B, 256]
            enroll_emb = enroll_emb.unsqueeze(1)
        enroll_emb = self.scoring_linear2(enroll_emb)

        # scoring model
        test_feature = self.testencoder(test_utt) # [B, 2560, 26]   [b, f, t]
        test_feature = test_feature.transpose(1, 2)
        test_feature = self.scoring_linear1(test_feature)

        outputs = self.transformer_model(enroll_emb, test_feature)[:, 0, :]

        score = self.mlp(outputs).squeeze(1)
        score = torch.sigmoid(score)

        return score



    def evaluate_forward(self, enroll_utt, test_utt):
        # enroll model
        with torch.no_grad():
            enroll_emb = self.enrollmodel(enroll_utt)  # [B, 256]
            enroll_emb = enroll_emb.unsqueeze(1)
        enroll_emb = self.scoring_linear2(enroll_emb)

        # scoring model
        test_feature = self.testencoder(test_utt)  # [B, 2560, 26]   [b, f, t]
        test_feature = test_feature.transpose(1, 2)
        test_feature = self.scoring_linear1(test_feature)

        outputs = self.transformer_model(enroll_emb, test_feature)[:, 0, :]

        score = self.mlp(outputs).squeeze(1)
        score = torch.sigmoid(score)

        return score




    def training_step(self, batch, batch_idx):
        enroll_utt, test_utt, label_is_match = batch
        pre_score = self.forward(enroll_utt, test_utt)
        BCE_loss = self.BCEloss (pre_score, label_is_match.float ())
        predictions = (pre_score >= 0.5).int ()  # 暂时假设阈值为0.5
        accuracy = (predictions == label_is_match).float ().mean ()
        loss = BCE_loss
        self.log('train_loss', loss)
        self.log('acc', accuracy, prog_bar=True)
        output = OrderedDict({
            'loss': loss,
            })
        return output




    def train_dataloader(self):
        frames_len = np.random.randint(self.hparams.min_frames, self.hparams.max_frames)
        print("\nChunk size is: ", frames_len)

        if self.hparams.mode == 'clean':
            train_dataset = Train_Dataset(**dict(self.hparams))
            if self.hparams.devices == 1:
                batch_sampler = BalancedBatchSampler(dataset=train_dataset, start_epoch=self.current_epoch, **dict(self.hparams))
            else:
                batch_sampler = BalancedDistributedSampler(dataset=train_dataset, start_epoch=self.current_epoch, **dict(self.hparams))
        elif self.hparams.mode == 'multi':
            train_dataset = Mix_Train_Dataset(**dict(self.hparams))
            if self.hparams.devices == 1:
                batch_sampler = MixBatchSampler(dataset=train_dataset, **dict(self.hparams))
            else:
                batch_sampler = MixDistributedSampler(dataset=train_dataset, **dict(self.hparams))

        self.batch_sampler = batch_sampler

        loader = torch.utils.data.DataLoader(
                train_dataset,
                num_workers=self.hparams.num_workers,
                batch_sampler=self.batch_sampler,
                pin_memory=True
                )
        return loader


    def test_dataloader(self, trials):
        label = trials.T[0]
        enroll_list = trials.T[1]
        test_list = trials.T[2]

        print("\nnumber of trials: ", len(enroll_list))

        test_dataset = Test_Dataset(test_path=self.hparams.test_path, label=label, enroll_list=enroll_list,
                                    test_list=test_list, eval_frames=self.hparams.eval_frames, num_eval=0)
        loader = DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False)
        return loader


    def evaluate(self):
        if self.hparams.evaluate is False:
            eval_loader = self.test_dataloader(self.trials[np.random.choice(self.trials.shape[0], size=self.hparams.eval_size, replace=False)])
        else:
            eval_loader = self.test_dataloader(self.trials)
        print("extract eval speaker embedding...")

        all_scores = []
        all_labels = []
        with torch.no_grad():
            for idx, (enroll_utt, test_utt, label) in enumerate(tqdm(eval_loader)):
                enroll_utt = enroll_utt.cuda()
                test_utt = test_utt[:,:,:(self.hparams.max_len-1) * 8 * 160 + 240].cuda()

                score = self.evaluate_forward(enroll_utt, test_utt)

                all_scores.append(score[0].item())
                all_labels.append(int(label[0]))

        eer, th = compute_eer(all_scores, all_labels)

        c_miss = 1
        c_fa = 1
        fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
        mindcf_easy, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, c_miss, c_fa)
        mindcf_hard, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.001, c_miss, c_fa)
        print("Cosine EER: {:.3f}%  minDCF(0.01): {:.5f}  minDCF(0.001): {:.5f}".format(eer*100, mindcf_easy, mindcf_hard))
        self.log('cosine_eer', eer*100, sync_dist=True)
        self.log('minDCF(0.01)', mindcf_easy, sync_dist=True)
        self.log('minDCF(0.001)', mindcf_hard, sync_dist=True)

        return eer, th, mindcf_easy, mindcf_hard



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.hparams.lr_gamma, patience=1),
            "monitor": 'train_loss',
            'interval': 'epoch',
        }
        print("init {} optimizer with learning rate {}".format("ReduceLROnPlateau", optimizer.param_groups[0]['lr']))
        return { "optimizer": optimizer, "lr_scheduler": scheduler}


