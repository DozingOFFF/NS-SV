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

        WeightBCEloss = importlib.import_module('loss.' + self.hparams.loss_function).__getattribute__('LossFunction')
        self.BCEloss = WeightBCEloss(pos_weight=self.hparams.pos_weight)
        self.e_num = self.hparams.enroll_num
        self.expand_index = self.get_index(self.e_num)
    
    
    def get_index(self, step):
        index = torch.empty((0,))
        for i in range(self.hparams.batch_size):
            index = torch.cat((index, torch.arange(i, i + step) % self.hparams.batch_size))
        return index.int()


    def training_epoch_end(self, outputs):
        if (self.current_epoch + 1) % self.hparams.eval_interval == 0:
            self.eval()
            self.evaluate_fast()
        print('\nlr:', self.optimizers().param_groups[0]['lr'])


    def forward(self, enroll_utt, test_utt, extra_utt=None):
        # enroll model
        if self.hparams.mode == 'clean':
            with torch.no_grad():
                enroll_emb = self.enrollmodel(enroll_utt)
                enroll_emb = enroll_emb.unsqueeze(1)
            enroll_emb = self.scoring_linear2(enroll_emb)
            enroll_emb = enroll_emb[self.expand_index, :, :].view(self.hparams.batch_size, self.e_num, -1)

            test_feature = self.testencoder(test_utt)
            test_feature = test_feature.transpose(1, 2)
            test_feature = self.scoring_linear1(test_feature)

            output = self.transformer_model(enroll_emb, test_feature)[:, :self.e_num, :]

        elif self.hparams.mode == 'multi':
            with torch.no_grad():
                enroll_emb = self.enrollmodel(enroll_utt)
                enroll_emb = enroll_emb.unsqueeze(1)
                extra_emb = self.enrollmodel(extra_utt)
                extra_emb = extra_emb.unsqueeze(1)
            enroll_emb = self.scoring_linear2(enroll_emb)
            extra_emb = self.scoring_linear2(extra_emb)
            enroll_emb = enroll_emb[self.expand_index, :, :].view(self.hparams.batch_size, self.e_num, -1)
            enroll_emb = torch.cat((extra_emb, enroll_emb), dim=1)

            test_feature = self.testencoder(test_utt)
            test_feature = test_feature.transpose(1, 2)
            test_feature = self.scoring_linear1(test_feature)

            output = self.transformer_model(enroll_emb, test_feature)[:, :self.e_num+1, :]


        score = self.mlp(output)
        score = score.reshape(-1, 1).squeeze(1)
        score = torch.sigmoid(score)

        return score


    def evaluate_forward(self, enroll_embs, test_utt):
        enroll_embs = self.scoring_linear2(enroll_embs)  # [N, 1, 256]

        # scoring model
        test_feature, test_emb = self.testencoder(test_utt)
        test_feature = test_feature.transpose(1, 2)
        test_feature = self.scoring_linear1(test_feature)

        outputs = self.transformer_model(enroll_embs, test_feature)[:, :enroll_embs.size(0), :]
        score = self.mlp(outputs).squeeze(1)  # [N]
        score = torch.sigmoid(score)

        return score



    def training_step(self, batch, batch_idx):
        if self.hparams.mode == 'clean':
            enroll_utt, test_utt, enroll_label, test_label = batch
            pre_score = self.forward(enroll_utt, test_utt)
            enroll_label = enroll_label[self.expand_index]
            test_label = test_label.repeat_interleave(len(self.expand_index) // self.hparams.batch_size)
            label_is_match = (enroll_label == test_label).int()
        elif self.hparams.mode == 'multi':
            enroll_utt, extra_utt, test_utt, enroll_label, extra_label, test_label, mix_label = batch
            pre_score = self.forward(enroll_utt, test_utt, extra_utt)
            enroll_label = enroll_label[self.expand_index].view(-1, self.e_num)
            enroll_label = torch.cat((extra_label.unsqueeze(1), enroll_label), dim=1).reshape(-1, 1).squeeze(1)
            test_label_list = torch.cat((test_label.unsqueeze(1), mix_label.unsqueeze(1)), dim=1)
            test_label_list = test_label_list.repeat_interleave(self.e_num + 1, dim=0)
            label_is_match = torch.any(enroll_label.unsqueeze(1) == test_label_list, dim=1).int()

        BCE_loss = self.BCEloss(pre_score, label_is_match.float())
        predictions = (pre_score >= 0.5).int()
        accuracy = (predictions == label_is_match).float().mean()
        loss = BCE_loss
        self.log('train_loss', loss, prog_bar=True)
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

    def extract_enroll_embeddings(self, unique_enrolls):
        enroll_embs_dict = {}

        print("Extract enroll embeddings...")
        with torch.no_grad():
            for enroll_id in tqdm(unique_enrolls):
                enroll_dataset = Test_Dataset(
                    test_path=self.hparams.test_path,
                    label=['0'],
                    enroll_list=[enroll_id],
                    test_list=[enroll_id],
                    eval_frames=self.hparams.eval_frames,
                    num_eval=0
                )
                enroll_loader = DataLoader(enroll_dataset, batch_size=1, shuffle=False)
                enroll_utt, _, _ = next(iter(enroll_loader))
                enroll_utt = enroll_utt.cuda()
                enroll_emb = self.enrollmodel(enroll_utt)
                enroll_embs_dict[enroll_id] = enroll_emb.unsqueeze(1)

        return enroll_embs_dict

    def get_test_utterance(self, test_file):
        test_dataset = Test_Dataset(
            test_path=self.hparams.test_path,
            label=['0'],
            enroll_list=[test_file],
            test_list=[test_file],
            eval_frames=self.hparams.eval_frames,
            num_eval=0
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test_utt, _, _ = next(iter(test_loader))
        test_utt = test_utt[:, :, :(self.hparams.max_len - 1) * 8 * 160 + 240].cuda()
        return test_utt

    def process_trials(self, eval_trials):
        df = pd.DataFrame(eval_trials, columns=['labels', 'enrolls', 'tests'])

        # 分组并构建字典
        trials_by_test = {
            test_file: {
                'enrolls': group['enrolls'].values,
                'labels': group['labels'].astype(int).values
            }
            for test_file, group in df.groupby('tests')
        }

        return trials_by_test

    def evaluate_fast(self):
        eval_trials = self.trials

        print("Start evaluating...")

        # 1. 提取所有注册说话人嵌入向量
        unique_enrolls = np.unique(eval_trials.T[1])
        enroll_embs_dict = self.extract_enroll_embeddings(unique_enrolls)

        # 2. 处理trials，按测试语音分组
        trials_by_test = self.process_trials(eval_trials)

        # 3. 计算得分
        print("Compute scores...")
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for test_file in tqdm(trials_by_test.keys()):
                # 获取当前测试文件的trials信息
                current_trials = trials_by_test[test_file]
                current_enrolls = current_trials['enrolls']
                current_labels = current_trials['labels']

                # 获取测试语音
                test_utt = self.get_test_utterance(test_file)

                # 收集当前测试文件对应的所有注册说话人嵌入向量
                current_enroll_embs = torch.cat([enroll_embs_dict[enroll_id] for enroll_id in current_enrolls], dim=0)

                # 计算分数
                scores = self.evaluate_forward(current_enroll_embs, test_utt)

                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(current_labels)


        # 4. 计算性能指标
        eer, th = compute_eer(all_scores, all_labels)

        c_miss = 1
        c_fa = 1
        fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
        mindcf_easy, _, mindcf_easy_fnr, mindcf_easy_fpr = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, c_miss, c_fa)
        mindcf_hard, _, mindcf_hard_fnr, mindcf_hard_fpr = ComputeMinDcf(fnrs, fprs, thresholds, 0.001, c_miss, c_fa)

        print("Cosine EER: {:.3f}%  minDCF(0.01): {:.5f}  minDCF(0.001): {:.5f}".format(
            eer * 100, mindcf_easy, mindcf_hard))

        self.log('cosine_eer', eer * 100, sync_dist=True)
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


