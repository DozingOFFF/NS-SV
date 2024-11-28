import os
import yaml
import fire
import inspect
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from trainer.trainModel import trainModel
import torch
from argparse import ArgumentParser
from trainer.utils import set_seed

# os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5"

def cli_main(config='conf/ns_clean.yaml', **kwargs):
    # configs
    with open(config) as conf:
        yaml_config = yaml.load(conf, Loader=yaml.FullLoader)
    configs = dict(yaml_config, **kwargs)

    set_seed(configs['seed'])

    model = trainModel(**configs)

    if configs.get('evaluate', False) is not True:
        configs['default_root_dir'] = os.path.join(configs['exp_dir'],configs['save_path'])
        checkpoint_callback = ModelCheckpoint(save_top_k=configs['save_top_k'],filename='model-{epoch:02d}')
        configs['callbacks'] = [checkpoint_callback]
        if configs['auto_lr'] is True:
            configs['auto_lr_find'] = True
        valid_kwargs = inspect.signature(Trainer.__init__).parameters
        args = dict((arg, configs[arg]) for arg in valid_kwargs if arg in configs)
        if configs.get('checkpoint_path', None) is not None:
            print('trainer load: ', configs['checkpoint_path'])

            trainer = Trainer(**args, resume_from_checkpoint=configs['checkpoint_path'], accelerator='gpu', replace_sampler_ddp=False)
            trainer.fit(model)
        else:
            print('trainer from scratch')
            trainer = Trainer(**args, accelerator='gpu', replace_sampler_ddp=False)
            trainer.fit(model)
    else:
        state_dict = torch.load(configs['checkpoint_path'], map_location="cpu")["state_dict"]
        model.load_state_dict (state_dict)
        print ("initial parameter from pretrain model {}".format (configs['checkpoint_path']))
        model.cuda()
        model.eval()
        with torch.no_grad():
            model.evaluate()


if __name__ == '__main__':  # pragma: no cover
    fire.Fire(cli_main)

