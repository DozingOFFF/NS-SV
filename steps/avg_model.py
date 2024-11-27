import os

import torch

from argparse import ArgumentParser

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def add_model_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--version_path', type=str, default="")
    parser.add_argument('--avg_last_k', type=int, default=10)

    return parser


parser = ArgumentParser()
parser = add_model_specific_args(parser)
args = parser.parse_args()


# 初始化平均权重
avg_weights = None
# 获取目录中所有的模型文件
checkpoint_dir = os.path.join(args.version_path, 'checkpoints')
model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
# 选取最后的k个epoch的模型文件
assert len(model_files) >= args.avg_last_k
recent_model_files = sorted(model_files)[-args.avg_last_k:]
# 遍历 checkpoint 文件夹，计算平均权重
for file_name in recent_model_files:
    checkpoint = torch.load(os.path.join(checkpoint_dir, file_name))
    model_weights = checkpoint['state_dict']
    if avg_weights is None:
        avg_weights = model_weights
    else:
        # 累加权重
        for key in avg_weights:
            avg_weights[key] += model_weights[key]

# 计算平均值
for key in avg_weights:
    avg_weights[key] = torch.true_divide(avg_weights[key], args.avg_last_k)

checkpoint['state_dict'] = avg_weights
# 保存平均权重到文件
print('save:', os.path.join(args.version_path, 'avg'+str(args.avg_last_k))+'.ckpt')
torch.save(checkpoint, os.path.join(args.version_path, 'avg'+str(args.avg_last_k))+'.ckpt')