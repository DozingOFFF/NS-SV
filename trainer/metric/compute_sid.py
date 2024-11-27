import os
import fire
from tqdm import tqdm


def main(scores_file, topk):
    topk = list(topk)
    assert all(isinstance(x, int) and x > 0 for x in topk)
    topk.sort()
    
    print('load scores ...')
    score_dict = {}
    target_score = {}
    with open(scores_file, 'r') as f:
        for line in tqdm(f.readlines()):
            line = line.strip().split()
            enroll = line[1]
            test = line[2]
            score = float(line[3])
            if line[0] == '1':
                target_score[test] = score
            if test not in score_dict:
                score_dict[test] = {}
            score_dict[test][enroll] = score

    print('compute Top-K ACC ...')
    topk_num = [0] * len(topk)
    for test in target_score.keys():
        sorted_scores = sorted(list(score_dict[test].values()), reverse=True)
        for i, k in enumerate(topk):
            if target_score[test] >= sorted_scores[min(k - 1, len(sorted_scores) - 1)]:
                topk_num[i] += 1

    print("\n----- {} -----".format(os.path.basename(scores_file)))
    for i, k in enumerate(topk):
        print("SID Top-{} ACC: {:.2f}%".format(k, 100 * topk_num[i] / len(target_score)))


if __name__ == "__main__":
    fire.Fire(main)
