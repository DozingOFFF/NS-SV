# license: LGPLv3
# version_14: 2020-01-10
# author: Andreas Nautsch (EURECOM)
# git: https://gitlab.eurecom.fr/nautsch/cllr

from cllr import cllr, min_cllr
import argparse
import pandas

parser = argparse.ArgumentParser(description='Computing Cllr and min Cllr for binary decision classifiers.')
parser.add_argument('-s', dest='score_file', type=str, nargs=1, required=True, help='path to score file')
parser.add_argument('-k', dest='key_file', type=str, nargs=1, required=True,   help='path to key file')
parser.add_argument('-e', dest='compute_eer', action='store_true', help='flag: compute ROCCH-EER')

args = parser.parse_args()
# args = parser.parse_args('-s scores.txt -k key.txt'.split(' '))
# args = parser.parse_args('-s scores.txt -k key.txt -e'.split(' '))

scr = pandas.read_csv(args.score_file[0], sep=' ', header=None).pivot_table(index=0, columns=1, values=2)
key = pandas.read_csv(args.key_file[0], sep=' ', header=None).replace('nontarget', False).replace('target', True).pivot_table(index=0, columns=1, values=2)

tar = scr.values[key.values == True]
non = scr.values[key.values == False]

cllr_act = cllr(tar, non)
if args.compute_eer:
    cllr_min, eer = min_cllr(tar, non, compute_eer=True)
else:
    cllr_min = min_cllr(tar, non)

print("Cllr (min/act): %.3f/%.3f" % (cllr_min, cllr_act))
if args.compute_eer:
    print("     ROCCH-EER: %2.3f%%" % (100*eer))

print("")
