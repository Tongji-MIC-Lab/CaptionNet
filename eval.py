# -*- coding: utf-8 -*-
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from general_eval import GeneralEvalCap
import argparse
import os
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

def eval(args):

    resFile = args.result_file
    if args.dataset == 'flickr30k':
        with open(args.ann_file, 'r') as f:
            gts = json.load(f)
            gts = gts['images']
        with open(os.path.join(args.result_path,resFile), 'r') as f:
            res = json.load(f)
        GeneralEval = GeneralEvalCap(gts, res)
        GeneralEval.evaluate()
        for metric, score in GeneralEval.eval.items():
            print('%s: %.3f' % (metric, score))
        return
    annFile= args.ann_file
    coco = COCO(annFile)
    cocoRes = coco.loadRes(os.path.join(args.result_path,resFile))
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        print ('%s: %.3f'%(metric, score))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--result_file', type=str, default=
    'CA1_b3_captions_val2014_fakecap_results.json')
    parser.add_argument('--result_path', type=str, default=
    'results')
    parser.add_argument('--ann_file', type=str, default='data/annotations/karpathy_split_test.json')
    parser.add_argument('--epoch', type=int, default=10)

    args = parser.parse_args()
    eval(args)