import pandas as pd
import json
import argparse
import nltk
from random import shuffle, seed
import numpy as np
import os

def prepro(args):
    annotations = pd.read_table(args.ann_path, sep='\t', header=None, names=['image', 'caption'])
    images = []
    all = {}
    all['images'] = []
    size = np.size(annotations, 0)
    for i in range(size):
        image_id = annotations['image'][i].split('.')[0]
        if image_id not in images:
            images.append(image_id)
        all['images'].append({'image_id':image_id, 'caption':annotations['caption'][i].lower()})
    seed(123)
    shuffle(images)
    data = {}
    data['val'] = images[:args.num_val]
    data['test'] = images[args.num_val:args.num_val+args.num_test]
    data['train'] = images[args.num_test+args.num_val:]
    for sub in ['val','test','train']:
        subset = {}
        subset['images'] = []
        for caption in all['images']:
            if caption['image_id'] in data[sub]:
                if sub == 'train':
                     caption['caption'] = nltk.word_tokenize(caption['caption'])
                subset['images'].append(caption)
        with open(os.path.join(args.target_path, 'flickr30k_%s.json'%sub), 'w') as f:
            json.dump(subset, f)
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', type=str, default='../data/annotations/results_20130124.token')
    parser.add_argument('--num_val', type=int, default=1000)
    parser.add_argument('--num_test', type=int, default=1000)
    parser.add_argument('--target_path', type=str, default='../data/annotations')
    args = parser.parse_args()
    prepro(args)