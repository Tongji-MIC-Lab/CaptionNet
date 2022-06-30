# Ref: https://github.com/ruotianluo/self-critical.pytorch/blob/master/scripts/prepro_ngrams.py

import json
import argparse
from six.moves import cPickle
from collections import defaultdict

def precook(s, n=4, out=False):
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):
    return [precook(ref, n) for ref in refs]


def create_crefs(refs):
    crefs = []
    for ref in refs:
        # ref is a list of 5 captions
        crefs.append(cook_refs(ref))
    return crefs


def compute_doc_freq(crefs):
    document_frequency = defaultdict(float)
    for refs in crefs:
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
            document_frequency[ngram] += 1
    return document_frequency


def build_dict(imgs):
    tmp_dict = {}
    for ann in imgs['annotations']:
        if not ann['image_id'] in tmp_dict.keys():
            tmp_dict[ann['image_id']] = []
        tmp_dict[ann['image_id']].append(' '.join(ann['caption']))
    count_imgs = len(tmp_dict.keys())
    print('total imgs:', count_imgs)
    refs_words = []
    for key in tmp_dict.keys():
        refs_words.append(tmp_dict[key])
    ngram_words = compute_doc_freq(create_crefs(refs_words))
    #ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))

    return ngram_words, count_imgs


def main(params):

    imgs = json.load(open(params['input_json'], 'r'))
    ngram_words, ref_len = build_dict(imgs)

    cPickle.dump({'document_frequency': ngram_words, 'ref_len': ref_len},
                 open(params['output_pkl'] + '-words.p', 'wb+'), protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', default='../data/annotations/karpathy_split_train.json', help='input json file to process into hdf5')
    parser.add_argument('--output_pkl', default='../data/coco-train', help='output pickle file')
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
