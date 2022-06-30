
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
from utility import clean

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def tensor2str(self, tensor):
        result = []
        for i in range(len(tensor)):
            if self.idx2word[tensor[i].item()] not in ['<pad>', '<end>']:
                result.append(self.idx2word[tensor[i].item()])
            else:
                break
        return ' '.join(result)

def build_vocab(args):
    """Build a simple vocabulary wrapper."""
    json_file = args.caption_path
    threshold = args.threshold
    counter = Counter()
    if args.dataset == 'coco':
        coco = COCO(json_file)
        ids = coco.anns.keys()

        for i, id in enumerate(ids):
            caption = clean(coco.anns[id]['caption'])
            counter.update(caption)

    elif args.dataset == 'flickr30k':
        import json
        with open(json_file, 'r') as f:
            dataset = json.load(f)
        for i, c in enumerate(dataset['images']):
            caption = clean(c['caption'])
            counter.update(caption)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
        print (word)
    return vocab

def main(args):
    vocab = build_vocab(args)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/annotations/karpathy_split_train.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=5,
                        help='minimum word count threshold')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='')
    args = parser.parse_args()
    main(args)