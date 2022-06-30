# Evaluation class for flickr30k dataset
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

class GeneralEvalCap:
    def __init__(self, gts, res):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.gts = {}
        image_ids = []
        for item in gts:
            if item['image_id'] not in image_ids:
                image_ids.append(item['image_id'])
        for id in image_ids:
            self.gts[id] = []
            for item in gts:
                if id == item['image_id']:
                    self.gts[id].append(item)
        self.res = {}
        for item in res:
            self.res[item['image_id']] = [item]

    def evaluate(self):

        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(self.gts)
        res = tokenizer.tokenize(self.res)


        # =================================================
        # Set up scorers
        # =================================================

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            #(Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            #(Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:

            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)

        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]