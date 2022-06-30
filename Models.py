import torch
import torch.nn as nn
from build_vocab import Vocabulary
import torchvision.models as models
import torch.nn.functional as F
from math import sqrt
import utility
import numpy as np

def load_model(args, device, test=False):
    if args.model_name == 'GLO':
        model = GLO(args).to(device)
    elif args.model_name == 'CAP':
        model = CAP(args).to(device)
    elif args.model_name == 'TopDown':
        model = TopDown(args).to(device)
    else:
        raise AssertionError("No model named %s\n" % args.model_name)
    multi_devices = [args.device]
    if not test and args.use_multi_gpu:
        multi_devices.extend(args.other_devices)
    model = nn.DataParallel(model, device_ids=multi_devices)
    return model if not test else model.eval()

class GLOEncoder(nn.Module):
    def __init__(self, args):
        super(GLOEncoder, self).__init__()
        if args.cnn == 'resnet':
            if args.resnet == '152':
                self.CNN = models.resnet152(pretrained=True)
                modules = list(self.CNN.children())[:-2]
                self.CNN = nn.Sequential(*modules)
            elif args.resnet == '101':
                self.CNN = models.resnet101(pretrained=True)
                modules = list(self.CNN.children())[:-2]
                self.CNN = nn.Sequential(*modules)
            if args.sequential_length > 0:
                self.linear1 = nn.Sequential(nn.Linear(2048, args.embed_size), nn.ReLU(), nn.Dropout(0.5))
                self.linear2 = nn.Sequential(nn.Linear(int((args.crop_size / 32) ** 2), args.sequential_length))
            else:
                self.linear1 = nn.Sequential(nn.Linear(2048, args.hidden_size), nn.ReLU(), nn.Dropout(0.5))
                self.linear2 = nn.Sequential(nn.Linear(2048, args.hidden_size), nn.ReLU(), nn.Dropout(0.5))
            utility.init_parameter(list(self.linear1.children())[0])
            utility.init_parameter(list(self.linear2.children())[0])
            self.global_cap = nn.LSTM(args.embed_size, args.hidden_size, batch_first=True, num_layers=1)
        else:
            self.CNN = None
        self.sequential_length = args.sequential_length

    def forward(self, images, fine_tune=False):
        batch_size = images.size(0)
        if fine_tune:
            v = self.CNN(images)
        else:
            with torch.no_grad():
                v = self.CNN(images)
        v = v.view(batch_size, 2048, -1)
        if self.sequential_length > 0:
            _, h_c = self.global_cap(self.linear1(self.linear2(v).transpose(1, 2)))
        else:
            v_mean = torch.mean(v.transpose(1, 2), dim=1)
            h_c = (self.linear1(v_mean).unsqueeze(0), self.linear2(v_mean).unsqueeze(0))
        return h_c


class GLODecoder(nn.Module):
    def __init__(self, args):
        super(GLODecoder, self).__init__()
        self.embed = nn.Sequential(nn.Embedding(args.vocab_size, args.embed_size), nn.ReLU(), nn.Dropout(0.5))
        self.lstm = nn.LSTM(args.embed_size, args.hidden_size, num_layers=args.num_layers, batch_first=True)
        self.mlp = nn.Linear(args.hidden_size, args.vocab_size)
        next(self.embed.children()).weight.data.uniform_(-0.1, 0.1)
        utility.init_parameter(self.mlp)
        self.dropout = nn.Dropout(0.5)

    def forward(self, h_c, captions, lengths):
        h_c2 = (h_c[0], h_c[1])
        h_c1 = (h_c2[0].clone().fill_(0), h_c2[0].clone().fill_(0))
        h_c = (torch.cat((h_c1[0], h_c2[0]), 0), torch.cat((h_c1[1], h_c2[1]), 0))
        output, _ = self.lstm(self.embed(captions), h_c)
        output = self.mlp(self.dropout(output))
        return output


class GLO(nn.Module):
    def __init__(self, args):
        super(GLO, self).__init__()
        self.encoder = GLOEncoder(args)
        self.decoder = GLODecoder(args)

    def forward(self, images, captions, lengths, args, vocab=None):
        # To do: add beam search option to SCST
        if not args.SCST:
            h_c = self.encoder(images, fine_tune=args.cnn_fine_tune)
            outputs = self.decoder(h_c, captions, lengths)
            return outputs
        else:
            if self.training:
                v = self.encoder(images, dropout_prob=0.5, fine_tune=args.cnn_fine_tune)
                gen_results, sample_logprobs = self.sample(v, vocab, sample_max=False, dropout_prob=0.5)
                return gen_results, sample_logprobs
            else:
                with torch.no_grad():
                    v = self.encoder(images, fine_tune=args.cnn_fine_tune)
                    greedy_res, _ = self.sample(v, vocab, sample_max=True)
                return greedy_res, _

    def get_params(self):
        return list(self.encoder.linear1.parameters()) \
                 + list(self.encoder.linear2.parameters()) \
                 + list(self.encoder.global_cap.parameters()) \
                 + list(self.decoder.parameters())

    def beam_search(self, image, vocab, beam_size, test=True):
        vocab_size = len(vocab.idx2word)
        h_c = self.encoder(image) if test else image
        max_len = 30
        result = torch.zeros(beam_size, max_len).long().cuda()
        final_result = []
        final_score = []
        input_ = torch.Tensor([vocab.word2idx['<start>']]).view(1, 1).long().cuda()
        mask = torch.zeros(beam_size, vocab_size).cuda()
        for b in range(beam_size):
            mask[b][utility.get_special_tag_id_without_end(vocab)] = float('-inf')
        h_c2 = (h_c[0], h_c[1])
        h_c1 = (h_c2[0].clone().fill_(0), h_c2[0].clone().fill_(0))
        h_c = (torch.cat((h_c1[0], h_c2[0]), 0), torch.cat((h_c1[1], h_c2[1]), 0))
        output, h_c = self.decoder.lstm(self.decoder.embed(input_), h_c)
        output = self.decoder.mlp(output.squeeze(0))
        output = F.log_softmax(output, dim=1) + mask[0]
        topk_prob, topk = output[0].topk(beam_size)
        hs, cs = h_c[0].expand(beam_size, -1), h_c[1].expand(beam_size, -1)
        pre_output = torch.zeros(beam_size * vocab_size).cuda()
        for i in range(beam_size):
            pre_output[i * vocab_size: (i + 1) * vocab_size] = topk_prob[i].expand(vocab_size)
        time_step = 0
        while (True):
            new_result = torch.zeros(beam_size, max_len).long().cuda()
            finish = torch.zeros(beam_size).long().cuda()
            for i in range(beam_size):
                id = topk[i] % vocab_size
                which_sequence = int(topk[i] / vocab_size)
                new_result[i] = result[which_sequence]
                new_result[i][time_step] = id
                if vocab.idx2word[id.item()] == '<end>' or time_step >= max_len:
                    if time_step != 1:
                        final_result.append(new_result[i])
                        final_score.append(pre_output[i * vocab_size])
                    finish[i] = 1
            new_beam_size = beam_size - finish.sum().item()
            new_pre_output = torch.zeros(new_beam_size * vocab_size).cuda()
            new_topk = torch.zeros(new_beam_size).cuda()
            result = torch.zeros(new_beam_size, max_len).long().cuda()
            j = 0
            for i in range(beam_size):
                if finish[i] != 1:
                    new_pre_output[j * vocab_size: (1 + j) * vocab_size] = pre_output[
                                                                           i * vocab_size: (1 + i) * vocab_size]
                    new_topk[j] = topk[i]
                    result[j] = new_result[i]
                    j += 1
            pre_output = new_pre_output
            topk = new_topk
            beam_size = new_beam_size
            if beam_size == 0: break
            new_hs, new_cs = torch.zeros(beam_size, h_c[0].size(1)).cuda(), \
                             torch.zeros(beam_size, h_c[0].size(1)).cuda()
            output = torch.zeros(beam_size * vocab_size).cuda()
            new_mask = torch.zeros(beam_size, vocab_size).cuda()
            for i in range(beam_size):
                which_hidden = int(topk[i] / vocab_size)
                input_ = torch.Tensor([[topk[i] % vocab_size]]).long().view(1, 1).cuda()
                new_mask[i] = mask[which_hidden]
                if not utility.isStopWord(vocab.idx2word[topk[i].item() % vocab_size]):
                    new_mask[i][int(topk[i].item()) % vocab_size] = float('-inf')
                out, h_c = self.decoder.lstm(self.decoder.embed(input_), (hs[which_hidden], cs[which_hidden]))
                out = self.decoder.mlp(out.squeeze(0))
                output[i * vocab_size: (i + 1) * vocab_size] = new_mask[i] + F.log_softmax(out, dim=1).squeeze()
                new_hs[i], new_cs[i] = h_c[0], h_c[1]
            output = output + pre_output
            topk_prob, topk = output.topk(beam_size)
            pre_output = torch.zeros(beam_size * vocab_size).cuda()
            for i in range(beam_size):
                pre_output[i * vocab_size: (i + 1) * vocab_size] = topk_prob[i].expand(vocab_size)
            hs, cs = new_hs, new_cs
            mask = new_mask
            time_step += 1
        final_result = torch.Tensor(final_result[final_score.index(max(final_score))])
        return vocab.tensor2str(final_result) if test else final_result

class CaptionNet(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(CaptionNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.t2h = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, text, hidden):
        h, c = hidden
        i_g = self.i2h(x) + self.h2h(h)

        i_t = i_g[:, :self.hidden_size].sigmoid()
        g_t = i_g[:, self.hidden_size:2 * self.hidden_size].tanh()
        o_f = (i_g[:, 2 * self.hidden_size:] + self.t2h(text)).sigmoid()
        o_t = o_f[:, :self.hidden_size]
        f_t = o_f[:, self.hidden_size:2 * self.hidden_size]
        c_t = torch.mul(g_t, i_t) + torch.mul(c, f_t)
        h_t = torch.mul(c_t.tanh(), o_t)
        self.f_t = torch.sum(f_t).item()

        return h_t, c_t


class CAPEncoder(nn.Module):
    def __init__(self, args):
        super(CAPEncoder, self).__init__()
        if args.cnn == 'resnet':
            if args.resnet == '152':
                self.CNN = models.resnet152(pretrained=True)
                modules = list(self.CNN.children())[:-2]
                self.CNN = nn.Sequential(*modules)
            elif args.resnet == '101':
                self.CNN = models.resnet101(pretrained=True)
                modules = list(self.CNN.children())[:-2]
                self.CNN = nn.Sequential(*modules)
            if args.eliminate_code & 2:
                self.linear1 = nn.Sequential(nn.Linear(2048, args.hidden_size), nn.ReLU(inplace=True))
                self.linear2 = nn.Sequential(nn.Linear(2048, args.hidden_size), nn.ReLU(inplace=True))
                print("****************Ablation study: eliminate IFE***************")
            else:
                self.linear1 = nn.Sequential(nn.Linear(2048, args.embed_size), nn.ReLU(inplace=True))
                self.linear2 = nn.Sequential(nn.Linear(int((args.crop_size / 32) ** 2), 10))
        else:
            self.CNN = None
            if args.eliminate_code & 2:
                self.linear1 = nn.Sequential(nn.Linear(2048, args.hidden_size), nn.ReLU(inplace=True))
                self.linear2 = nn.Sequential(nn.Linear(2048, args.hidden_size), nn.ReLU(inplace=True))
                print("****************Ablation study: eliminate IFE***************")
            else:
                self.linear1 = nn.Sequential(nn.Linear(2048, args.embed_size), nn.ReLU(inplace=True))
                self.linear2 = nn.Sequential(nn.Linear(36, 10))

        self.embed = nn.Sequential(nn.Linear(2048, args.hidden_size), nn.ReLU(inplace=True))
        utility.init_parameter(list(self.linear1.children())[0])
        utility.init_parameter(list(self.linear2.children())[0])
        utility.init_parameter(list(self.embed.children())[0])
        self.global_cap = nn.LSTM(args.embed_size, args.hidden_size, batch_first=True, num_layers=1)
        self.hidden_size = args.hidden_size
        self.cnn = args.cnn
        self.dropout = nn.Dropout(0.5)
        self.eliminate_code = args.eliminate_code

    def forward(self, images, dropout_prob=0.5, fine_tune=False):
        batch_size = images.size(0)
        self.dropout.p = dropout_prob
        if self.cnn == 'resnet':
            if fine_tune:
                cnn_fixed = nn.Sequential(*list(self.CNN.children())[:5])
                cnn_tune = nn.Sequential(*list(self.CNN.children())[5:])
                with torch.no_grad():
                    v = cnn_fixed(images)
                v = cnn_tune(v)
            else:
                with torch.no_grad():
                    v = self.CNN(images)
            v = v.view(batch_size, 2048, -1)
        else:
            v = images.transpose(1, 2)  # Faster R-CNN features
        if self.eliminate_code & 2:
            v_mean = torch.mean(v.transpose(1, 2), dim=1)
            h_c = (self.dropout(self.linear1(v_mean)).unsqueeze(0),
                   self.dropout(self.linear2(v_mean)).unsqueeze(0))
            return h_c, self.dropout(self.embed(v.transpose(1, 2)))
        if self.cnn == 'rcnn':
            _, h_c = self.global_cap(self.dropout(self.linear1(v.transpose(1, 2))))
            return h_c, self.dropout(self.embed(v.transpose(1, 2)))
        else:
            _, h_c = self.global_cap(self.dropout(self.linear1(self.linear2(v).transpose(1, 2))))
            return h_c, self.dropout(self.embed(v.transpose(1, 2)))


class CAPAtt(nn.Module):
    def __init__(self, args):
        super(CAPAtt, self).__init__()
        self.embed = nn.Sequential(nn.Embedding(args.vocab_size, args.embed_size), nn.ReLU(inplace=True))
        self.w_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.w_g = nn.Linear(args.hidden_size, args.hidden_size)
        self.w_c = nn.Linear(args.hidden_size, args.hidden_size)
        self.w_h = nn.Linear(args.hidden_size, 1)
        self.eliminate_code = args.eliminate_code
        if args.eliminate_code & 1:
            self.decoder = nn.LSTMCell(2 * args.hidden_size, args.hidden_size)
            print("****************Ablation study: eliminate CapNet***************")
        else:
            self.decoder = CaptionNet(args.hidden_size, args.hidden_size)
        self.text_cap = nn.LSTMCell(args.embed_size, args.hidden_size)
        self.mlp = nn.Linear(args.hidden_size, args.vocab_size)
        list(self.embed.children())[0].weight.data.uniform_(-0.1, 0.1)
        utility.init_parameter(self.mlp)
        self.dropout = nn.Dropout(0.5)

    def forward(self, v, h_c1, h_c2, input_, dropout_prob=0.5, analyze=False):
        self.dropout.p = dropout_prob
        x1 = self.dropout(self.embed(input_))
        h1, c1 = self.text_cap(x1, h_c1)
        Wv_V = self.w_v(v)
        Wg_H = self.w_g(h1).unsqueeze(1).expand_as(Wv_V)
        Wg_C = self.w_c(h_c2[1]).unsqueeze(1).expand_as(Wv_V)
        if self.eliminate_code & 1:
            alpha = self.w_h(torch.tanh(Wv_V + Wg_H))
        else:
            alpha = self.w_h(torch.tanh(Wv_V + Wg_H + Wg_C))
        att_weights = F.softmax(alpha, dim=1).transpose(1, 2)
        alpha_v = torch.bmm(att_weights, v).squeeze(1)
        if self.eliminate_code & 1:
            h2, c2 = self.decoder(torch.cat((alpha_v, h1), 1), h_c2)
        else:
            h2, c2 = self.decoder(alpha_v, h1, h_c2)
        output = self.mlp(self.dropout(h2))
        if analyze:
            return output, (h1, c1), (h2, c2), self.decoder.f_t
        return output, (h1, c1), (h2, c2), None


class CAPDecoder(nn.Module):
    def __init__(self, args):
        super(CAPDecoder, self).__init__()
        self.att_core = CAPAtt(args)
        self.vocab_size = args.vocab_size

    def forward(self, h_c, captions, lengths):
        outputs = torch.zeros(captions.size(0), captions.size(1), self.vocab_size).cuda()
        (h_c, v) = h_c
        h_c2 = (h_c[0][0], h_c[1][0])
        h_c1 = (h_c2[0].clone().fill_(0), h_c2[0].clone().fill_(0))
        for i in range(captions.size(1)):
            output, h_c1, h_c2, _ = self.att_core(v, h_c1, h_c2, captions[:, i])
            outputs[:, i, :] = output
        return outputs


class CAP(nn.Module):
    def __init__(self, args):
        super(CAP, self).__init__()
        self.encoder = CAPEncoder(args)
        self.decoder = CAPDecoder(args)
        self.cnn = args.cnn

    def forward(self, images, captions, lengths, args, vocab=None):
        if not args.SCST:
            h_c = self.encoder(images, fine_tune=args.cnn_fine_tune)
            outputs = self.decoder(h_c, captions, lengths)
            return outputs
        else:
            if self.training:
                dropout_prob = 0 if args.disable_dropout else 0.5
                v = self.encoder(images, dropout_prob=dropout_prob, fine_tune=args.cnn_fine_tune)
                gen_results, sample_logprobs = self.sample(v, vocab, sample_max=False,
                                                           dropout_prob=dropout_prob, beam_size=args.sample_beam_size)
                return gen_results, sample_logprobs
            else:
                with torch.no_grad():
                    v = self.encoder(images, fine_tune=args.cnn_fine_tune)
                    greedy_res, _ = self.sample(v, vocab, sample_max=True, beam_size=args.baseline_beam_size)
                return greedy_res, _


    def get_params(self):
        return list(self.encoder.linear1.parameters()) \
                 + list(self.encoder.linear2.parameters()) \
                 + list(self.encoder.embed.parameters()) \
                 + list(self.encoder.global_cap.parameters()) \
                 + list(self.decoder.parameters())

    def beam_search(self, image, vocab, beam_size, test=True, dropout_prob=0.0):
        vocab_size = len(vocab)
        finished_beam = 0
        (h_c, v) = self.encoder(image) if test else image
        max_len = 30
        result = torch.zeros(beam_size, max_len).cuda().long()
        final_score = torch.zeros(beam_size).cuda()
        logprob = torch.zeros(max_len + 1, beam_size, max_len).cuda()
        final_result = torch.zeros(beam_size, max_len).cuda().long()
        final_logprob = torch.zeros(beam_size, max_len).cuda()
        input_ = torch.Tensor([vocab.word2idx['<start>']]).view(1).long().cuda()
        mask = torch.zeros(beam_size, vocab_size).cuda()
        for b in range(beam_size):
            mask[b][utility.get_special_tag_id_without_end(vocab)] = float('-inf')
        h_c2 = (h_c[0][0], h_c[1][0])
        h_c1 = (h_c2[0].clone().fill_(0), h_c2[0].clone().fill_(0))
        output, h_c1, h_c2, _ = self.decoder.att_core(v, h_c1, h_c2, input_, dropout_prob=dropout_prob)
        output = F.log_softmax(output, dim=1) + mask[0]
        topk_prob, topk = output[0].topk(beam_size)
        h1s, c1s = [h_c1[0] for _ in range(beam_size)], [h_c1[1] for _ in range(beam_size)]
        h2s, c2s = [h_c2[0] for _ in range(beam_size)], [h_c2[1] for _ in range(beam_size)]
        pre_output = torch.zeros(beam_size * vocab_size).cuda()
        for i in range(beam_size):
            pre_output[i * vocab_size: (i + 1) * vocab_size] = topk_prob[i].expand(vocab_size)
            logprob[0][i][0] = torch.max(output[0])
        time_step = 0
        while True:
            new_result = torch.zeros(beam_size, max_len).cuda().long()
            finish = torch.zeros(beam_size).cuda().long()
            for i in range(beam_size):
                id = topk[i] % vocab_size
                which_sequence = int(topk[i] / vocab_size)
                new_result[i] = result[which_sequence]
                logprob[time_step + 1][i] = logprob[time_step][which_sequence]
                if vocab.idx2word[id.item()] == '<end>' or time_step >= max_len - 1:
                    final_result[finished_beam] = new_result[i]
                    final_logprob[finished_beam] = logprob[time_step + 1][i]
                    final_score[finished_beam] = pre_output[i * vocab_size]
                    finished_beam += 1
                    finish[i] = 1
                    continue
                new_result[i][time_step] = id
            if finish.sum().item() > 0:
                new_beam_size = beam_size - finish.sum().item()
                if new_beam_size == 0:
                    break
                new_pre_output = torch.zeros(new_beam_size * vocab_size).cuda()
                new_topk = torch.zeros(new_beam_size).cuda()
                result = torch.zeros(new_beam_size, max_len).cuda().long()
                j = 0
                for i in range(beam_size):
                    if finish[i] != 1:
                        new_pre_output[j * vocab_size: (1 + j) * vocab_size] = pre_output[
                                                                               i * vocab_size: (1 + i) * vocab_size]
                        new_topk[j] = topk[i]
                        result[j] = new_result[i]
                        j += 1
                pre_output = new_pre_output
                topk = new_topk
                beam_size = new_beam_size
            else:
                result = new_result
            new_h1s, new_c1s = [], []
            new_h2s, new_c2s = [], []
            output = torch.zeros(beam_size * vocab_size).cuda()
            new_mask = torch.zeros(beam_size, vocab_size).cuda()
            for i in range(beam_size):
                which_hidden = int(topk[i] / vocab_size)
                input_ = torch.Tensor([[topk[i] % vocab_size]]).long().view(1).cuda()
                new_mask[i] = mask[which_hidden]
                if not utility.isStopWord(vocab.idx2word[topk[i].item() % vocab_size]):
                    new_mask[i][int(topk[i].item() % vocab_size)] = float('-inf')
                out, h_c1, h_c2, _ = self.decoder.att_core(v, (h1s[which_hidden], c1s[which_hidden]),
                                                           (h2s[which_hidden],c2s[which_hidden]),input_, dropout_prob=dropout_prob)
                out = new_mask[i] + F.log_softmax(out, dim=1).squeeze()
                prev_bad = torch.zeros(vocab_size).cuda()
                if topk[i] % vocab_size in utility.get_bad_ending_id(vocab):
                    prev_bad[vocab.word2idx['<end>']] = float('-inf')
                    out = out + prev_bad
                output[i * vocab_size: (i + 1) * vocab_size] = out
                new_h1s.append(h_c1[0]), new_c1s.append(h_c1[1])
                new_h2s.append(h_c2[0]), new_c2s.append(h_c2[1])
                logprob[time_step + 1][i][time_step + 1] = torch.max(out)
            output = output + pre_output
            topk_prob, topk = output.topk(beam_size)
            pre_output = torch.zeros(beam_size * vocab_size).cuda()
            for i in range(beam_size):
                pre_output[i * vocab_size: (i + 1) * vocab_size] = topk_prob[i].expand(vocab_size)
            h1s, c1s = new_h1s, new_c1s
            h2s, c2s = new_h2s, new_c2s
            mask = new_mask
            time_step += 1
        final_result = final_result[torch.argmax(final_score)]
        final_logprob = final_logprob[torch.argmax(final_score)]
        return vocab.tensor2str(final_result) if test else final_result, final_logprob

    def sample(self, v, vocab, sample_max=False, dropout_prob=0.5, beam_size=1):
        max_len = 30
        batch = v[1].size(0)
        if beam_size > 1:
            seq = torch.zeros(batch, max_len).fill_(0).long().cuda()
            seqLogprobs = torch.zeros(batch, max_len).cuda()
            for i in range(batch):
                seq[i], seqLogprobs[i] = self.beam_search(((v[0][0][:,i].unsqueeze(0), v[0][1][:,i].unsqueeze(0)), 
                                                        v[1][i].unsqueeze(0)), vocab,
                                                        beam_size, test=False, dropout_prob=dropout_prob)
            return seq, seqLogprobs
        (h_c, v) = v
        h_c2 = (h_c[0][0], h_c[1][0])
        h_c1 = (h_c2[0].clone().fill_(0), h_c2[0].clone().fill_(0))
        seq = torch.zeros(batch, max_len).fill_(0).long().cuda()
        seqLogprobs = torch.zeros(batch, max_len).cuda()
        input_ = torch.zeros(batch).fill_(vocab.word2idx['<start>']).long().cuda()
        mask = torch.zeros(batch).fill_(1).long().cuda()
        for t in range(max_len):
            output, h_c1, h_c2, _ = self.decoder.att_core(v, h_c1, h_c2, input_, dropout_prob=dropout_prob)
            logprobs = F.log_softmax(output, dim=1)
            if sample_max == True:
                sampleLogprobs, input_ = torch.max(logprobs, 1)
                input_ = input_.view(-1).long()
            else:
                input_ = torch.distributions.Categorical(logits=logprobs.detach()).sample()
                sampleLogprobs = logprobs.gather(1, input_.unsqueeze(1))
                input_ = input_.view(-1).long()
            input_ = input_ * mask
            seq[:, t] = input_
            seqLogprobs[:, t] = sampleLogprobs.view(-1)
            for b in range(batch):
                if mask[b] == 1 and vocab.idx2word[input_[b].item()] in ['<end>', '<pad>', '<start>']:
                    mask[b] = 0
            if mask.sum() == 0:
                break
        return seq, seqLogprobs


class CAPEnsemble(nn.Module):
    def __init__(self, models):
        super(CAPEnsemble, self).__init__()
        self.models = nn.ModuleList(models)

    def beam_search(self, image, vocab, beam_size):
        vocab_size = len(vocab.idx2word)
        num_models = len(self.models)
        h_c, v = [], []
        for m in self.models:
            (_1, _2) = m.module.encoder(image)
            h_c.append(_1)
            v.append(_2)
        max_len = 30
        result = torch.zeros(beam_size, max_len).long().cuda()
        final_result = []
        final_score = []
        input_ = torch.Tensor([vocab.word2idx['<start>']]).view(1).long().cuda()
        mask = torch.zeros(beam_size, vocab_size).cuda()
        for b in range(beam_size):
            mask[b][utility.get_special_tag_id_without_end(vocab)] = float('-inf')
        h_c2, h_c1 = [], []
        for i in range(num_models):
            h_c2.append((h_c[i][0][0], h_c[i][1][0]))
            h_c1.append((h_c[i][0][0].clone().fill_(0), h_c[i][0][0].clone().fill_(0)))
        output = torch.zeros(1, vocab_size).cuda()
        for i, m in enumerate(self.models):
            _1, h_c1[i], h_c2[i], _ = m.module.decoder.att_core(v[i], h_c1[i], h_c2[i], input_, dropout_prob=0)
            output += F.log_softmax(_1, dim=1) + mask[0]
        output /= num_models
        topk_prob, topk = output[0].topk(beam_size)
        h1s, c1s, h2s, c2s = [], [], [], []
        for i in range(num_models):
            h1s.append(h_c1[i][0].expand(beam_size, -1))
            c1s.append(h_c1[i][1].expand(beam_size, -1))
            h2s.append(h_c2[i][0].expand(beam_size, -1))
            c2s.append(h_c2[i][1].expand(beam_size, -1))
        pre_output = torch.zeros(beam_size * vocab_size).cuda()
        for i in range(beam_size):
            pre_output[i * vocab_size: (i + 1) * vocab_size] = topk_prob[i].expand(vocab_size)
        time_step = 0
        while (True):
            new_result = torch.zeros(beam_size, max_len).long().cuda()
            finish = torch.zeros(beam_size).long().cuda()
            for i in range(beam_size):
                id = topk[i] % vocab_size
                which_sequence = int(topk[i] / vocab_size)
                new_result[i] = result[which_sequence]
                new_result[i][time_step] = id
                if vocab.idx2word[id.item()] == '<end>' or time_step >= max_len - 1:
                    final_result.append(new_result[i])
                    final_score.append(pre_output[i * vocab_size])
                    finish[i] = 1
            if finish.sum().item() > 0:
                new_beam_size = beam_size - finish.sum().item()
                if new_beam_size == 0:
                    break
                new_pre_output = torch.zeros(new_beam_size * vocab_size).cuda()
                new_topk = torch.zeros(new_beam_size).cuda()
                result = torch.zeros(new_beam_size, max_len).cuda().long()
                j = 0
                for i in range(beam_size):
                    if finish[i] != 1:
                        new_pre_output[j * vocab_size: (1 + j) * vocab_size] = pre_output[
                                                                               i * vocab_size: (1 + i) * vocab_size]
                        new_topk[j] = topk[i]
                        result[j] = new_result[i]
                        j += 1
                pre_output = new_pre_output
                topk = new_topk
                beam_size = new_beam_size
            else:
                result = new_result
            new_h1s, new_c1s, new_h2s, new_c2s = [], [], [], []
            for i in range(num_models):
                new_h1s.append(torch.zeros(beam_size, h_c1[0][0].size(1)).cuda())
                new_c1s.append(torch.zeros(beam_size, h_c1[0][0].size(1)).cuda())
                new_h2s.append(torch.zeros(beam_size, h_c1[0][0].size(1)).cuda())
                new_c2s.append(torch.zeros(beam_size, h_c1[0][0].size(1)).cuda())
            output = torch.zeros(beam_size * vocab_size).cuda()
            new_mask = torch.zeros(beam_size, vocab_size).cuda()
            for i in range(beam_size):
                which_hidden = int(topk[i] / vocab_size)
                input_ = torch.Tensor([[topk[i] % vocab_size]]).long().view(1).cuda()
                new_mask[i] = mask[which_hidden]
                if not utility.isStopWord(vocab.idx2word[topk[i].item() % vocab_size]):
                    new_mask[i][int(topk[i].item()) % vocab_size] = float('-inf')
                out = torch.zeros(vocab_size).cuda()
                for j, m in enumerate(self.models):
                    _1, h_c1[j], h_c2[j], _ = m.module.decoder.att_core(v[j],
                                                                 (h1s[j][which_hidden].unsqueeze(0),
                                                                  c1s[j][which_hidden].unsqueeze(0)),
                                                                 (h2s[j][which_hidden].unsqueeze(0),
                                                                  c2s[j][which_hidden].unsqueeze(0)),
                                                                 input_)
                    out += new_mask[i] + F.log_softmax(_1, dim=1).squeeze()
                out /= num_models
                prev_bad = torch.zeros(vocab_size).cuda()
                if topk[i] % vocab_size in utility.get_bad_ending_id(vocab):
                    prev_bad[vocab.word2idx['<end>']] = float('-inf')
                    out = out + prev_bad
                output[i * vocab_size: (i + 1) * vocab_size] = out
                for j in range(num_models):
                    new_h1s[j][i], new_c1s[j][i] = h_c1[j][0], h_c1[j][1]
                    new_h2s[j][i], new_c2s[j][i] = h_c2[j][0], h_c2[j][1]
            output = output + pre_output
            topk_prob, topk = output.topk(beam_size)
            pre_output = torch.zeros(beam_size * vocab_size).cuda()
            for i in range(beam_size):
                pre_output[i * vocab_size: (i + 1) * vocab_size] = topk_prob[i].expand(vocab_size)
            h1s, c1s = new_h1s, new_c1s
            h2s, c2s = new_h2s, new_c2s
            mask = new_mask
            time_step += 1
        final_result = final_result[final_score.index(max(final_score))]
        return vocab.tensor2str(final_result)


class TopDownEncoder(nn.Module):
    def __init__(self, args):
        super(TopDownEncoder, self).__init__()
        if args.cnn == 'resnet':
            if args.resnet == '152':
                self.CNN = models.resnet152(pretrained=True)
                modules = list(self.CNN.children())[:-2]
                self.CNN = nn.Sequential(*modules)
        else:
            self.CNN = None
        self.linear1 = nn.Sequential(nn.Linear(2048, args.hidden_size), nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Linear(2048, args.hidden_size), nn.ReLU(inplace=True))
        utility.init_parameter(list(self.linear1.children())[0])
        utility.init_parameter(list(self.linear2.children())[0])
        self.dropout = nn.Dropout(0.5)
        self.cnn = args.cnn

    def forward(self, images, dropout_prob=0.5, fine_tune=False):
        self.dropout.p = dropout_prob
        if self.cnn == 'rcnn':
            v = images
            v_mean = torch.mean(v, dim=1)
        else:
            batch_size = images.size(0)
            if fine_tune:
                cnn_fixed = nn.Sequential(*list(self.CNN.children())[:5])
                cnn_tune = nn.Sequential(*list(self.CNN.children())[5:])
                with torch.no_grad():
                    v = cnn_fixed(images)
                v = cnn_tune(v)
            else:
                with torch.no_grad():
                    v = self.CNN(images)
            v = v.view(batch_size, 2048, -1).transpose(1, 2)
            v_mean = torch.mean(v, dim=1)
        return self.dropout(self.linear2(v)), self.dropout(self.linear1(v_mean))


class TopDownAtt(nn.Module):
    def __init__(self, args):
        super(TopDownAtt, self).__init__()
        self.embed = nn.Sequential(nn.Embedding(args.vocab_size, args.embed_size), nn.ReLU(inplace=True))
        self.w_g = nn.Linear(args.hidden_size, args.hidden_size)
        self.w_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.w_h = nn.Linear(args.hidden_size, 1)
        self.att_lstm = nn.LSTMCell(2 * args.hidden_size + args.embed_size, args.hidden_size)
        self.lan_lstm = nn.LSTMCell(2 * args.hidden_size, args.hidden_size)
        self.mlp = nn.Linear(args.hidden_size, args.vocab_size)
        list(self.embed.children())[0].weight.data.uniform_(-0.1, 0.1)
        utility.init_parameter(self.mlp)
        self.dropout = nn.Dropout(0.5)

    def forward(self, v, v_mean, h_c1, h_c2, input_, dropout_prob=0.5):
        self.dropout.p = dropout_prob
        x1 = self.dropout(self.embed(input_))
        h1, c1 = self.att_lstm(torch.cat((h_c2[0], v_mean, x1), 1), h_c1)
        Wg_H = self.w_g(h1).unsqueeze(1).expand_as(v)
        alpha = self.w_h(torch.tanh(self.w_v(v) + Wg_H))
        att_weights = F.softmax(alpha, dim=1).transpose(1, 2)
        alpha_v = torch.bmm(att_weights, v).squeeze(1)
        h2, c2 = self.lan_lstm(torch.cat((alpha_v, h1), 1), h_c2)
        output = self.mlp(self.dropout(h2))
        return output, (h1, c1), (h2, c2)


class TopDownDecoder(nn.Module):
    def __init__(self, args):
        super(TopDownDecoder, self).__init__()
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.att_core = TopDownAtt(args)

    def forward(self, v, captions, lengths):
        outputs = torch.zeros(captions.size(0), captions.size(1), self.vocab_size).cuda()
        (v, v_mean) = v
        h_c2 = (torch.zeros(v.size(0), self.hidden_size).cuda(), torch.zeros(v.size(0), self.hidden_size).cuda())
        h_c1 = (torch.zeros(v.size(0), self.hidden_size).cuda(), torch.zeros(v.size(0), self.hidden_size).cuda())
        for i in range(captions.size(1)):
            output, h_c1, h_c2 = self.att_core(v, v_mean, h_c1, h_c2, captions[:, i])
            outputs[:, i, :] = output
        return outputs


class TopDown(nn.Module):
    def __init__(self, args):
        super(TopDown, self).__init__()
        self.hidden_size = args.hidden_size
        self.encoder = TopDownEncoder(args)
        self.decoder = TopDownDecoder(args)

    def forward(self, images, captions, lengths, args, vocab=None):
        if not args.SCST:
            v = self.encoder(images, fine_tune=args.cnn_fine_tune)
            outputs = self.decoder(v, captions, lengths)
            return outputs
        else:
            if self.training:
                dropout_prob = 0 if args.disable_dropout else 0.5
                v = self.encoder(images, dropout_prob=dropout_prob, fine_tune=args.cnn_fine_tune)
                gen_results, sample_logprobs = self.sample(v, vocab, sample_max=False,
                                                           dropout_prob=dropout_prob, beam_size=args.sample_beam_size)
                return gen_results, sample_logprobs
            else:
                with torch.no_grad():
                    v = self.encoder(images, fine_tune=args.cnn_fine_tune)
                    greedy_res, _ = self.sample(v, vocab, sample_max=True, beam_size=args.baseline_beam_size)
                return greedy_res, _

    def get_params(self):
        return list(self.encoder.linear1.parameters()) \
                + list(self.encoder.linear2.parameters()) \
                + list(self.decoder.parameters())

    def beam_search(self, image, vocab, beam_size, test=True, dropout_prob=0.0):
        vocab_size = len(vocab)
        finished_beam = 0
        (v, v_mean) = self.encoder(image) if test else image
        max_len = 30
        result = torch.zeros(beam_size, max_len).long().cuda()
        logprob = torch.zeros(max_len + 1, beam_size, max_len).cuda()
        final_result = torch.zeros(beam_size, max_len).long().cuda()
        final_logprob = torch.zeros(beam_size, max_len).cuda()
        final_score = torch.zeros(beam_size).cuda()
        input_ = torch.Tensor([vocab.word2idx['<start>']]).view(1).long().cuda()
        mask = torch.zeros(beam_size, vocab_size).cuda()
        for b in range(beam_size):
            mask[b][utility.get_special_tag_id_without_end(vocab)] = float('-inf')
        h_c2 = (torch.zeros(v.size(0), self.hidden_size).cuda(), torch.zeros(v.size(0), self.hidden_size).cuda())
        h_c1 = (torch.zeros(v.size(0), self.hidden_size).cuda(), torch.zeros(v.size(0), self.hidden_size).cuda())
        output, h_c1, h_c2 = self.decoder.att_core(v, v_mean, h_c1, h_c2, input_, dropout_prob=dropout_prob)
        output = F.log_softmax(output, dim=1) + mask[0]
        topk_prob, topk = output[0].topk(beam_size)
        h1s, c1s = h_c1[0].expand(beam_size, -1), h_c1[1].expand(beam_size, -1)
        h2s, c2s = h_c2[0].expand(beam_size, -1), h_c2[1].expand(beam_size, -1)
        pre_output = torch.zeros(beam_size * vocab_size).cuda()
        for i in range(beam_size):
            pre_output[i * vocab_size: (i + 1) * vocab_size] = topk_prob[i].expand(vocab_size)
            logprob[0][i][0] = torch.max(output[0])
        time_step = 0
        while True:
            new_result = torch.zeros(beam_size, max_len).long().cuda()
            finish = torch.zeros(beam_size).long().cuda()
            for i in range(beam_size):
                idx = topk[i] % vocab_size
                which_sequence = int(topk[i] / vocab_size)
                new_result[i] = result[which_sequence]
                logprob[time_step + 1][i] = logprob[time_step][which_sequence]
                if vocab.idx2word[idx.item()] == '<end>' or time_step >= max_len - 1:
                    if time_step != 1:
                        final_result[finished_beam] = new_result[i]
                        final_logprob[finished_beam] = logprob[time_step + 1][i]
                        final_score[finished_beam] = pre_output[i * vocab_size]
                        finished_beam += 1
                    finish[i] = 1
                    continue
                new_result[i][time_step] = idx
            if finish.sum().item() > 0:
                new_beam_size = beam_size - finish.sum().item()
                if new_beam_size == 0:
                    break
                new_pre_output = torch.zeros(new_beam_size * vocab_size).cuda()
                new_topk = torch.zeros(new_beam_size).cuda()
                result = torch.zeros(new_beam_size, max_len).long().cuda()
                j = 0
                for i in range(beam_size):
                    if finish[i] != 1:
                        new_pre_output[j * vocab_size: (1 + j) * vocab_size] = pre_output[
                                                                               i * vocab_size: (1 + i) * vocab_size]
                        new_topk[j] = topk[i]
                        result[j] = new_result[i]
                        j += 1
                pre_output = new_pre_output
                topk = new_topk
                beam_size = new_beam_size
            else:
                result = new_result
            new_h1s, new_c1s = torch.zeros(beam_size, h_c1[0].size(1)).cuda(), torch.zeros(beam_size,
                                                                                           h_c1[0].size(1)).cuda()
            new_h2s, new_c2s = torch.zeros(beam_size, h_c1[0].size(1)).cuda(), torch.zeros(beam_size,
                                                                                           h_c1[0].size(1)).cuda()
            output = torch.zeros(beam_size * vocab_size).cuda()
            new_mask = torch.zeros(beam_size, vocab_size).cuda()
            for i in range(beam_size):
                which_hidden = int(topk[i] / vocab_size)
                input_ = torch.Tensor([[topk[i] % vocab_size]]).long().view(1).cuda()
                new_mask[i] = mask[which_hidden]
                if not utility.isStopWord(vocab.idx2word[topk[i].item() % vocab_size]):
                    new_mask[i][int(topk[i].item()) % vocab_size] = float('-inf')
                out, h_c1, h_c2 = self.decoder.att_core(v, v_mean,
                                                        (h1s[which_hidden].unsqueeze(0),
                                                         c1s[which_hidden].unsqueeze(0)),
                                                        (h2s[which_hidden].unsqueeze(0),
                                                         c2s[which_hidden].unsqueeze(0)),
                                                        input_, dropout_prob=dropout_prob)
                out = new_mask[i] + F.log_softmax(out, dim=1).squeeze()
                prev_bad = torch.zeros(vocab_size).cuda()
                if topk[i] % vocab_size in utility.get_bad_ending_id(vocab):
                    prev_bad[vocab.word2idx['<end>']] = float('-inf')
                    out = out + prev_bad
                output[i * vocab_size: (i + 1) * vocab_size] = out
                new_h1s[i], new_c1s[i] = h_c1[0], h_c1[1]
                new_h2s[i], new_c2s[i] = h_c2[0], h_c2[1]
                logprob[time_step + 1][i][time_step + 1] = torch.max(out)
            output = output + pre_output
            topk_prob, topk = output.topk(beam_size)
            pre_output = torch.zeros(beam_size * vocab_size).cuda()
            for i in range(beam_size):
                pre_output[i * vocab_size: (i + 1) * vocab_size] = topk_prob[i].expand(vocab_size)
            h1s, c1s = new_h1s, new_c1s
            h2s, c2s = new_h2s, new_c2s
            mask = new_mask
            time_step += 1
        final_result = final_result[torch.argmax(final_score)]
        final_logprob = final_logprob[torch.argmax(final_score)]
        return vocab.tensor2str(final_result) if test else final_result, final_logprob

    def sample(self, v, vocab, sample_max=False, dropout_prob=0.5, beam_size=1):
        max_len = 30
        batch = v[0].size(0)
        if beam_size > 1:
            seq = torch.zeros(batch, max_len).fill_(0).long().cuda()
            seqLogprobs = torch.zeros(batch, max_len).cuda()
            for i in range(batch):
                seq[i], seqLogprobs[i] = self.beam_search((v[0][i].unsqueeze(0), v[1][i].unsqueeze(0)), vocab,
                                                          beam_size, test=False, dropout_prob=dropout_prob)
            return seq, seqLogprobs
        (v, v_mean) = v
        h_c2 = (torch.zeros(v.size(0), self.hidden_size).cuda(), torch.zeros(v.size(0), self.hidden_size).cuda())
        h_c1 = (torch.zeros(v.size(0), self.hidden_size).cuda(), torch.zeros(v.size(0), self.hidden_size).cuda())
        seq = torch.zeros(batch, max_len).fill_(0).long().cuda()
        seqLogprobs = torch.zeros(batch, max_len).cuda()
        input_ = torch.zeros(batch).fill_(vocab.word2idx['<start>']).long().cuda()
        mask = torch.zeros(batch).fill_(1).long().cuda()
        for t in range(max_len):
            output, h_c1, h_c2 = self.decoder.att_core(v, v_mean, h_c1, h_c2, input_, dropout_prob=dropout_prob)
            logprobs = F.log_softmax(output, dim=1)
            if sample_max == True:
                sampleLogprobs, input_ = torch.max(logprobs, 1)
                input_ = input_.view(-1).long()
            else:
                input_ = torch.distributions.Categorical(logits=logprobs.detach()).sample()
                sampleLogprobs = logprobs.gather(1, input_.unsqueeze(1))
                input_ = input_.view(-1).long()
            input_ = input_ * mask
            seq[:, t] = input_
            seqLogprobs[:, t] = sampleLogprobs.view(-1)
            for b in range(batch):
                if mask[b] == 1 and vocab.idx2word[input_[b].item()] in ['<end>', '<pad>', '<start>']:
                    mask[b] = 0
            if mask.sum() == 0:
                break
        return seq, seqLogprobs
