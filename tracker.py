import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
import logging
import os
import re
import json
from collections import defaultdict
from pprint import pformat

with open('value_len_config.json') as f:
    value_len_config = json.load(f)


def pad(seqs, emb, device, pad=0):
    lens = [len(s) for s in seqs]
    max_len = max(lens)
    padded = torch.LongTensor([s + (max_len - l) * [pad] for s, l in zip(seqs, lens)])
    return emb(padded.to(device)), lens


def run_rnn(rnn, inputs, lens):
    order = np.argsort(lens)[::-1].tolist()
    reindexed = inputs.index_select(0, inputs.data.new(order).long())
    reindexed_lens = [lens[i] for i in order]
    packed = nn.utils.rnn.pack_padded_sequence(reindexed, reindexed_lens, batch_first=True)
    outputs, _ = rnn(packed)
    padded, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, padding_value=0.)
    reverse_order = np.argsort(order).tolist()
    recovered = padded.index_select(0, inputs.data.new(reverse_order).long())
    return recovered


class FixedEmbedding(nn.Embedding):
    """
    this is the same as `nn.Embedding` but detaches the result from the graph and has dropout after lookup.
    """

    def __init__(self, *args, dropout=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = dropout

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        out.detach_()
        return F.dropout(out, self.dropout, self.training)


class Tracker(nn.Module):

    def __init__(self, args, ontology, vocab):
        super().__init__()
        self.optimizer = None
        self.args = args
        ed, hd, dropout = args.ed, args.hd, args.dropout['global']
        self.ontology = ontology
        self.vocab = vocab
        self.emb_fixed = FixedEmbedding(len(vocab), ed, dropout=dropout)

        self.domain_encoder = nn.Sequential(nn.Linear(ed, hd), nn.ReLU(), nn.Linear(hd, hd))
        self.utt_encoder = RNNEncoder(ed, hd)
        self.slot_encoder = nn.Sequential(nn.Linear(ed, hd), nn.ReLU(), nn.Linear(hd, hd))

        self.du_attention = DotAttention(hd, 2 * hd, hd)
        self.domain_scorer = nn.Sequential(nn.Linear(3 * hd, hd), nn.Tanh(), nn.Linear(hd, 1))

        self.su_attention = DotAttention(hd, 2 * hd, hd)
        self.slot_scorer = nn.Sequential(nn.Linear(3 * hd, hd), nn.Tanh(), nn.Linear(hd, 2))
        self.span_value_scorer = nn.Sequential(nn.Linear(3 * hd, hd), nn.Tanh(), nn.Linear(hd, 2))

        self.uu_attention = DotAttention(4 * hd, 4 * hd, hd)
        self.uds_encoder = RNNEncoder(4 * hd, hd)

        self.summer = Summer(2 * hd, hd)
        self.pointer_network = PointerNetwork(hd)

    @property
    def device(self):
        if self.args.gpu is not None and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.args.lr)

    def load_emb(self, words):
        new = self.emb_fixed.weight.data.new
        self.emb_fixed.weight.data.copy_(new(words))

    def get_emb(self, words):
        return self.emb_fixed(torch.LongTensor(words).to(self.device))

    def forward(self, batch):
        # encode domains
        batch_num = len(batch)
        all_str_domains = self.ontology.domains
        all_domain_num = len(self.ontology.domains)
        all_numeric_domains = self.ontology.num['domains']
        all_domain_enc = self.domain_encoder(self.get_emb(all_numeric_domains))  # dl*hd

        # encode slots
        all_slot_enc = {}
        for d in all_numeric_domains:
            domain_slot_enc = []
            for s in self.ontology.num[str(d)]:
                slot_enc = self.slot_encoder(self.get_emb(s))
                slot_enc = torch.mean(slot_enc, dim=0)
                domain_slot_enc.append(slot_enc)
            all_slot_enc[d] = torch.stack(domain_slot_enc, dim=0)

        # encode utterance
        eos = self.vocab.word2index('<eos>')
        utterance, utterance_lens = pad([t.num['transcript'] for t in batch], self.emb_fixed, self.device, pad=eos)
        batch_utterance_enc = self.utt_encoder(utterance, utterance_lens)  # b*ul*2hd

        loss = 0
        predictions = []
        for b in range(batch_num):
            prediction = {}
            # predict the domain
            utterance_enc = batch_utterance_enc[b, :utterance_lens[b], :]  # ul*2hd
            du_attention = self.du_attention(all_domain_enc, utterance_enc)  # dn*2hd
            if batch[b].last_domain == '' or batch[b].num['last_domain'] not in all_numeric_domains:
                last_domain_enc = torch.zeros_like(all_domain_enc[0])
            else:
                last_domain_index = all_numeric_domains.index(batch[b].num['last_domain'])
                last_domain_enc = all_domain_enc[last_domain_index]  # hd
            tiled_pre_domain_enc = last_domain_enc.unsqueeze(0).repeat(all_domain_num, 1)  # dn*hd
            domain_logits = self.domain_scorer(torch.cat([du_attention, tiled_pre_domain_enc], dim=1)).squeeze(1)  # dn
            domain_p = F.softmax(domain_logits, dim=0)
            if self.training:
                predicted_domain_index = all_numeric_domains.index(batch[b].num['domain'])  # 1
                domain_truth = np.zeros(all_domain_num)
                domain_truth[predicted_domain_index] = 1
                domain_truth = torch.FloatTensor(domain_truth).to(self.device)
                loss += F.binary_cross_entropy(domain_p, domain_truth)
            else:
                predicted_domain_index = domain_p.argmax(dim=0).cpu().tolist()  # 1
            predicted_domain_enc = all_domain_enc[predicted_domain_index]  # hd
            predicted_domain = all_numeric_domains[predicted_domain_index]
            prediction['domain'] = all_str_domains[predicted_domain_index]
            if self.training and not batch[b].turn_label:
                prediction['slots'] = []
                predictions.append(prediction)
                continue

            # predict the slots
            current_numeric_slots = self.ontology.num[str(predicted_domain)]
            current_str_slots = self.ontology.domain_slots[prediction['domain']]
            current_slot_num = len(current_numeric_slots)
            current_slot_enc = all_slot_enc[predicted_domain]  # sn*hd
            su_attentin = self.su_attention(current_slot_enc, utterance_enc)  # sn*2hd
            tiled_domain_enc = predicted_domain_enc.unsqueeze(0).repeat(current_slot_num, 1)  # sn*hd
            slot_logits = self.slot_scorer(torch.cat([su_attentin, tiled_domain_enc], dim=1))  # sn*2
            slot_p = F.softmax(slot_logits, dim=1)  # sn*2
            if self.training:
                slot_truth = np.zeros((current_slot_num, 2))  # not mentioned, mentioned
                span_value_truth = np.zeros((current_slot_num, 2))  # not span value, span value
                for ts, tvt in zip(batch[b].turn_slots, batch[b].turn_value_type):
                    tsi = current_str_slots.index(ts)
                    slot_truth[tsi, 1] = 1
                    span_value_truth[tsi] = tvt
                slot_truth[slot_truth.sum(axis=1) == 0, 0] = 1
                mentioned_slot_index = slot_truth[:, 1] == 1
                mentioned_slot_num = sum(mentioned_slot_index)
                if mentioned_slot_num == 0:
                    prediction['slots'] = []
                    predictions.append(prediction)
                    continue
                span_value_truth = span_value_truth[mentioned_slot_index]
                span_value_truth[span_value_truth.sum(axis=1) == 0, 0] = 1
                slot_truth = torch.FloatTensor(slot_truth).to(self.device)  # sn*2
                span_value_truth = torch.FloatTensor(span_value_truth).to(self.device)  # msn*2
                loss += F.binary_cross_entropy(slot_p, slot_truth)/current_slot_num
                not_mentioned_slot_index = slot_truth[:, 0] == 1
                mentioned_slot_index = slot_truth[:, 1] == 1
                mentioned_slot_su = su_attentin[mentioned_slot_index]  # msn*2hd
                tiled_domain_enc = predicted_domain_enc.unsqueeze(0).repeat(mentioned_slot_num, 1)  # msn*hd
                # msn*2
                span_value_logits = self.span_value_scorer(torch.cat([mentioned_slot_su, tiled_domain_enc], dim=1))
                span_value_p = F.softmax(span_value_logits, dim=1)  # msn*2
                loss += F.binary_cross_entropy(span_value_p, span_value_truth)/mentioned_slot_num
                # not_span_value_index = span_value_truth[:, 0] == 1
                span_value_index = span_value_truth[:, 1] == 1
            else:
                not_mentioned_slot_index = slot_p.argmax(dim=1) == 0  #
                mentioned_slot_index = slot_p.argmax(dim=1) == 1  # psn
                mentioned_slot_num = sum(mentioned_slot_index)
                if mentioned_slot_num == 0:
                    prediction['slots'] = []
                    predictions.append(prediction)
                    continue
                mentioned_slot_su = su_attentin[mentioned_slot_index]  # psn*2hd
                tiled_domain_enc = predicted_domain_enc.unsqueeze(0).repeat(mentioned_slot_num, 1)  # msn*hd
                # msn*2
                span_value_logits = self.span_value_scorer(torch.cat([mentioned_slot_su, tiled_domain_enc], dim=1))
                span_value_p = F.softmax(span_value_logits, dim=1)  # psn*2
                # not_span_value_index = span_value_p.argmax(dim=1) == 0
                span_value_index = span_value_p.argmax(dim=1) == 1
            predicted_slot_enc = current_slot_enc[mentioned_slot_index]
            svslot_enc = predicted_slot_enc[span_value_index]  # svn*hd

            # predict the values
            span_value_num = sum(span_value_index)
            mentioned_slot_index_list = [i for i, m in enumerate(mentioned_slot_index) if m]
            span_value_index_list = [m for s, m in zip(span_value_index, mentioned_slot_index_list) if s]
            predicted_values = -torch.ones((current_slot_num, 2), dtype=torch.int32)  # sn*2
            predicted_values[not_mentioned_slot_index, 1] = 0
            predicted_values[mentioned_slot_index, 1] = 1
            span_value_loss = 0
            for s in range(span_value_num):
                predicted_slot = current_str_slots[span_value_index_list[s]]
                domain_slot = torch.cat([predicted_domain_enc, svslot_enc[s]], dim=0).unsqueeze(0)  # 1*2hd
                domain_slot = domain_slot.repeat(utterance_lens[b], 1)  # ul*2hd
                uds = torch.cat([domain_slot, utterance_enc], dim=1)  # ul*4hd
                uds_attention = self.uu_attention(uds, uds).unsqueeze(0)  # 1*ul*4hd
                uds_enc = self.uds_encoder(uds_attention, [utterance_lens[b]]).squeeze(0)  # ul*2hd
                init = self.summer(uds_enc)  # 2hd
                p1, p2 = self.pointer_network(init, uds_enc)  # ul
                outer = p1.unsqueeze(dim=1).mm(p2.unsqueeze(dim=0))  # ul*ul
                outer = outer.triu().tril(value_len_config['{}-{}'.format(prediction['domain'], predicted_slot)])
                start = outer.max(dim=1)[0].argmax(dim=0, keepdim=True)
                end = outer.max(dim=0)[0].argmax(dim=0, keepdim=True)
                predicted_values[span_value_index_list[s]] = torch.cat([start, end], dim=0)
                if self.training:
                    start_truth = np.zeros(utterance_lens[b])
                    start_truth[batch[b].turn_value_start[s]] = 1
                    start_truth = torch.FloatTensor(start_truth).to(self.device)
                    end_truth = np.zeros(utterance_lens[b])
                    end_truth[batch[b].turn_value_end[s]] = 1
                    end_truth = torch.FloatTensor(end_truth).to(self.device)
                    span_value_loss += F.binary_cross_entropy(p1, start_truth)
                    span_value_loss += F.binary_cross_entropy(p2, end_truth)
            if span_value_num > 0:
                loss += span_value_loss/span_value_num

            # get predictions
            prediction['slots'] = []
            for i in range(current_slot_num):
                slot_strs = self.ontology.domain_slots[prediction['domain']]
                if predicted_values[i, 0] == -1 and predicted_values[i, 1] != -1:
                    if predicted_values[i, 1] == 1:
                        prediction['slots'].append(('{}-{}'.format(prediction['domain'], slot_strs[i]), ''))
                elif predicted_values[i, 0] != -1 and predicted_values[i, 1] != -1:
                    start, end = predicted_values[i, 0].cpu().tolist(), predicted_values[i, 1].cpu().tolist()
                    prediction['slots'].append(('{}-{}'.format(prediction['domain'], slot_strs[i]),
                                                ' '.join(batch[b].transcript[start:end + 1])))
            predictions.append(prediction)
        loss /= batch_num
        return loss, predictions

    def get_train_logger(self):
        logger = logging.getLogger('train-{}'.format(self.__class__.__name__))
        formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.args.dout, 'train.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def run_train(self, train, dev, args):
        track = defaultdict(list)
        iteration = 0
        best = {}
        logger = self.get_train_logger()
        if self.optimizer is None:
            self.set_optimizer()

        for epoch in range(args.epoch):
            logger.info('starting epoch {}'.format(epoch))

            # train and update parameters
            self.train()
            for batch in train.batch(batch_size=args.batch_size, shuffle=True):
                iteration += 1
                self.zero_grad()
                loss, predictions = self.forward(batch)
                loss.backward()
                self.optimizer.step()
                track['loss'].append(loss.item())

            # evalute on train and dev
            summary = {'iteration': iteration, 'epoch': epoch}
            for k, v in track.items():
                summary[k] = sum(v) / len(v)
            summary.update({'eval_train_{}'.format(k): v for k, v in self.run_eval(train, args).items()})
            summary.update({'eval_dev_{}'.format(k): v for k, v in self.run_eval(dev, args).items()})

            # do early stopping saves
            stop_key = 'eval_dev_{}'.format(args.stop)
            train_key = 'eval_train_{}'.format(args.stop)
            if best.get(stop_key, 0) <= summary[stop_key]:
                best_dev = '{:f}'.format(summary[stop_key])
                best_train = '{:f}'.format(summary[train_key])
                best.update(summary)
                self.save(
                    best,
                    identifier='epoch={epoch},iter={iteration},train_{key}={train},dev_{key}={dev}'.format(
                        epoch=epoch, iteration=iteration, train=best_train, dev=best_dev, key=args.stop,
                    )
                )
                self.prune_saves()
                dev.record_preds(
                    preds=self.run_pred(dev, self.args),
                    to_file=os.path.join(self.args.dout, 'dev.pred.json'),
                )
            summary.update({'best_{}'.format(k): v for k, v in best.items()})
            logger.info(pformat(summary))
            track.clear()

    def run_pred(self, dev, args):
        self.eval()
        predictions = []
        for batch in dev.batch(batch_size=args.batch_size):
            loss, batch_predictions = self.forward(batch)
            predictions += batch_predictions
        return predictions

    def run_eval(self, dev, args):
        predictions = self.run_pred(dev, args)
        return dev.evaluate_preds(predictions)

    def save_config(self):
        fname = '{}/config.json'.format(self.args.dout)
        with open(fname, 'wt') as f:
            logging.info('saving config to {}'.format(fname))
            json.dump(vars(self.args), f, indent=2)

    @classmethod
    def load_config(cls, fname, ontology, **kwargs):
        with open(fname) as f:
            logging.info('loading config from {}'.format(fname))
            args = object()
            for k, v in json.load(f):
                setattr(args, k, kwargs.get(k, v))
        return cls(args, ontology)

    def save(self, summary, identifier):
        fname = '{}/{}.t7'.format(self.args.dout, identifier)
        logging.info('saving model to {}'.format(fname))
        state = {
            'args': vars(self.args),
            'model': self.state_dict(),
            'summary': summary,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, fname)

    def load(self, fname):
        logging.info('loading model from {}'.format(fname))
        state = torch.load(fname)
        self.load_state_dict(state['model'])
        self.set_optimizer()
        self.optimizer.load_state_dict(state['optimizer'])

    def get_saves(self, directory=None):
        if directory is None:
            directory = self.args.dout
        files = [f for f in os.listdir(directory) if f.endswith('.t7')]
        scores = []
        for fname in files:
            re_str = r'dev_{}=([0-9\.]+)'.format(self.args.stop)
            dev_acc = re.findall(re_str, fname)
            if dev_acc:
                score = float(dev_acc[0].strip('.'))
                scores.append((score, os.path.join(directory, fname)))
        if not scores:
            raise Exception('No files found!')
        scores.sort(key=lambda tup: tup[0], reverse=True)
        return scores

    def prune_saves(self, n_keep=5):
        scores_and_files = self.get_saves()
        if len(scores_and_files) > n_keep:
            for score, fname in scores_and_files[n_keep:]:
                os.remove(fname)

    def load_best_save(self, directory):
        if directory is None:
            directory = self.args.dout

        scores_and_files = self.get_saves(directory=directory)
        if scores_and_files:
            assert scores_and_files, 'no saves exist at {}'.format(directory)
            score, fname = scores_and_files[0]
            self.load(fname)


class RNNEncoder(nn.Module):
    def __init__(self, din, dhidden):
        super().__init__()
        self.rnn = nn.LSTM(din, dhidden, bidirectional=True, batch_first=True)

    def forward(self, x, x_len, dropout=0.2):
        utt = run_rnn(self.rnn, x, x_len)
        utt = F.dropout(utt, dropout, self.training)
        return utt


class DotAttention(nn.Module):
    def __init__(self, dquery, dref, dout):
        super().__init__()
        self.d = dout
        self.query_w = nn.Linear(dquery, dout, bias=False)
        self.ref_w = nn.Linear(dref, dout, bias=False)

    def forward(self, query, ref, dropout=0.2):
        dquery = F.dropout(query, dropout, self.training)
        dref = F.dropout(ref, dropout, self.training)
        wquery = self.query_w(dquery).relu()
        wref = self.ref_w(dref).relu()
        logits = wquery.mm(wref.transpose(1, 0)) / (self.d ** 0.5)  # dl*ul
        scores = F.softmax(logits, dim=1)
        return scores.mm(ref)


class Summer(nn.Module):
    def __init__(self, din, dh):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(din, dh), nn.Tanh(), nn.Linear(dh, 1))

    def forward(self, inputs, dropout=0.2):
        dinputs = F.dropout(inputs, dropout, self.training)
        logits = self.scorer(dinputs)
        s = F.softmax(logits, dim=0)
        return (s * inputs).sum(dim=0)


class PointerNetwork(nn.Module):
    def __init__(self, dh):
        super().__init__()
        self.rnn_cell = nn.LSTMCell(4 * dh, 2 * dh)
        self.pointer = Pointer(4 * dh, dh)

    def forward(self, init, match, dropout=0.2):
        dropout_mask = F.dropout(torch.ones_like(init), dropout, self.training)
        dmatch = F.dropout(match, dropout, self.training)
        start, p1 = self.pointer(init * dropout_mask, dmatch)
        dstart = F.dropout(start, dropout, self.training)
        dstart = dstart.unsqueeze(0)
        init = init.unsqueeze(0)
        _, state = self.rnn_cell(dstart, (init, init))
        state = state.squeeze(0)
        end, p2 = self.pointer(state * dropout_mask, dmatch)
        return p1, p2


class Pointer(nn.Module):
    def __init__(self, din, dh):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(din, dh), nn.Tanh(), nn.Linear(dh, 1))

    def forward(self, state, match):
        state = state.unsqueeze(0).repeat(match.shape[0], 1)
        match = torch.cat([state, match], dim=1)
        logits = self.scorer(match)
        s = F.softmax(logits, dim=0)
        return (s * match).sum(dim=0), s.squeeze(dim=1)
