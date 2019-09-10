import json
from collections import defaultdict
import numpy as np


def st1():
    for dataset in ['train', 'dev', 'test']:
        to, nd, nv, nm, dc, y, n, sv, ot = 0, 0, 0, 0, 0, 0, 0, 0, 0
        with open('data/ann/{}.json'.format(dataset)) as f:
            for d in json.load(f)['dialogues']:
                for t in d['turns']:
                    for s, v in t['turn_label']:
                        to += 1
                        if s.split('-')[0] != t['domain']:
                            nd += 1
                        elif v == '':
                            nv += 1
                        elif 'not mentioned' in v:
                            nm += 1
                        elif v == 'dontcare':
                            dc += 1
                        elif v == 'yes':
                            y += 1
                        elif v == 'no':
                            n += 1
                        elif ' '.join(t['transcript']).find(v) != -1:
                            sv += 1
                        else:
                            ot += 1
        print(nd/to, nv/to, nm/to, dc/to, y/to, n/to, sv/to, ot/to)


def st2():
    for dataset in ['train']:
        print(dataset, '===========================================')
        value_lens = defaultdict(list)
        value_len_config = {}
        with open('data/ann/{}.json'.format(dataset)) as f:
            for d in json.load(f)['dialogues']:
                for t in d['turns']:
                    for s, v in t['turn_label']:
                        value_lens[s].append(len(v.split()))
        for v, lens in value_lens.items():
            print(v, 'quantile{}: {}'.format(0.99, np.quantile(np.array(lens), 0.9)))
            value_len_config[v] = int(np.ceil(np.quantile(np.array(lens), 0.9)))
        with open('value_len_config.json', 'w', encoding='utf8') as f:
            json.dump(value_len_config, f, indent=4)


if __name__ == '__main__':
    with open('value_len_config.json') as f:
        value_len_config = json.load(f)