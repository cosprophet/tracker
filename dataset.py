import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm

dev_test_domains = {'restaurant', 'hotel', 'attraction', 'taxi', 'train'}


def annotate(sent):
    return sent.split()


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


class Turn:

    def __init__(self, turn_id, transcript, turn_label, turn_slots, turn_value_type, turn_value_start, turn_value_end,
                 belief_state, system_acts, system_transcript, last_domain, domain, num=None):
        self.id = turn_id
        self.transcript = transcript
        self.turn_label = turn_label
        self.turn_slots = turn_slots
        self.turn_value_type = turn_value_type
        self.turn_value_start = turn_value_start
        self.turn_value_end = turn_value_end
        self.belief_state = belief_state
        self.system_acts = system_acts
        self.system_transcript = system_transcript
        self.last_domain = last_domain
        self.domain = domain
        self.num = num or {}

    def to_dict(self):
        return {'turn_id': self.id, 'transcript': self.transcript, 'turn_label': self.turn_label,
                'turn_slots': self.turn_slots, 'turn_value_type': self.turn_value_type,
                'turn_value_start': self.turn_value_start, 'turn_value_end': self.turn_value_end,
                'belief_state': self.belief_state, 'system_acts': self.system_acts,
                'system_transcript': self.system_transcript, 'last_domain': self.last_domain,
                'domain': self.domain, 'num': self.num}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def annotate_raw(cls, raw):
        system_acts = []
        for a in raw['system_acts']:
            if isinstance(a, list):
                s, v = a
                system_acts.append(['inform'] + s.split() + ['='] + v.split())
            else:
                system_acts.append(['request'] + a.split())

        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number', 'y': 'yes', 'n': 'no'}
        turn_label = [(fix.get(s.strip(), s.strip()), fix.get(v.strip(), v.strip())) for s, v in raw['turn_label']]
        transcript_tokens = annotate(raw['transcript'])
        transcript_spans = convert_idx(raw['transcript'], transcript_tokens)

        turn_slots = [s.split('-')[1] for s, v in turn_label if s.split('-')[0] == raw['domain'] and v != '']
        turn_value_type, turn_value_start, turn_value_end = [], [], []
        for s, v in turn_label:
            if s.split('-')[0] != raw['domain'] or v == '':
                continue
            t = [0, 0]
            value_start = raw['transcript'].find(v)
            if value_start == -1:
                t[0] = 1
                turn_value_type.append(t)
                continue
            value_end = value_start + len(v)
            value_span = []
            for idx, span in enumerate(transcript_spans):
                if not (value_end <= span[0] or value_start >= span[1]):
                    value_span.append(idx)
            start, end = value_span[0], value_span[-1]
            t[1] = 1
            turn_value_type.append(t), turn_value_start.append(start), turn_value_end.append(end)
        assert len(turn_slots) == len(turn_value_type)

        return cls(
            turn_id=raw['turn_idx'],
            transcript=transcript_tokens,
            system_acts=system_acts,
            turn_label=turn_label,
            turn_slots=turn_slots,
            turn_value_type=turn_value_type,
            turn_value_start=turn_value_start,
            turn_value_end=turn_value_end,
            belief_state=raw['belief_state'],
            system_transcript=raw['system_transcript'],
            last_domain=raw['last_domain'],
            domain=raw['domain']
        )

    def numericalize_(self, vocab):
        self.num['last_domain'] = vocab.word2index(self.last_domain, train=True)
        self.num['domain'] = vocab.word2index(self.domain, train=True)
        self.num['transcript'] = vocab.word2index([w.lower() for w in self.transcript], train=True)


class Dialogue:

    def __init__(self, dialogue_id, turns):
        self.id = dialogue_id
        self.turns = turns

    def __len__(self):
        return len(self.turns)

    def to_dict(self):
        return {'dialogue_id': self.id, 'turns': [t.to_dict() for t in self.turns]}

    @classmethod
    def from_dict(cls, d):
        return cls(d['dialogue_id'], [Turn.from_dict(t) for t in d['turns']])

    @classmethod
    def annotate_raw(cls, raw):
        return cls(raw['dialogue_idx'], [Turn.annotate_raw(t) for t in raw['dialogue']])


class Dataset:

    def __init__(self, dialogues):
        self.dialogues = dialogues

    def __len__(self):
        return len(self.dialogues)

    def iter_turns(self):
        for d in self.dialogues:
            for t in d.turns:
                if t.domain not in dev_test_domains:
                    continue
                yield t

    def to_dict(self):
        return {'dialogues': [d.to_dict() for d in self.dialogues]}

    @classmethod
    def from_dict(cls, d):
        return cls([Dialogue.from_dict(dd) for dd in d['dialogues']])

    @classmethod
    def annotate_raw(cls, fname):
        with open(fname) as f:
            data = json.load(f)
            return cls([Dialogue.annotate_raw(d) for d in tqdm(data)])

    def numericalize_(self, vocab):
        for t in self.iter_turns():
            t.numericalize_(vocab)

    def extract_ontology(self):
        domains = set()
        domain_slots = defaultdict(set)
        for t in self.iter_turns():
            domains.add(t.domain)
            for s, v in t.turn_label:
                domain = s.split('-')[0].lower()
                if domain not in dev_test_domains:
                    continue
                domain_slots[domain].add(s.split('-')[1].lower())
        return Ontology(sorted(list(domains)), {k: sorted(list(v)) for k, v in domain_slots.items()})

    def batch(self, batch_size, shuffle=False):
        turns = list(self.iter_turns())
        if shuffle:
            np.random.shuffle(turns)
        for i in tqdm(range(0, len(turns), batch_size)):
            yield turns[i:i+batch_size]

    def evaluate_preds(self, preds):
        domain = []
        slot = []
        inform = []
        joint_slot = []
        joint_goal = []
        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number', 'y': 'yes', 'n': 'no'}
        i = 0
        for d in self.dialogues:
            pred_state = {}
            for t in d.turns:
                if t.domain not in dev_test_domains:
                    continue
                gold_domain = t.domain
                pred_domain = preds[i]['domain']
                gold_inform = set([(s, v) for s, v in t.turn_label])
                gold_slot = set([s for s, v in t.turn_label])
                pred_inform = set([(s, v) for s, v in preds[i]['slots']])
                pred_slot = set([s for s, v in preds[i]['slots']])
                domain.append(gold_domain == pred_domain)
                slot.append(gold_slot == pred_slot)
                inform.append(gold_inform == pred_inform)

                gold_recovered = set()
                pred_recovered = set()
                gold_slot_recovered = set()
                pred_slot_recovered = set()
                for s, v in pred_inform:
                    pred_state[s] = v
                for b in t.belief_state:
                    for s, v in b['slots']:
                        gold_slot_recovered.add(fix.get(s.strip(), s.strip()))
                        gold_recovered.add((fix.get(s.strip(), s.strip()), fix.get(v.strip(), v.strip())))
                for s, v in pred_state.items():
                    pred_slot_recovered.add(s)
                    pred_recovered.add((s, v))
                joint_slot.append(gold_slot_recovered == pred_slot_recovered)
                joint_goal.append(gold_recovered == pred_recovered)
                i += 1
        return {'domain': np.mean(domain), 'slot': np.mean(slot), 'turn_inform': np.mean(inform),
                'joint_slot': np.mean(joint_slot), 'joint_goal': np.mean(joint_goal)}

    def record_preds(self, preds, to_file):
        data = self.to_dict()
        i = 0
        for d in data['dialogues']:
            for t in d['turns']:
                if t['domain'] not in dev_test_domains:
                    continue
                t['pred'] = sorted(list(preds[i]))
                i += 1
        with open(to_file, 'wt') as f:
            json.dump(data, f, indent=4)


class Ontology:

    def __init__(self, domains=None, domain_slots=None, num=None):
        self.domains = domains or []
        self.domain_slots = domain_slots or {}
        self.num = num or {}

    def __add__(self, another):
        new_domains = sorted(list(set(self.domains + another.domains)))
        new_domain_slots = {d: sorted(list(set(self.domain_slots.get(d, []) + another.domain_slots.get(d, [])))) for
                            d in new_domains}
        return Ontology(new_domains, new_domain_slots)

    def __radd__(self, another):
        return self if another == 0 else self.__add__(another)

    def to_dict(self):
        return {'domains': self.domains, 'domain_slots': self.domain_slots, 'num': self.num}

    def numericalize_(self, vocab):
        self.num['domains'] = [vocab.word2index(d, train=True) for d in self.domains]
        for d, ss in self.domain_slots.items():
            self.num[vocab.word2index(d, train=True)] = [[vocab.word2index(w, train=True) for w in s.split()]
                                                         for s in ss]

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
