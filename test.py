import json


def f1():
    with open('data/raw/dev.json') as f:
        data = json.load(f)
        for dialogue in data:
            d = dialogue['dialogue']
            for turn in d:
                if turn['system_transcript'] == 'ok great and what will be your destination ?':
                    a = 1


with open('data/ann/ontology.json') as f:
    data = json.load(f)
    a = 1