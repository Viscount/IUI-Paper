import json
import numpy as np


def load_json(filename):
    return json.load(open(filename, 'r', encoding='utf-8'))

def save_json(filename, dict_obj):
    json.dump(dict_obj, open(filename, 'w', encoding='utf-8'))

def write_file(filename, lines):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()

def sigmoid(x):
    return 1 / (1 + np.exp(-1. * x))
