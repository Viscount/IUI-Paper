import json
import glob


ROOT = '/home/yikun/Documents/tsc_storyline_dataset'
SEASON_COINS = lambda root: root + '/coins.npy'
SEASON_LABELS = ROOT + '/season_labels'
ALL_SEASON_LIST = ROOT + '/all_list.txt'
TRAIN_SEASON_LIST = ROOT + '/train_list.txt'
TEST_SEASON_LIST = ROOT + '/test_list.txt'
MAX_SEQ_LEN = 30
EMBED_SIZE = 200
TRAIN_DATA = ROOT + '/train_data.npz'
TEST_DATA = ROOT + '/test_data.npz'


def list_season_dirs():
    season_list = []
    for s in glob.glob(ROOT + '/s*'):
        season_list.append(s)
    return season_list

def list_tsc_files(suffix):
    tsc_list = []
    for s in glob.glob(ROOT + '/s*'):
        for e in glob.glob(s + '/e*' + suffix):
            tsc_list.append(e)
    return tsc_list

