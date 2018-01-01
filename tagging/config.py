import numpy as np
import glob


ROOT = '/home/yikun/Documents/tsc_tagging_dataset'

TAG2ID = ROOT + '/tag2id'

ID2TAG = ROOT + '/id2tag'

SEASON_LABELS = ROOT + '/season_labels'

SEASON_TRAIN_LIST = ROOT + '/season_train_list.txt'

SEASON_TEST_LIST = ROOT + '/season_test_list.txt'

TSC_TRAIN_LIST = ROOT + '/tsc_train_list.txt'

TSC_TEST_LIST = ROOT + '/tsc_test_list.txt'

EMBED_SIZE = 200

TRAIN_DATA = ROOT + '/train_data.npz'

TEST_DATA = ROOT + '/test_data.npz'


#def tag_freq():
#    tagROOT_DIR = os.path.dirname(os.path.abspath(__file__))2id = load_json(TAG2ID)
#    labels = load_json(SEASON_LABELS)
#    counts = np.zeros(len(tag2id), dtype=int)
#    for s in labels:
#        for t in labels[s]:
#            counts[t] += 1
#    counts = np.sort(counts)[::-1]
#    print(counts)



