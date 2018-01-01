import os
import re
import shutil
import string
import MySQLdb
import jieba
from tqdm import tqdm
from zhon.hanzi import punctuation

from tagging.config import *
from util.word_segment import WordSegment
from util.tool import *


class DataExtractor(object):
    def __init__(self, datadir):
        self.datadir = datadir

    def connect_to_db(self):
        db = MySQLdb.connect(user='root', passwd='1234', db='dynamicdb2', charset='utf8')
        return db.cursor()

    def fetch_season_tag(self):
        c = self.connect_to_db()
        query_s = 'select season_id,tags from bangumi;'
        c.execute(query_s)
        season_tag = c.fetchall()
        c.close()
        # Process tags
        season_tag = dict(map(lambda x: (x[0], x[1].split('|')), season_tag))
        tags = set([t for st in season_tag.values() for t in st])
        distinct_tags = sorted([t for t in tags if t])  # remove empty tag
        print('%d distinct tags in total!' % (len(distinct_tags)))
        ids = range(len(distinct_tags))
        tag2id = dict(zip(list(distinct_tags), ids))
        save_json(TAG2ID, tag2id)
        id2tag = dict(zip(ids, list(distinct_tags)))
        save_json(ID2TAG, id2tag)
        # Save labels for seasons
        f_tag2id = lambda tags: sorted([tag2id[t] for t in tags if t])
        labels = {s: f_tag2id(t) for s, t in season_tag.items()}
        save_json(SEASON_LABELS, labels)
        # Create directory for each season
        for season in season_tag.keys():
            season_dir = '%s/s%s' % (self.datadir, season)
            if os.path.exists(season_dir):
                shutil.rmtree(season_dir)
            os.mkdir(season_dir)

    def fetch_season_intro(self):
        c = self.connect_to_db()
        query_s = 'select season_id,introduction from bangumi;'
        c.execute(query_s)
        season_intros = c.fetchall()
        c.close()
        # Save intro
        for s_i in season_intros:
            season_dir = '%s/s%s' % (self.datadir, s_i[0])
            with open(season_dir + '/intro_raw', 'w', encoding='utf-8') as f:
                f.write(s_i[1])

    def fetch_tsc(self, season, cursor):
        season_dir = '%s/s%s' % (self.datadir, season)
        # Fetch episodes by season
        query_e = 'select episode_id from episode where season_id=%s;' % (season)
        cursor.execute(query_e)
        episodes = sorted([e[0] for e in cursor.fetchall()])
        for e in episodes:
            # Fetch tsc by episode
            tsc_file = '%s/e%s_raw' % (season_dir, e)
            query_c = 'select content from danmaku where episode_id=%d order by playback_time;' % (e)
            cursor.execute(query_c)
            tsc_list = [tsc[0] for tsc in cursor.fetchall()]
            write_file(tsc_file, tsc_list)

    def fetch_tsc_all(self):
        c = self.connect_to_db()
        seasons = list(load_json(SEASON_LABELS).keys())
        pbar = tqdm(total=len(seasons))
        for s in seasons:
            self.fetch_tsc(s, c)
            pbar.update(1)
        c.close()


class Preprocessor(object):
    def __init__(self):
        #special_characters = u"└O┘°Д°ヽ•̀ω•́ゝ↑￣▽￣♀ω•́•ｸﾞｯ๑•̀ㅂ•́و✧̀﻿˙ー˙→↓\s"
        #self.pattern = u"[%s%s%s]" % (string.punctuation, punctuation, special_characters)
        self.segment = WordSegment().word_segment

    #def preprocess_tsc(self, raw_tsc, cut_word=True):
    #    # Remove punctuation and special characters
    #    tsc = re.sub(self.pattern, "", raw_tsc.strip())
    #    # Replace 23333+ and 6666+
    #    tsc = re.sub(r'(.+?)\1+', r'\1\1', tsc)
    #    if not tsc:
    #        return None
    #    if not cut_word:
    #        return tsc
    #    words = jieba.cut(tsc)
    #    return words

    def preprocess_intro(self):
        seasons = glob.glob(ROOT + '/s*')
        pbar = tqdm(total=len(seasons))
        for s in seasons:
            pbar.update(1)
            if not os.path.isdir(s):
                continue
            lines = read_file(s + '/intro_raw')
            doc = []
            for line in lines:
                words = self.segment(line)
                if words:
                    doc.extend(words)
            with open(s + '/intro_clean', 'w', encoding='utf-8') as f:
                f.write(' '.join(doc))


    def preprocess_tsc(self):
        tsc_list = self.list_tsc_files('_raw')
        pbar = tqdm(total=len(tsc_list))
        for tsc_raw in tsc_list:
            pbar.update(1)
            tsc_clean = tsc_raw[:-4] + '_clean'
            with open(tsc_raw, 'r', encoding='utf-8') as infile, \
                    open(tsc_clean, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    tsc = self.segment(line.strip())
                    if not tsc:
                        continue
                    outfile.write(' '.join(tsc) + '\n')

    def list_tsc_files(self, suffix):
        tsc_list = []
        for s in glob.glob(ROOT + '/s*'):
            for e in glob.glob(s + '/e*' + suffix):
                tsc_list.append(e)
        return tsc_list

    def reset_train_test_list(self, ratio=0.8):
        # Set train/test list for seasons
        labels = load_json(SEASON_LABELS)
        seasons = []
        for filename in glob.glob(ROOT + '/s*'):
            items = filename.split('/')
            s = items[-1][1:]
            if s in labels.keys():
                seasons.append(filename)
        perm = np.random.permutation(len(seasons))
        n_train = int(len(seasons) * ratio)
        season_train = [seasons[perm[i]] for i in range(n_train)]
        season_test = [seasons[perm[i]] for i in range(n_train, len(seasons))]
        write_file(SEASON_TRAIN_LIST, season_train)
        write_file(SEASON_TEST_LIST, season_test)

        # Set train/test list for episodes
        def set_episodes(seasons, output_file):
            ep_list = []
            for s in seasons:
                eps = glob.glob(s + '/e*_clean')
                ep_list += eps
            write_file(output_file, ep_list)
        set_episodes(season_train, TSC_TRAIN_LIST)
        set_episodes(season_test, TSC_TEST_LIST)


if __name__ == '__main__':
    de = DataExtractor(ROOT)
    #de.fetch_season_tag()
    #de.fetch_season_intro()
    #de.fetch_tsc_all()

    pre = Preprocessor()
    #pre.preprocess_intro()
    #pre.preprocess_tsc()
    pre.reset_train_test_list()
