import os
import re
import shutil
import string
import MySQLdb
from tqdm import tqdm
import numpy as np
from zhon.hanzi import punctuation

from storyline.util import *

class StorylineDataExtractor(object):
    def __init__(self, datadir):
        self.datadir = datadir

    def connect_to_db(self):
        db = MySQLdb.connect(user='root', passwd='1234', db='dynamicdb2', charset='utf8')
        return db.cursor()

    def fetch_season_coin(self, season_id, cursor, season_dir):
        # Fetch episodes by season
        query_e = 'select `index`,episode_id,coins from episode where season_id=%s;' % (season_id)
        cursor.execute(query_e)
        records = [(int(e[0]), e[1], e[2]) for e in cursor.fetchall() if e[0].isdigit()]
        if len(records) == 0:
            return None
        records = sorted(records, key=lambda x: x[0])
        eps_coins = np.array([r[1:] for r in records])
        coins = eps_coins[:,1]
        coins[coins==None] = 0
        eps_coins[:,1] = coins
        eps_coins = eps_coins.astype(np.int32)
        np.save(SEASON_COINS(season_dir), eps_coins)
        return eps_coins

    def fetch_season_tsc(self, epid_list, cursor, season_dir):
        for epid in epid_list:
            tsc_file = '%s/e%d_raw' % (season_dir, int(epid))
            query_c = 'select content from danmaku where episode_id=%d order by playback_time;' % (epid)
            cursor.execute(query_c)
            tsc_list = [tsc[0] for tsc in cursor.fetchall()]
            write_file(tsc_file, tsc_list)

    def fetch_season_all(self):
        query_s = 'select season_id from bangumi;'
        c = self.connect_to_db()
        c.execute(query_s)
        season_ids = [i[0] for i in c.fetchall()]
        pbar = tqdm(total=len(season_ids))
        for sid in season_ids:
            pbar.update(1)
            season_dir = '%s/s%s' % (self.datadir, sid)
            if os.path.exists(season_dir):
                shutil.rmtree(season_dir)
            os.mkdir(season_dir)
            eps_coins = self.fetch_season_coin(sid, c, season_dir)
            if eps_coins is not None:
                self.fetch_season_tsc(eps_coins[:,0], c, season_dir)
        c.close()


class TscPreprocessor(object):
    def __init__(self):
        special_characters = u"└O┘°Д°ヽ•̀ω•́ゝ↑￣▽￣♀ω•́•ｸﾞｯ๑•̀ㅂ•́و✧̀﻿˙ー˙→↓\s"
        self.pattern = u"[%s%s%s]" % (string.punctuation, punctuation, special_characters)

    def preprocess_tsc(self, raw_tsc, cut_word=True):
        # Remove punctuation and special characters
        tsc = re.sub(self.pattern, "", raw_tsc.strip())
        # Replace 23333+ and 6666+
        tsc = re.sub(r'(.+?)\1+', r'\1\1', tsc)
        if not tsc:
            return None
        if not cut_word:
            return tsc
        words = jieba.cut(tsc)
        return words

    def preprocess_tscfile(self, input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as infile, \
                open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                tsc = self.preprocess_tsc(line, True)
                if not tsc:
                    continue
                outfile.write(' '.join(tsc) + '\n')

    def preprocess_all(self):
        tsc_list = list_tsc_files('_raw')
        pbar = tqdm(total=len(tsc_list))
        for tsc_raw in tsc_list:
            tsc_clean = tsc_raw[:-4] + '_clean'
            self.preprocess_tscfile(tsc_raw, tsc_clean)
            pbar.update(1)

    def filter_all_seasons_list(self):
        # Filter by number of episodes, coins, etc
        # Generate ALL_SEASON_LIST
        all_list = []
        for season in list_season_dirs():
            if not os.path.exists(SEASON_COINS(season)):
                continue
            eps_coins = np.load(SEASON_COINS(season))
            coins = eps_coins[:,1]
            if len(coins) > MAX_SEQ_LEN or len(coins) < 3:
                continue
            nonzero_count = np.count_nonzero(coins)
            nonzero_ratio = np.float(nonzero_count) / len(coins)
            if nonzero_ratio < 0.7:
                continue
            all_list.append(season)
        print(len(all_list), 'videos are left!')
        write_file(ALL_SEASON_LIST, all_list)

    def reset_train_test_list(self, ratio=0.8):
        seasons = list(read_file(ALL_SEASON_LIST))
        perm = np.random.permutation(len(seasons))
        n_train = int(len(seasons) * ratio)
        with open(TRAIN_SEASON_LIST, 'w', encoding='utf-8') as f:
            for i in range(n_train):
                f.write(seasons[perm[i]] + '\n')
        with open(TEST_SEASON_LIST, 'w', encoding='utf-8') as f:
            for i in range(n_train, len(seasons)):
                f.write(seasons[perm[i]] + '\n')


if __name__ == '__main__':
    de = StorylineDataExtractor(ROOT)
    #de.fetch_season_all()

    tp = TscPreprocessor()
    #tp.preprocess_all()
    tp.filter_all_seasons_list()
    tp.reset_train_test_list()
