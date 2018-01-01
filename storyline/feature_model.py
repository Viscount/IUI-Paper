import multiprocessing
from abc import ABCMeta, abstractmethod
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
import pandas as pd

from storyline.util import *


class FeatureModel(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.train_list = list(read_file(TRAIN_SEASON_LIST))
        self.test_list = list(read_file(TEST_SEASON_LIST))
        self.suffix = ''
        self.cores = multiprocessing.cpu_count()

    @abstractmethod
    def train_model(self, output):
        """ Train model and save to output file.
        :param output: file name of output
        """
        pass

    @abstractmethod
    def load_model(self, model_file):
        """ Load model from file.
        :param model_file: file name of model.
        :return: model object.
        """
        pass

    @abstractmethod
    def convert2vec(self, doc, model):
        """ Convert a document to vector.
        :param doc: list of strings.
        :param model: word2vec model.
        :return: vector representation of the doc.
        """
        pass

    def list_episode_files(self, season_dir):
        eps_coins = np.load(SEASON_COINS(season_dir))
        eps_list = []
        for eps in eps_coins[:,0]:
            eps_file = season_dir + '/e%d_clean' % (eps)
            eps_list.append(eps_file)
        return eps_list

    def convert2vec_all(self, model_file):
        model = self.load_model(model_file)
        pbar = tqdm(total=len(self.train_list)+len(self.test_list))
        def _convert(season_list):
            for season in season_list:
                eps_files = self.list_episode_files(season)
                for eps in eps_files:
                    doc = list(read_file(eps))
                    doc_vec = self.convert2vec(doc, model)
                    output_file = eps + self.suffix
                    doc_vec.dump(output_file)
                pbar.update(1)
        _convert(self.train_list)
        _convert(self.test_list)

    def make_dataset(self, dataset_type):
        pass

    def convert_coins(self, coins, alpha=1.1):
        coins = coins.astype(np.float32)
        coins = pd.Series(coins).interpolate().values
        threshold = alpha * np.sum(coins) / len(coins)
        labels = (coins > threshold).astype(np.float32)
        return labels


class Doc2vecModel(FeatureModel):
    def __init__(self, size):
        super(Doc2vecModel, self).__init__()
        self.suffix = '_d2v'
        self.size = size  # vector dimension
        self.min_count = 10  # ignore words with frequency lower than this

    class TscTaggedDocument(object):
        def __init__(self, doc_list):
            self.doc_list = doc_list
        def __iter__(self):
            for doc in self.doc_list:
                lines = read_file(doc)
                words = [w for line in lines for w in line.strip().split(' ')]
                yield TaggedDocument(words=words, tags=[doc])

    def train_model(self, output):
        docs = []
        for season in self.train_list:
            eps_list = self.list_episode_files(season)
            docs.extend(eps_list)
        tagged_docs = self.TscTaggedDocument(docs)
        models = [
            # PV-DBOW
            Doc2Vec(dm=0, dbow_words=1, size=self.size, window=8, min_count=self.min_count, iter=10, workers=self.cores),
            # PV-DM w/average
            #Doc2Vec(dm=1, dm_mean=1, size=self.size, window=8, min_count=self.min_count, iter=10, workers=self.cores),
        ]
        print('Building vocabulary......')
        models[0].build_vocab(tagged_docs)
        print('Training doc2vec model......')
        #print(str(models[0]), models[0].corpus_count)
        models[0].train(tagged_docs, total_examples=models[0].corpus_count, epochs=models[0].iter)
        print('Vocabulary size:', len(models[0].wv.vocab))
        models[0].save(output)
        #models[1].reset_from(models[0])
        #print(str(models[1]))

    def load_model(self, model_file):
        return gensim.models.Doc2Vec.load(model_file)

    def convert2vec(self, doc, model):
        words = [w for line in doc for w in line.split(' ')]
        if len(words) == 0:
            doc_vec = np.zeros(self.size)
        else:
            doc_vec = model.infer_vector(words)
        return doc_vec

    def _make_sequence_dataset(self, season_list, output_file):
        X, Y = [], []
        # 1) Get list of sequences
        for season in season_list:
            #print(season)
            eps_coins = np.load(SEASON_COINS(season))
            label_seq = self.convert_coins(eps_coins[:,1])
            #label_seq.reshape((1, len(label_seq)))
            Y.append(label_seq)
            #print(label_seq.shape)
            data_seq = []
            for eps in eps_coins[:,0]:
                eps_file = season + '/e%d_clean%s' % (eps, self.suffix)
                data_seq.append(np.load(eps_file))
            data_seq = np.array(data_seq)
            X.append(data_seq)
        # 2) Pad zero-vectors to align sequence length
        seqlen = []
        for i in range(len(X)):
            seqlen.append(len(Y[i]))
            zero_pad = np.zeros((MAX_SEQ_LEN-X[i].shape[0], X[i].shape[1]))
            X[i] = np.concatenate([X[i], zero_pad], axis=0)
            X[i] = X[i].reshape((1, X[i].shape[0], X[i].shape[1]))
            zero_pad = np.zeros((MAX_SEQ_LEN-Y[i].shape[0]))
            Y[i] = np.concatenate([Y[i], zero_pad], axis=0)
            Y[i] = Y[i].reshape((1, Y[i].shape[0]))
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        seqlen = np.array(seqlen, dtype=np.int32)
        data = {'data': X, 'label': Y, 'seqlen': seqlen}
        np.savez(output_file, **data)
        print(X.shape, Y.shape, seqlen.shape)

    def _make_separate_dataset(self, season_list, output_file):
        X, Y = [], []
        for season in season_list:
            #print(season)
            eps_coins = np.load(SEASON_COINS(season))
            eps_seq = eps_coins[:,0]
            label_seq = self.convert_coins(eps_coins[:,1])
            for i,eps in enumerate(eps_seq):
                eps_file = season + '/e%d_clean%s' % (eps, self.suffix)
                X.append(np.load(eps_file).reshape([1, EMBED_SIZE]))
                Y.append(label_seq[i])
        X = np.concatenate(X, axis=0)
        Y = np.array(Y)
        print('X:', X.shape, 'Y:', Y.shape)
        data = {'data': X, 'label': Y}
        np.savez(output_file, **data)

    def make_dataset(self, data_type):
        """
        :param data_type: 'seq' or 'sep'
        :return:
        """
        print('Creating dataset.....')
        if data_type == 'seq':
            self._make_sequence_dataset(self.train_list, TRAIN_DATA)
            self._make_sequence_dataset(self.test_list, TEST_DATA)
        elif data_type == 'sep':
            self._make_separate_dataset(self.train_list, TRAIN_DATA)
            self._make_separate_dataset(self.test_list, TEST_DATA)

if __name__ == '__main__':
    d2v = Doc2vecModel(EMBED_SIZE)
    model_file = '../.cache/sl_d2v_s%d' % (EMBED_SIZE)
    #d2v.train_model(model_file)
    #d2v.convert2vec_all(model_file)
    d2v.make_dataset('seq')
