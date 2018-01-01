import multiprocessing
from abc import ABCMeta, abstractmethod

import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm

from tagging.config import *
from util.tool import *


class FeatureModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, train_list_file, test_list_file):
        self.train_list = list(read_file(train_list_file))
        self.test_list = list(read_file(test_list_file))
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
    def convert_doc2vec(self, doc, model):
        """ Convert a document to vector.
        :param doc: list of strings.
        :param model: word2vec model.
        :return: vector representation of the doc.
        """
        pass

    def convert_doc2vec_all(self, model_file):
        model = self.load_model(model_file)
        pbar = tqdm(total=len(self.train_list)+len(self.test_list))
        def _convert(file_list):
            for filename in file_list:
                doc = list(read_file(filename))
                doc_vec = self.convert_doc2vec(doc, model)
                output_file = filename + self.suffix
                doc_vec.dump(output_file)
                pbar.update(1)
        _convert(self.train_list)
        _convert(self.test_list)

    def make_dataset(self):
        labels = load_json(SEASON_LABELS)
        n_tags = len(load_json(TAG2ID))
        def _make_dataset(file_list, output_file):
            X, Y = [], []
            for filename in file_list:
                items = filename.split('/')
                s = items[-2][1:]
                if s not in labels.keys():
                    continue
                data_vec = np.load(filename + self.suffix)
                label_vec = np.zeros(n_tags)
                label_vec[labels[s]] = 1.
                X.append(data_vec.reshape((1, data_vec.shape[0])))
                Y.append(label_vec.transpose())
            X = np.concatenate(X, axis=0)
            Y = np.array(Y)
            data = {'data': X, 'label': Y}
            np.savez(output_file, **data)
        print('Creating dataset.....')
        _make_dataset(self.train_list, TRAIN_DATA)
        _make_dataset(self.test_list, TEST_DATA)


class TscDoc2vecModel(FeatureModel):
    def __init__(self, size):
        super(TscDoc2vecModel, self).__init__(TSC_TRAIN_LIST, TSC_TEST_LIST)
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
        tsc_docs = self.TscTaggedDocument(self.train_list)
        models = [
            # PV-DBOW
            Doc2Vec(dm=0, dbow_words=1, size=self.size, window=8, min_count=self.min_count, iter=10, workers=self.cores),
            # PV-DM w/average
            #Doc2Vec(dm=1, dm_mean=1, size=self.size, window=8, min_count=self.min_count, iter=10, workers=self.cores),
        ]
        print('Building vocabulary......')
        models[0].build_vocab(tsc_docs)
        print('Training doc2vec model......')
        #print(str(models[0]), models[0].corpus_count)
        models[0].train(tsc_docs, total_examples=models[0].corpus_count, epochs=models[0].iter)
        print('Vocabulary size:', len(models[0].wv.vocab))
        models[0].save(output)
        #models[1].reset_from(models[0])
        #print(str(models[1]))

    def load_model(self, model_file):
        return gensim.models.Doc2Vec.load(model_file)

    def convert_doc2vec(self, doc, model):
        words = [w for line in doc for w in line.split(' ')]
        if len(words) == 0:
            doc_vec = np.zeros(self.size)
        else:
            doc_vec = model.infer_vector(words)
        return doc_vec


class TscWord2vecModel(FeatureModel):
    def __init__(self, size):
        super(TscWord2vecModel, self).__init__(TSC_TRAIN_LIST, TSC_TEST_LIST)
        self.suffix = '_w2v'
        self.size = size  # vector dimension
        self.min_count = 10  # ignore words with frequency lower than this

    class TscCorpus(object):
        def __init__(self, doc_list):
            self.doc_list = doc_list
        def __iter__(self):
            for doc in self.doc_list:
                # 1) TSC as doc
                #for line in open(doc, 'r', encoding='utf-8'):
                #    yield line.strip().split(' ')
                # 2) Video as doc
                lines = read_file(doc)
                words = [w for line in lines for w in line.strip().split(' ')]
                yield words

    def train_model(self, output):
        sentences = self.TscCorpus(self.train_list)
        print('Training word2vec model......')
        model = gensim.models.Word2Vec(sentences, size=self.size, min_count=self.min_count, workers=self.cores)
        print('Vocabulary size:', len(model.wv.vocab))
        model.save(output)
        # Some results below:
        # voc_size = 244050 if min_count = 5
        # voc_size = 157416 if min_count = 10

    def load_model(self, model_file):
        return gensim.models.Word2Vec.load(model_file)

    def convert_doc2vec(self, doc, model):
        doc_vec = []
        for words in doc:
            word_vecs = [model.wv[w] for w in words if w in model.wv.vocab]
            doc_vec.extend(word_vecs)
        if len(doc_vec) == 0:
            doc_vec = np.zeros(self.size)
        else:
            doc_vec = np.mean(np.array(doc_vec), axis=0)
        return doc_vec


class IntroDoc2vecModel(TscDoc2vecModel):
    def __init__(self, size):
        super(IntroDoc2vecModel, self).__init__(size)
        self.suffix = '_seasond2v'
        train_list = list(read_file(SEASON_TRAIN_LIST))
        self.train_list = [t + '/intro_clean' for t in train_list]
        test_list = list(read_file(SEASON_TEST_LIST))
        self.test_list = [t + '/intro_clean' for t in test_list]


if __name__ == '__main__':
    d2v = TscDoc2vecModel(EMBED_SIZE)
    model_file = '../.cache/tag_d2v_s%d' % (EMBED_SIZE)
    #d2v.train_model(model_file)
    #d2v.convert_doc2vec_all(model_file)
    d2v.make_dataset()

    w2v = TscWord2vecModel(EMBED_SIZE)
    model_file = '../.cache/tag_w2v_s%d' % (EMBED_SIZE)
    #w2v.train_model(model_file)
    #w2v.convert_doc2vec_all(model_file)
    #w2v.make_dataset()

    season_d2v = IntroDoc2vecModel(EMBED_SIZE)
    model_file = '../.cache/intro_d2v_s%d' % (EMBED_SIZE)
    #season_d2v.train_model(model_file)
    #season_d2v.convert_doc2vec_all(model_file)
    #season_d2v.make_dataset()
