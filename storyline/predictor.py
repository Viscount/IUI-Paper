import os
import shutil
import random
import sklearn as sk
import sklearn.metrics
import numpy as np
import tensorflow as tf

from storyline.util import *


class BaseModel(object):
    def __init__(self, name):
        self.name = name

    def _fc_layer(self, x, size):
        return tf.contrib.layers.fully_connected(x, size, activation_fn=None, scope='fc')

    def _relu_layer(self, x):
        return tf.nn.relu(x)

    def _dropout_layer(self, x, keep_prob, is_training):
        return tf.layers.dropout(x, keep_prob, training=is_training)

    def _convert2onehot(self, Y, n_class=2):
        tmp = np.zeros(list(Y.shape) + [n_class])
        tmp[list(np.indices(tmp.shape[:-1])) + [Y.astype(np.int)]] = 1
        return tmp

    def save_weights(self, sess, filename):
        weights = {}
        for v in tf.trainable_variables():
            weights[v.name] = sess.run(v)
        np.savez(filename, **weights)


class LstmModel(BaseModel):

    class DataWrapper(object):
        def __init__(self, data_file, is_validate):
            # Load data
            data = np.load(data_file)
            self.data = data['data']
            self.label = data['label']
            self.seqlen = data['seqlen']
            # Create validation set if is_validate=True
            if is_validate:
                idx = int(0.9 * self.data.shape[0])
                self.val_data = self.data[idx:, :]
                self.data = self.data[:idx, :]
                self.val_label = self.label[idx:, :]
                self.label = self.label[:idx, :]
                self.val_seqlen = self.seqlen[idx:]
                self.seqlen = self.seqlen[:idx]
            print(self.data.shape, self.label.shape, self.seqlen.shape)

        def next_batch(self, batch_size):
            if not batch_size or batch_size >= self.data.shape[0]:
                batch_size = self.data.shape[0]
            batch = random.sample(range(self.data.shape[0]), batch_size)
            return self.data[batch, :], self.label[batch, :], self.seqlen[batch]

    def __init__(self, name):
        super(LstmModel, self).__init__(name)
        self.h1_size = 256
        self.h2_size = 128
        self.h3_size = 128
        self.rnn_size = 128
        self.num_classes = 2

    def _seq2mask(self, seqlen, multi_output=False):
        """
        :param seqlen:
        :return:
        """
        mask = np.zeros((seqlen.shape[0], MAX_SEQ_LEN, self.num_classes))
        for i in range(seqlen.shape[0]):
            if multi_output:
                mask[i,:seqlen[i],:] = 1.0
            else:
                mask[i,seqlen[i]-1,:] = 1.0
        return mask

    def create_net(self, X, seqlen, is_training):
        """ Create network structure.
        :param X: [batch_size x max_seq_len x embed_size]
        :param seqlen: list of actual sequence length, [batch_size x 1]
        :param is_training: is training phase or testing phase?
        :return: output tensor
        """
        layers = {}
        with tf.variable_scope('h1'):
            layers['h1/fc'] = self._fc_layer(X, self.h1_size)
            layers['h1/relu'] = self._relu_layer(layers['h1/fc'])
            layers['h1/drop'] = self._dropout_layer(layers['h1/relu'], 0.3, is_training)

        with tf.variable_scope('h2'):
            layers['h2/fc'] = self._fc_layer(layers['h1/drop'], self.h2_size)
            layers['h2/relu'] = self._relu_layer(layers['h2/fc'])
            layers['h2/drop'] = self._dropout_layer(layers['h2/relu'], 0.3, is_training)

        #with tf.variable_scope('h3'):
        #    layers['h3/fc'] = self._fc_layer(layers['h2/drop'], self.h3_size)
        #    layers['h3/relu'] = self._relu_layer(layers['h3/fc'])
        #    layers['h3/drop'] = self._dropout_layer(layers['h3/relu'], 0.3, is_training)

        with tf.variable_scope('rnn1'):
            cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size, activation=tf.tanh, forget_bias=10)
            #cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2, state_is_tuple=True)
            layers['rnn1/output'], layers['rnn1/state'] = tf.nn.dynamic_rnn(
                cell, layers['h2/drop'], sequence_length=seqlen, dtype=tf.float32)
            layers['rnn1/drop'] = self._dropout_layer(layers['rnn1/output'], 0.3, is_training)

        with tf.variable_scope('out'):
            layers['out/fc'] = self._fc_layer(layers['rnn1/drop'], self.num_classes)

        self.layers = layers
        return layers['out/fc']

    def train(self, batch_size, learning_rate, num_steps, weights_file, report_file):
        # (1) Get dataset
        dataset = self.DataWrapper(TRAIN_DATA, is_validate=True)
        val_x, val_y, val_seqlen = dataset.val_data, dataset.val_label, dataset.val_seqlen
        val_y = self._convert2onehot(val_y)
        val_mask = self._seq2mask(val_seqlen, True)

        # (2) Create network
        X = tf.placeholder(tf.float32, [None, MAX_SEQ_LEN, EMBED_SIZE], name='X')
        Y = tf.placeholder(tf.float32, [None, MAX_SEQ_LEN, self.num_classes], name='Y')
        seqlen = tf.placeholder(tf.int32, [None], name='seqlen')
        mask = tf.placeholder(tf.float32, [None, MAX_SEQ_LEN, self.num_classes], name='mask')
        lr = tf.placeholder(tf.float32, [], name='lr')
        is_training = tf.placeholder(tf.bool, name='is_training')
        with tf.variable_scope(self.name):
            out = self.create_net(X, seqlen, is_training)
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=out)
            #cross_ent = tf.divide(tf.reduce_sum(cross_ent, [1, 2]), tf.cast(seqlen, tf.float32))
            cross_ent_sum = tf.reduce_sum(tf.multiply(cross_ent, mask), [1, 2])
            cross_ent_mean = tf.reduce_mean(cross_ent_sum)
            #var_weights = [v for v in tf.trainable_variables() if v.name.endswith('weights:0')]
            #l2_regularizer = tf.contrib.layers.l2_regularizer(0.001)
            #regularizer = tf.contrib.layers.apply_regularization(l2_regularizer, var_weights)
            train_loss = cross_ent_mean
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(train_loss)

        # (3) Start training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(1, num_steps + 1):
                batch_x, batch_y, batch_seqlen = dataset.next_batch(batch_size)
                batch_y = self._convert2onehot(batch_y)
                # Subsampling seqlen
                #batch_seqlen = np.array([np.random.randint(1, i+1) for i in batch_seqlen])
                batch_mask = self._seq2mask(batch_seqlen, True)
                sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, seqlen: batch_seqlen, mask: batch_mask,
                                               lr: learning_rate, is_training: True})
                if step % 5000 == 0:
                    learning_rate *= 0.9

                # Validation
                if step % 200 == 0 or step == 1:
                    y_pred, loss, ce = sess.run([out, train_loss, cross_ent], feed_dict={
                            X: val_x, Y: val_y, seqlen:val_seqlen, mask: val_mask, is_training: False})
                    res = self.report(np.argmax(val_y, 2), np.argmax(y_pred, 2), val_seqlen)
                    line = 'Step: ' + str(step)
                    line += '| Val_Loss: {:.4f}'.format(loss)
                    line += '| Val_Accuracy: {:.4f}'.format(res['accuracy'])
                    line += '| Val_Precision: {:.4f}'.format(res['precision'])
                    line += '| Val_Recall: {:.4f}'.format(res['recall'])
                    print(line)
                    with open(report_file, 'a') as f:
                        f.write(line + '\n')
            self.save_weights(sess, weights_file)

    def report(self, y_real, y_pred, seqlen):
        y_real_fix, y_pred_fix = [], []
        for i in range(y_real.shape[0]):
            y_real_fix.append(y_real[i,:seqlen[i]])
            y_pred_fix.append(y_pred[i,:seqlen[i]])
        y_real_fix = np.concatenate(y_real_fix)
        y_pred_fix = np.concatenate(y_pred_fix)
        result = {}
        result['accuracy'] = sk.metrics.accuracy_score(y_real_fix, y_pred_fix)
        result['precision'] = sk.metrics.precision_score(y_real_fix, y_pred_fix)
        result['recall'] = sk.metrics.recall_score(y_real_fix, y_pred_fix)
        return result


class ClassifyModel(BaseModel):

    class DataWrapper(object):
        def __init__(self, data_file, is_validate):
            # Load data
            data = np.load(data_file)
            self.data = data['data']
            self.label = data['label']
            # Create validation set if is_validate=True
            if is_validate:
                idx = int(0.9 * self.data.shape[0])
                self.val_data = self.data[idx:, :]
                self.data = self.data[:idx, :]
                self.val_label = self.label[idx:]
                self.label = self.label[:idx]
            print(self.data.shape, self.label.shape)

        def next_batch(self, batch_size):
            if not batch_size or batch_size >= self.data.shape[0]:
                batch_size = self.data.shape[0]
            batch = random.sample(range(self.data.shape[0]), batch_size)
            return self.data[batch, :], self.label[batch]

    def __init__(self, name):
        super(ClassifyModel, self).__init__(name)
        self.h1_size = 256
        self.h2_size = 128
        self.h3_size = 128
        self.num_classes = 2

    def create_net(self, X, is_training):
        layers = {}

        with tf.variable_scope('h1'):
            layers['h1/fc'] = self._fc_layer(X, self.h1_size)
            layers['h1/relu'] = self._relu_layer(layers['h1/fc'])
            layers['h1/drop'] = self._dropout_layer(layers['h1/relu'], 0.3, is_training)

        with tf.variable_scope('h2'):
            layers['h2/fc'] = self._fc_layer(layers['h1/drop'], self.h2_size)
            layers['h2/relu'] = self._relu_layer(layers['h2/fc'])
            layers['h2/drop'] = self._dropout_layer(layers['h2/relu'], 0.3, is_training)

        with tf.variable_scope('h3'):
            layers['h3/fc'] = self._fc_layer(layers['h2/drop'], self.h3_size)
            #layers['h3/res'] = tf.add(layers['h1/drop'], layers['h3/fc'])
            layers['h3/relu'] = self._relu_layer(layers['h3/fc'])
            layers['h3/drop'] = self._dropout_layer(layers['h3/relu'], 0.3, is_training)

        with tf.variable_scope('out'):
            layers['out/fc'] = self._fc_layer(layers['h3/drop'], self.num_classes)

        self.layers = layers
        return self.layers['out/fc']

    def train(self, batch_size, learning_rate, num_steps, weights_file, report_file):
        # (1) Get dataset
        dataset = self.DataWrapper(TRAIN_DATA, is_validate=True)
        val_x, val_y = dataset.val_data, dataset.val_label
        val_y = self._convert2onehot(val_y)

        # (2) Create network
        X = tf.placeholder(tf.float32, [None, EMBED_SIZE], name='X')
        Y = tf.placeholder(tf.float32, [None, self.num_classes], name='Y')
        lr = tf.placeholder(tf.float32, [], name='lr')
        is_training = tf.placeholder(tf.bool, name='is_training')
        with tf.variable_scope(self.name):
            out = self.create_net(X, is_training)
            cross_ent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=out))
            var_weights = [v for v in tf.trainable_variables() if v.name.endswith('weights:0')]
            l2_regularizer = tf.contrib.layers.l2_regularizer(0.001)
            regularizer = tf.contrib.layers.apply_regularization(l2_regularizer, var_weights)
            train_loss = cross_ent

            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(train_loss)

        # (3) Start training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(1, num_steps + 1):
                batch_x, batch_y = dataset.next_batch(batch_size)
                batch_y = self._convert2onehot(batch_y)
                if step % 20000 == 0:
                    learning_rate = learning_rate * 0.5
                sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, is_training: True, lr: learning_rate})
                if step % 500 == 0 or step == 1:
                    y_pred, loss = sess.run([out, train_loss], feed_dict={X: val_x, Y: val_y, is_training: False})
                    res = self.report(np.argmax(val_y, 1), np.argmax(y_pred, 1))
                    line = 'Step: ' + str(step)
                    line += '| Val_Loss: {:.4f}'.format(loss)
                    line += '| Val_Accuracy: {:.4f}'.format(res['accuracy'])
                    line += '| Val_Precision: {:.4f}'.format(res['precision'])
                    line += '| Val_Recall: {:.4f}'.format(res['recall'])
                    print(line)
                    with open(report_file, 'a') as f:
                        f.write(line + '\n')
            self.save_weights(sess, weights_file)

    def report(self, y_real, y_pred):
        result = {}
        result['accuracy'] = sk.metrics.accuracy_score(y_real, y_pred)
        result['precision'] = sk.metrics.precision_score(y_real, y_pred)
        result['recall'] = sk.metrics.recall_score(y_real, y_pred)
        result['support'] = np.sum(y_real).astype(int)
        result['total'] = y_real.shape[0]
        return result


if __name__ == '__main__':
    model_name = 'sl_d2vs%d_lstm_allin' % (EMBED_SIZE)
    result_dir = '../result/' + model_name
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    train_file = '%s/train.txt' % (result_dir)
    test_file = '%s/test.txt' % (result_dir)
    weights_file = '../.cache/%s.npz' % (model_name)

    #bincls_model = ClassifyModel('sl0')
    #bincls_model.train(32, 1e-5, 50000, weights_file, train_file)

    lstm_model = LstmModel('sl1')
    lstm_model.train(64, 1e-5, 100000, weights_file, train_file)

