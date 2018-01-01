import os
import random
import shutil
import sklearn as sk
import sklearn.metrics
import tensorflow as tf
import numpy as np

from tagging.config import *
from util.tool import *


class DataWrapper(object):
    def __init__(self, data_file, is_validate=True):
        # Load data
        data = np.load(data_file)
        self.data = data['data']
        self.label = data['label']
        self.label_ratio = np.sum(self.label, axis=0) / self.label.shape[0]
        if is_validate:
            idx = int(0.9 * self.data.shape[0])
            self.val_data = self.data[idx:, :]
            self.data = self.data[:idx, :]
            self.val_label = self.label[idx:, :]
            self.label = self.label[:idx, :]
        print(self.data.shape, self.label.shape)

    def next_batch(self, batch_size):
        if not batch_size or batch_size >= self.data.shape[0]:
            batch_size = self.data.shape[0]
        batch = random.sample(range(self.data.shape[0]), batch_size)
        return self.data[batch, :], self.label[batch, :]


class BaseModel(object):
    def __init__(self, name):
        self.name = name

    def _fc_layer(self, x, size):
        return tf.contrib.layers.fully_connected(x, size, activation_fn=None, scope='fc')

    def _bn_layer(self, x, is_training):
        return tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=is_training, scope='bn')

    def _relu_layer(self, x):
        return tf.nn.relu(x)

    def _elu_layer(self, x):
        return tf.nn.elu(x)

    def _dropout_layer(self, x, keep_prob, is_training):
        return tf.layers.dropout(x, keep_prob, training=is_training)

    def _convert2onehot(self, Y, n_class=2):
        tmp = np.zeros(list(Y.shape) + [n_class])
        tmp[list(np.indices(tmp.shape[:-1])) + [Y.astype(np.int)]] = 1
        return tmp

    #def _convert2onehot(self, Y, n_class=2):
    #    m = Y.shape[0]
    #    tmp = np.zeros((m, n_class))
    #    tmp[np.arange(m),Y.astype(int)] = 1.0
    #    return tmp

    def save_weights(self, sess, filename):
        weights = {}
        for v in tf.trainable_variables():
            weights[v.name] = sess.run(v)
        np.savez(filename, **weights)


class NNModel(BaseModel):
    def __init__(self, name):
        super(NNModel, self).__init__(name)
        self.h1_size = 256
        self.h2_size = 256
        self.h3_size = 256
        self.h4_size = 512
        self.h5_size = 512
        self.h6_size = 512
        self.h7_size = 256
        self.h8_size = 256
        self.h9_size = 256
        self.num_classes = len(load_json(ID2TAG))

    def create_net(self, X, is_training):
        return self.create_shallow_net(X, is_training)
        #return self.create_deep_net(X, is_training)

    def create_shallow_net(self, X, is_training):
        layers = {}

        with tf.variable_scope('h1'):
            layers['h1/fc'] = self._fc_layer(X, 256)
            layers['h1/relu'] = self._relu_layer(layers['h1/fc'])
            layers['h1/drop'] = self._dropout_layer(layers['h1/relu'], 0.1, is_training)

        with tf.variable_scope('h2'):
            layers['h2/fc'] = self._fc_layer(layers['h1/drop'], 128)
            layers['h2/relu'] = self._relu_layer(layers['h2/fc'])
            layers['h2/drop'] = self._dropout_layer(layers['h2/relu'], 0.1, is_training)

        with tf.variable_scope('h3'):
            layers['h3/fc'] = self._fc_layer(layers['h2/drop'], 128)
            layers['h3/relu'] = self._relu_layer(layers['h3/fc'])
            layers['h3/drop'] = self._dropout_layer(layers['h3/relu'], 0.1, is_training)

        with tf.variable_scope('out'):
            layers['out/fc'] = self._fc_layer(layers['h3/drop'], self.num_classes)

        self.layers = layers
        return self.layers['out/fc']

    def create_deep_net(self, X, is_training):
        layers = {}

        with tf.variable_scope('h1'):
            layers['h1/fc'] = self._fc_layer(X, self.h1_size)
            layers['h1/relu'] = self._relu_layer(layers['h1/fc'])
            layers['h1/drop'] = self._dropout_layer(layers['h1/relu'], 0.1, is_training)

        with tf.variable_scope('h2'):
            layers['h2/fc'] = self._fc_layer(layers['h1/drop'], self.h2_size)
            layers['h2/relu'] = self._relu_layer(layers['h2/fc'])
            layers['h2/drop'] = self._dropout_layer(layers['h2/relu'], 0.1, is_training)

        with tf.variable_scope('h3'):
            layers['h3/fc'] = self._fc_layer(layers['h2/drop'], self.h3_size)
            layers['h3/res'] = tf.add(layers['h1/drop'], layers['h3/fc'])
            layers['h3/relu'] = self._relu_layer(layers['h3/res'])
            layers['h3/drop'] = self._dropout_layer(layers['h3/relu'], 0.1, is_training)

        with tf.variable_scope('h4'):
            layers['h4/fc'] = self._fc_layer(layers['h3/drop'], self.h4_size)
            layers['h4/relu'] = self._relu_layer(layers['h4/fc'])
            layers['h4/drop'] = self._dropout_layer(layers['h4/relu'], 0.1, is_training)

        with tf.variable_scope('h5'):
            layers['h5/fc'] = self._fc_layer(layers['h4/drop'], self.h5_size)
            layers['h5/relu'] = self._relu_layer(layers['h5/fc'])
            layers['h5/drop'] = self._dropout_layer(layers['h5/relu'], 0.1, is_training)

        with tf.variable_scope('h6'):
            layers['h6/fc'] = self._fc_layer(layers['h5/drop'], self.h6_size)
            layers['h6/res'] = tf.add(layers['h4/drop'], layers['h6/fc'])
            layers['h6/relu'] = self._relu_layer(layers['h6/res'])
            layers['h6/drop'] = self._dropout_layer(layers['h6/relu'], 0.1, is_training)

        with tf.variable_scope('h7'):
            layers['h7/fc'] = self._fc_layer(layers['h6/drop'], self.h7_size)
            layers['h7/relu'] = self._relu_layer(layers['h7/fc'])
            layers['h7/drop'] = self._dropout_layer(layers['h7/relu'], 0.1, is_training)

        with tf.variable_scope('h8'):
            layers['h8/fc'] = self._fc_layer(layers['h7/drop'], self.h8_size)
            layers['h8/relu'] = self._relu_layer(layers['h8/fc'])
            layers['h8/drop'] = self._dropout_layer(layers['h8/relu'], 0.1, is_training)

        with tf.variable_scope('h9'):
            layers['h9/fc'] = self._fc_layer(layers['h8/drop'], self.h9_size)
            layers['h9/res'] = tf.add(layers['h7/drop'], layers['h9/fc'])
            layers['h9/relu'] = self._relu_layer(layers['h9/res'])
            layers['h9/drop'] = self._dropout_layer(layers['h9/relu'], 0.1, is_training)

        with tf.variable_scope('out'):
            layers['out/fc'] = self._fc_layer(layers['h9/drop'], self.num_classes)

        self.layers = layers
        return self.layers['out/fc']

    def train(self, batch_size, learning_rate, num_steps, weights_file, report_file, label_id):
        # (1) Get dataset
        dataset = DataWrapper(TRAIN_DATA, is_validate=True)
        label_ratio = dataset.label_ratio#[label_id]
        val_x, val_y = dataset.val_data, dataset.val_label#[:,label_id]
        #val_y = self._convert2onehot(val_y)

        # (2) Create network
        X = tf.placeholder(tf.float32, [None, EMBED_SIZE], name='X')
        Y = tf.placeholder(tf.float32, [None, self.num_classes], name='Y')
        is_training = tf.placeholder(tf.bool, name='is_training')
        lr = tf.placeholder(tf.float32, [], name='lr')
        label_ratio = np.log(label_ratio)
        weighted = 2 - label_ratio / np.min(label_ratio)
        weighted = tf.constant(weighted, dtype=tf.float32)
        with tf.variable_scope(self.name):
            out = self.create_net(X, is_training)
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=out)
            weighted_cross_ent = tf.multiply(cross_ent, weighted)
            weighted_cross_ent = tf.reduce_mean(weighted_cross_ent)
            #var_loss = [tf.nn.l2_loss(v) for v in tf.trainable_variables() if v.name.endswith('weights:0')]
            #regularizer = tf.add_n(var_loss)
            #train_loss = weighted_cross_ent
            train_loss = tf.reduce_mean(cross_ent)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(train_loss)

        # (3) Start training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #sess.run(tf.local_variables_initializer())
            for step in range(1, num_steps + 1):
                batch_x, batch_y = dataset.next_batch(batch_size)
                #batch_y = self._convert2onehot(batch_y[:,label_id])
                sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, is_training: True, lr: learning_rate})

                if step % 5000 == 0:
                    learning_rate = learning_rate * 0.95

                if step % 500 == 0 or step == 1:
                    training_loss, = sess.run([train_loss], feed_dict={X: batch_x, Y: batch_y, is_training: False})
                    y_pred, val_loss = sess.run([out, train_loss], feed_dict={X: val_x, Y: val_y, is_training: False})
                    y_pred = sigmoid(y_pred)
                    y_pred[y_pred > 0.5] = 1.
                    y_pred[y_pred <= 0.5] = 0.
                    res = self.report(val_y.flatten(), y_pred.flatten())
                    line = 'Step: ' + str(step)
                    line += '| Train_Loss: {:.4f}'.format(training_loss)
                    line += '| Val_Loss: {:.4f}'.format(val_loss)
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

    def save_weights(self, sess, filename):
        weights = {}
        #for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        for v in tf.trainable_variables():
            weights[v.name] = sess.run(v)
        np.savez(filename, **weights)

    def test(self, weights_file, output_file, label_id):
        # (1) Get dataset
        dataset = DataWrapper(TEST_DATA, is_validate=False)

        # (2) Create network
        X = tf.placeholder(tf.float32, [None, EMBED_SIZE], name='X')
        Y = tf.placeholder(tf.float32, [None, self.num_classes], name='Y')
        is_training = tf.placeholder(tf.bool, name='is_training')
        with tf.variable_scope(self.name, reuse=True):
            out = self.create_net(X, is_training)

        # (3) Start testing
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for v in tf.trainable_variables():
                weights = np.load(weights_file)
                sess.run(v.assign(weights[v.name]))
            batch_x, batch_y = dataset.next_batch(None)
            #batch_y = self._convert2onehot(batch_y[:,label_id])
            y_pred, = sess.run([out], feed_dict={X: batch_x, Y: batch_y, is_training: False})
            y_pred = sigmoid(y_pred)
            y_pred[y_pred > 0.5] = 1.
            y_pred[y_pred <= 0.5] = 0.
            res = self.report(batch_y.flatten(), y_pred.flatten())
            with open(output_file, 'a') as outfile:
                outfile.write(weights_file + '\n')
                line = 'Accuracy=%2.4f | ' % (res['accuracy'])
                line += 'Precision=%2.4f | ' % (res['precision'])
                line += 'Recall=%2.4f | ' % (res['recall'])
                line += 'Support=%d/%d\n\n' % (res['support'], res['total'])
                outfile.write(line)


def main(model_name):
    result_dir = '../result/' + model_name
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    train_file = lambda label: '%s/train_c%d.txt' % (result_dir, label)
    test_file = '%s/test.txt' % (result_dir)
    weights_file = lambda label: '../.cache/%s_c%d.npz' % (model_name, label)

    for label_id in range(1,len(load_json(ID2TAG))):
        nn_model = NNModel('cls' + str(label_id))
        nn_model.train(128, 1e-4, 100000, weights_file(label_id), train_file(label_id), label_id)
        nn_model.test(weights_file(label_id), test_file, label_id)
        break


if __name__ == '__main__':
    # TSC-based doc2vec tagging
    model_name = 'tag_d2vs%d_42cls_weight_small' % (EMBED_SIZE)
    main(model_name)

    # TSC-based word2vec tagging
    #model_name = 'tag_w2vs%d_42cls_weight' % (EMBED_SIZE)
    #main(model_name)

    # Intro-based doc2vec tagging
    #model_name = 'intro_d2vs%d_42cls_weight' % (EMBED_SIZE)
    #main(model_name)
