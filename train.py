import preprocessing
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Classifier():
    def __init__(self, scope, img_w, img_h, n_classes):
        self.scope = scope
        self.n_classes = n_classes

        self.input = tf.placeholder(tf.float32, [None, img_h, img_w, 3])

        self.conv1 = slim.conv2d(
                self.input,
                num_outputs=32, kernel_size=[8, 8],
                stride=[2, 2], padding='Valid',
                scope=self.scope+'_conv1'
        )

        self.classes = slim.fully_connected(
                slim.flatten(self.conv1), self.n_classes,
                scope=self.scope+'_fc',
                activation_fn=None
        )

        self.targets = tf.placeholder(tf.int32, [None])
        self.targets_onehot = tf.one_hot(self.targets, self.n_classes)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.targets_onehot,
                logits=self.classes
        ))
        self.train_step = tf.train.RMSPropOptimizer(1e-3).minimize(self.loss)


def train():
    img_h, img_w = 128, 128
    train_steps = int(1e4)
    batch_size = 10

    nn = Classifier('classifier', img_w, img_h, len(preprocessing.CLASSES))
    dataset = list(map(lambda f:f.strip(), open('good_train', 'r').readlines()))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter('summaries/one')

        for t in range(train_steps):
            
            images, labels = preprocessing.get_batch(dataset, 10, (128, 128))

            loss, _ = sess.run([nn.loss, nn.train_step], feed_dict={
                nn.input   : images,
                nn.targets : labels
            })

            summary = tf.Summary()
            summary.value.add(tag='Loss', simple_value=float(loss))
            summary_writer.add_summary(summary, t)
            summary_writer.flush()

            if t % 10 == 0: print(loss)
            if t % 500 == 0: saver.save(sess, 'saves/one', global_step=t)


if __name__ == "__main__":
    train()

