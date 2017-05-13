import preprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Classifier():
    def __init__(self, scope, img_w, img_h, n_classes, dropout_keep_prob=1.0):
        self.scope = scope
        self.n_classes = n_classes
        self.dropout_keep_prob = dropout_keep_prob

        self.input = tf.placeholder(tf.float32, [None, img_h, img_w, 3])

        self.conv1 = slim.conv2d(
                tf.nn.dropout(self.input, self.dropout_keep_prob),
                num_outputs=32, kernel_size=[8, 8],
                stride=[2, 2], padding='Valid',
                scope=self.scope+'_conv1'
        )
        self.conv2 = slim.conv2d(
                tf.nn.dropout(self.conv1, self.dropout_keep_prob),
                num_outputs=32, kernel_size=[4, 4],
                stride=[2, 2], padding='Valid',
                scope=self.scope+'_conv2'
        )

        self.classes = slim.fully_connected(
                tf.nn.dropout(slim.flatten(self.conv2), self.dropout_keep_prob),
                self.n_classes,
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
    train_steps = int(1e5)
    batch_size = 10
    model_name = 'dropout'

    nn = Classifier('classifier', img_w, img_h, len(preprocessing.CLASSES), 0.6)
    dataset = list(map(lambda f:f.strip(), open('good_train', 'r').readlines()))
    validation_dataset = list(map(lambda f:f.strip(), 
                                  open('good_validation', 'r').readlines()))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter('summaries/'+model_name)

        for t in range(train_steps):
            
            images, labels = preprocessing.get_batch(dataset, 10, (128, 128))

            loss, _ = sess.run([nn.loss, nn.train_step], feed_dict={
                nn.input   : images,
                nn.targets : labels
            })

            summary = tf.Summary()
            summary.value.add(tag='Loss', simple_value=float(loss))

            if t % 10 == 0: print(loss)
            if t % 2000 == 0: saver.save(sess, 'saves/'+model_name, global_step=t)
            if t % 50 == 0:
                images, labels = preprocessing.get_batch(validation_dataset, 20, (128, 128))
                classes = sess.run(nn.classes, feed_dict={nn.input:images})
                summary.value.add(tag='ValidationError',
                        simple_value=float(sum(np.argmax(classes, -1) != labels)))

            summary_writer.add_summary(summary, t)
            summary_writer.flush()



if __name__ == "__main__":
    train()

