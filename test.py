import preprocessing, train
import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm

def test():
    img_h, img_w = 128, 128

    nn = train.Classifier('classifier', img_w, img_h, len(preprocessing.CLASSES))
    dataset = list(map(lambda f:f.strip(), open('good_validation', 'r').readlines()))

    n_test = len(dataset)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './saves/grayscale_pool_stretch-30000')

        confusion = np.zeros((len(preprocessing.CLASSES), len(preprocessing.CLASSES)))
        good = 0
        good_case_insensitive = 0
        in_top3 = 0
        for i in tqdm(range(n_test)):
            image = preprocessing.open_image(dataset[i])
            label = preprocessing.get_class_index(dataset[i])
            classes = sess.run(nn.classes, feed_dict={ nn.input : [image] })
            predicted_label = np.argmax(classes[0])

            confusion[label, predicted_label] += 1
            if label == predicted_label:
                good += 1
            if preprocessing.CLASSES[label].lower() == preprocessing.CLASSES[predicted_label].lower():
                good_case_insensitive += 1
            if label in classes[0].argsort()[-3:]:
                in_top3 += 1

        print("Accuracy :", good/n_test*100)
        print("Case-insensitive accuracy :", good_case_insensitive/n_test*100)
        print("Top 3 accuracy :", in_top3/n_test*100)
        confusion = (
                confusion / np.array([np.sum(confusion, 1)]).transpose() * 255
        ).astype(np.uint8)
        cv2.imshow('Confusion', cv2.resize(confusion, None, fx=10, fy=10,
            interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(-1)

if __name__ == "__main__":
    test()
