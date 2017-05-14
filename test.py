import preprocessing, train
import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm
import argparse

def test(dataset_name, model_name, store_misclassified):
    img_h, img_w = 64, 64

    nn = train.Classifier('classifier', img_w, img_h, len(preprocessing.CLASSES))
    dataset = list(map(lambda f:f.strip(), open(dataset_name, 'r').readlines()))

    n_test = len(dataset)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_name)

        # setting up metrics
        confusion = np.zeros((len(preprocessing.CLASSES), 
                              len(preprocessing.CLASSES)))
        correct, correct_case_insensitive, in_top3 = 0, 0, 0

        for i in tqdm(range(n_test)):

            # get data
            image = preprocessing.open_image(dataset[i], (img_h, img_w))
            label = preprocessing.get_class_index(dataset[i])

            # predict
            classes = sess.run(nn.classes, feed_dict={ nn.input : [image] })
            predicted_label = np.argmax(classes[0])

            # update metrics
            confusion[label, predicted_label] += 1
            if label == predicted_label: correct += 1
            if (preprocessing.CLASSES[label].lower()
                    == preprocessing.CLASSES[predicted_label].lower()):
                correct_case_insensitive += 1
            else:
                if store_misclassified:
                    towrite = np.concatenate((image, np.zeros((20, img_w, 1))))
                    cv2.putText(
                            towrite,
                            preprocessing.CLASSES[predicted_label],
                            (0, img_h+20), cv2.FONT_HERSHEY_PLAIN, 2, 255)
                    cv2.imwrite('misclassified/'+str(i)+'.png', towrite)
            if label in classes[0].argsort()[-3:]: in_top3 += 1

        # showing metrics
        print("Accuracy :", correct/n_test*100)
        print("Case-insensitive accuracy :", correct_case_insensitive/n_test*100)
        print("Top 3 accuracy :", in_top3/n_test*100)

        confusion = (
                confusion / np.array([np.sum(confusion, 1)]).transpose() * 255
        ).astype(np.uint8)
        cv2.imshow('Confusion', cv2.resize(confusion, None, fx=10, fy=10,
            interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=True, help='Dataset name')
    parser.add_argument('-m', type=str, required=True, help='Model name')
    parser.add_argument('-s', action='store_true', help='Store misclassified')

    opt = parser.parse_args()

    test(opt.d, opt.m, opt.s)
