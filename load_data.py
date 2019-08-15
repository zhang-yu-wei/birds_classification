import numpy as np
import cv2
import os
import sys


def process(class_path, class_name, save_name):
    path = class_path + '/' + class_name
    img_name = [name for name in os.listdir(path) if '.jpg' in name]

    length = len(img_name)

    # load images
    sys.stdout.write("[%s]" % (" " * length))
    sys.stdout.flush()
    sys.stdout.write("\b" * (length+1))
    count = 0
    for name in img_name:
        img = cv2.imread(path + '/' + name)
        img = img/255.0
        assert np.shape(img)[0] == 360
        assert np.shape(img)[1] == 640
        assert np.shape(img)[2] == 3
        assert (0 <= img).all() and (img <= 1.0).all()

        # change data size to 480*480
        cut_r = img[:, :160, 0].reshape(120, 480, 1)
        rest_r = img[:, 160:, 0].reshape(360, 480, 1)
        cut_g = img[:, :160, 1].reshape(120, 480, 1)
        rest_g = img[:, 160:, 1].reshape(360, 480, 1)
        cut_b = img[:, :160, 2].reshape(120, 480, 1)
        rest_b = img[:, 160:, 2].reshape(360, 480, 1)
        img_r = np.concatenate((rest_r, cut_r), axis=0)
        img_g = np.concatenate((rest_g, cut_g), axis=0)
        img_b = np.concatenate((rest_b, cut_b), axis=0)
        img_new = np.concatenate((img_r, img_g, img_b), axis=2)
        assert np.shape(img_new)[0] == 480
        assert np.shape(img_new)[1] == 480
        assert np.shape(img_new)[2] == 3

        sys.stdout.write("-")
        sys.stdout.flush()

        # save
        np.save(class_path + '/' + save_name + '/' + class_name + '_' + str(count), img_new)

        count += 1

        del img, img_new


if __name__ == '__main__':
    class_name = 'Class_0'
    class_path = '.'  # should be the directory containing class sets
    process(class_path, 'Class_0', 'class_0_array')