from __future__ import print_function
import numpy as np
import cv2
import os
import sys


test_percent = 0.2
valid_percent = 0.1


def process(class_path, class_name, save_name):

    path = class_path + '/' + class_name
    img_name = [name for name in os.listdir(path) if '.jpg' in name]

    length = len(img_name)
    test_num = round(test_percent*length)
    valid_num = round(valid_percent*length)

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

        # change data size to 640*640
        zeros = np.zeros((280, 640))
        img_r = np.concatenate((zeros, img[:, :, 0]), axis=0).reshape((640, 640, 1))
        img_g = np.concatenate((zeros, img[:, :, 1]), axis=0).reshape((640, 640, 1))
        img_b = np.concatenate((zeros, img[:, :, 2]), axis=0).reshape((640, 640, 1))
        img_new = np.concatenate((img_r, img_g, img_b), axis=2)
        assert np.shape(img_new)[0] == 640
        assert np.shape(img_new)[1] == 640
        assert np.shape(img_new)[2] == 3

        sys.stdout.write("-")
        sys.stdout.flush()

        if count < test_num:
            np.save(class_path + save_name + '/' + 'test/' + class_name + '_' + str(count), img_new)
        else:
            np.save(class_path + save_name + '/' + 'train/' + class_name + '_' + str(count), img_new)

        if count < valid_num:
            np.save(class_path + save_name + '/' + 'valid/' + class_name + '_' + str(count), img_new)

        count += 1

        del img, img_new


if __name__ == '__main__':
    process('.', 'Class_3', '/data')
