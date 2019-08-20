import numpy as np
import os
import random


class ImageGenerator():
    def __init__(self, path, batch_size=10, test_num=100, valid_num=100):
        self.batch_size = batch_size
        self.last = 0
        self.path = path

        self.train_list = [name for name in os.listdir(path + '/' + 'train') if '.npy' in name]
        random.shuffle(self.train_list)

        self.test_list = [name for name in os.listdir(path + '/' + 'test') if '.npy' in name]
        random.shuffle(self.test_list)

        self.valid_list = [name for name in os.listdir(path + '/' + 'valid') if '.npy' in name]
        random.shuffle(self.valid_list)

        self.train_num = len(self.train_list)
        self.test_num = min(len(self.test_list), test_num)
        self.valid_num = min(len(self.valid_list), valid_num)

    def next_batch(self):
        train_last = self.train_num - 1
        data_batch = np.zeros([self.batch_size, 480, 480, 3])
        label_batch = np.zeros([self.batch_size, 4])
        count = 0

        if self.last > train_last - self.batch_size:
            new_size = train_last - self.last
            left_size = self.batch_size - new_size
            while (self.last < train_last):
                data_name = self.train_list[self.last]
                data_batch[count] = np.load(self.path + '/' + 'train' + '/' + data_name)
                label_batch[count] = [1 if i == int(data_name[6]) else 0 for i in range(4)]
                count += 1
                self.last += 1
            self.last = 0
            while(self.last < left_size):
                data_name = self.train_list[self.last]
                data_batch[count] = np.load(self.path + '/' + 'train' + '/' + data_name)
                label_batch[count] = [1 if i == int(data_name[6]) else 0 for i in range(4)]
                count += 1
                self.last += 1
        else:
            while(count < self.batch_size):
                data_name = self.train_list[self.last]
                data_batch[count] = np.load(self.path + '/' + 'train' + '/' + data_name)
                label_batch[count] = [1 if i == int(data_name[6]) else 0 for i in range(4)]
                count += 1
                self.last += 1
        return data_batch, label_batch

    def get_valid(self):
        data_batch = np.zeros([self.valid_num, 480, 480, 3])
        label_batch = np.zeros([self.valid_num, 4])
        count = 0
        while(count < self.valid_num):
            data_name = self.valid_list[count]
            data_batch[count] = np.load(self.path + '/' + 'valid' + '/' + data_name)
            label_batch[count] = [1 if i == int(data_name[6]) else 0 for i in range(4)]
            count += 1

        return data_batch, label_batch

    def get_test(self):
        data_batch = np.zeros([self.test_num, 480, 480, 3])
        label_batch = np.zeros([self.test_num, 4])
        count = 0
        while(count < self.test_num):
            data_name = self.test_list[count]
            data_batch[count] = np.load(self.path + '/' + 'test' + '/' + data_name)
            label_batch[count] = [1 if i == int(data_name[6]) else 0 for i in range(4)]
            count += 1

        return data_batch, label_batch





