import os
import numpy as np
import math
import random

imgs0 = [name for name in os.listdir('./class_0_array') if '.npy' in name]
random.shuffle(imgs0)
train_num = math.ceil(0.7*len(imgs0))
count = 0
while(count<train_num):
    img = np.load('./class_0_array' + '/' + imgs0[count])
    np.save('./data/train' + '/' + imgs0[count][:-4], img)

    count += 1