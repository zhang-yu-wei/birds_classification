import os
import random
import numpy as np
import matplotlib.image as mpimg

path = './Class_3'
img_name = os.listdir(path)
random.shuffle(img_name)

print('loading done...')

img_num = len(img_name)
load_num = int(img_num*0.1)

print(img_num)

count = 0
total = np.zeros([360, 640, 3])
while(count<load_num):
  img = mpimg.imread(path + '/' + img_name[count])
  total += img
  count += 1
mean = total/load_num
np.save('mean_3', mean)