import tensorflow as tf
from utils import ImageGenerator
from net import Vgg16


generator = ImageGenerator('./data')
test_data, test_label = generator.get_test()

tf.reset_default_graph()

images = tf.placeholder(tf.float32, [None, 480, 480, 3])
true_out = tf.placeholder(tf.float32, [None, 4])
train_mode = tf.placeholder(tf.bool)

network = Vgg16(vgg16_npy_path='./train-save.npy')
network.build(images, train_mode)

with tf.device('/cpu:0'):
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    correct = tf.equal(tf.argmax(network.prob, 1), tf.argmax(true_out, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    acc = sess.run(accuracy, feed_dict={images: test_data, true_out: test_label, train_mode: False})
    print('Accuracy for test data: %s' % acc)
