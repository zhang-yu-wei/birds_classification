import tensorflow as tf
import net
from utils import ImageGenerator

batch_size = 2
epochs = 10
learning_rate = 0.0001
report_epoch = 1

gen = ImageGenerator(batch_size, True, './data')


with tf.device('/cpu:0'):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [batch_size, 480, 480, 3])
    true_out = tf.placeholder(tf.float32, [batch_size, 4])
    train_mode = tf.placeholder(tf.bool)

    net = net.Vgg16()
    net.build(images, train_mode)

    sess.run(tf.global_variables_initializer())

    cost = tf.reduce_sum((net.prob - true_out)**2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

    for epoch in range(epochs):
        batch, img_result = gen.next_batch()
        sess.run(train, feed_dict={images: batch, true_out: img_result, train_mode: True})

        if epoch%report_epoch == 0:

            print('epoch: ' + str(epoch) + '|' + 'train_loss: ' + str(train_loss) + '|' +
                  'valid_loss' + str(valid_loss))
        del batch, img_result


