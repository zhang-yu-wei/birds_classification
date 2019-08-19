import tensorflow as tf
import net
from utils import ImageGenerator

batch_size = 2
epochs = 10
learning_rate = 0.0001
report_epoch = 1
summary_path = './summary'
loss_history = []

generator = ImageGenerator(batch_size, True, './data')
valid_data, valid_label = generator.get_valid()

with tf.device('/cpu:0'):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [None, 480, 480, 3])
    true_out = tf.placeholder(tf.float32, [None, 4])
    train_mode = tf.placeholder(tf.bool)

    net = net.Vgg16()
    net.build(images, train_mode)

    sess.run(tf.global_variables_initializer())

    cost = tf.reduce_sum((net.prob - true_out)**2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    correct = tf.equal(tf.argmax(net.prob, 1), tf.argmax(true_out, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    tf.summary.scalar('cost', cost)
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summary_path + '/train', sess.graph)
    valid_writer = tf.summary.FileWriter(summary_path + '/valid')

    for epoch in range(epochs):
        batch, img_result = generator.next_batch()
        summary_train, _ = sess.run([merged, train], feed_dict={images: batch, true_out: img_result, train_mode: True})
        train_writer.add_summary(summary_train, epoch)

        if epoch % report_epoch == 0:
            summary_test, acc = sess.run([merged, accuracy],
                                         feed_dict={images: valid_data, true_out: valid_label, train_mode: False})
            valid_writer.add_summary(summary_test, epoch)
            print('Accuracy at step %s: %s' % (epoch, acc))

        del batch, img_result
