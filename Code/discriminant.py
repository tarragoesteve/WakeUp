import tensorflow as tf
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

training_epochs = 1



rate, data = wavfile.read("/home/esteve/PycharmProjects/WakeUp/Code/Ms_Pacman_Death.wav")
#rate = mostres/sec

f, t, Sxx = signal.spectrogram(x=data, fs=rate)
print(np.shape(Sxx))
# plt.pcolormesh(t, f, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

time_size = len(t)
frequency_size = len(f)
batch_size = 1



############################
def discriminant(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width(time), height(frequency), channels]
    input_layer = tf.reshape(features, [batch_size, time_size, frequency_size, 1])

    image_resize = tf.image.resize_images(input_layer, [32,32] )

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, time_size, frequency_size, 1]
    # Output Tensor Shape: [batch_size, time_size, frequency_size, 32]
    conv1 = tf.layers.conv2d(
        inputs=image_resize,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 32, 32, 32]
    # Output Tensor Shape: [batch_size, 16, 16, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 16, 16, 32]
    # Output Tensor Shape: [batch_size, 16, 16, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)


    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 16, 16, 64]
    # Output Tensor Shape: [batch_size, 8, 8, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, time_size/4, frequency_size/4, 64]
    # Output Tensor Shape: [batch_size, time_size/4 * frequency_size/4 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
    print(pool2_flat)


    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 8 * 8 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 1]
    logits = tf.layers.dense(inputs=dropout, units=1,activation=tf.nn.sigmoid)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.greater(x=logits,y=[.5]),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities":logits
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
#
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
        loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


train_data = [Sxx]
train_labels = [0]
batch_size = 1
features_ph = tf.placeholder(dtype=tf.float32)
labels_ph = tf.placeholder(dtype=tf.int32)
mode_ph = tf.placeholder(dtype=tf.string)
batch_size_ph = tf.placeholder(dtype=tf.int32)
result = discriminant(features=features_ph, labels=labels_ph, mode=mode_ph)

init = tf.global_variables_initializer()

print("Going to start session")
# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)


    print(sess.run(result, feed_dict= {features_ph: [Sxx]}))  # , labels_ph: Y_batch, mode_ph: tf.estimator.ModeKeys.EVAL, batch_size_ph: batch_size})

    # Fit all training data
    for epoch in range(training_epochs):
        for batch in range(len(train_data)/batch_size):
            X_batch = train_data[batch_size*batch:batch_size*(batch+1)]
            Y_batch = train_labels[batch_size*batch:batch_size*(batch+1)]
            print(type(X_batch), type(tf.estimator.ModeKeys.TRAIN),type(batch_size))
            print("Epoch: "+ str(epoch) + "/"+ str(training_epochs)+ " Batch:"+ str(batch) + "/" + str(len(train_data) / batch_size))
            # if (batch % 100 == 0):
            #     with tf.variable_scope("conv1",reuse=True):
            #         c = sess.run(loss, feed_dict={X: mnist.test.images[0:100], mode: tf.estimator.ModeKeys.EVAL, labels: np.asarray(mnist.test.labels[0:100], dtype=np.int32)})
            #         print("cost=", "{:.9f}".format(c))
            #         np.savetxt("E"+str(epoch)+"B"+str(batch)+ ".csv",sess.run(tf.get_variable("my_weights")), fmt='%.9e', delimiter=', ')