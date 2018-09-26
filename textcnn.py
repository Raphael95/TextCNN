import tensorflow as tf
import numpy as np
import data_helper
import time
from sklearn.model_selection import train_test_split


# to define the input units, hidden units and output units
input_units = 5000
hidden_units = 200
output_units = 15

# to define the embedding dim
embedding_dim = 100
vocab_num = 56981

# to define the filter num ans size
filters = [3, 4, 5, 6]
filter_num = 80

# create a session
session = tf.Session()

# initialize weight using gaussian distribute
def init_weight_by_gaussian(shape):
    init = tf.truncated_normal(shape=shape, stddev=0.1)
    weight = tf.Variable(init)
    return weight

# initialize bias all zero
def init_bias_zero(shape):
    init = tf.constant(0.0, shape=shape)
    bias = tf.Variable(init)
    return bias

# initialize bias all 0.1
def init_bias(shape):
    init = tf.constant(0.1, shape=shape)
    bias = tf.Variable(init)
    return bias

# define the convolution function
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')

# define the max pooling function
def max_pooling(x, after_filter_dim):
    return tf.nn.max_pool(x, ksize=[1, after_filter_dim, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

# define the input x, output y
x = tf.placeholder(tf.int64, [None, input_units])
y = tf.placeholder(tf.float32, [None, output_units])

# embedding layer
embedding = tf.Variable(tf.constant(0.0, shape=[vocab_num, embedding_dim]), trainable=True)
embedding_placeholder = tf.placeholder(tf.float32, [vocab_num, embedding_dim])
embedding_init = embedding.assign(embedding_placeholder)

embedding_output = tf.nn.embedding_lookup(embedding, x)
embedding_output = tf.expand_dims(embedding_output, axis=-1)

# convolutional layer
## iterate the filter and begin to convolution on the sentence
pooled_outs = []
for i, filter in enumerate(filters):
    filter_shape = [filter, embedding_dim, 1, filter_num]
    w_conv1 = init_weight_by_gaussian(filter_shape)
    b_conv1 = init_bias([filter_num])
    out_conv1 = tf.nn.relu(conv2d(embedding_output, w_conv1) + b_conv1)
    out_pooling1 = max_pooling(out_conv1, input_units - filter + 1)
    pooled_outs.append(out_pooling1)


total_filter_num = filter_num * len(filters)
out_poolings = tf.concat(pooled_outs, axis=3)
out_poolings_flat = tf.reshape(out_poolings, [-1, total_filter_num])

# define the fully connected layer
w_fc1 = init_weight_by_gaussian([total_filter_num, hidden_units])
b_fc1 = init_bias([hidden_units])
out_fc1 = tf.nn.relu(tf.matmul(out_poolings_flat, w_fc1) + b_fc1)

# drop out layer
keep_prob = tf.placeholder(tf.float32)
out_drop = tf.nn.dropout(out_fc1, keep_prob=keep_prob)

# softmax layer
w_fc2 = init_weight_by_gaussian([hidden_units, output_units])
b_fc2 = init_bias([output_units])
y_hat = tf.nn.softmax(tf.matmul(out_drop, w_fc2) + b_fc2)

# define the cross entropy loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1]))

# define the optimizer
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

# evaluate the accuracy of model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# load the data and preprocess input data
start_time = time.time()
preprocess = data_helper.preprocess()
x1, y1 = preprocess.load_data()
preprocess.tokenize_word(x1)
model = preprocess.generate_word2vec()
vocab, word_vec = preprocess.modify_word2vec(model)
input_data = preprocess.cal_sentence_word(vocab)
max_len = preprocess.max_len_sentence(input_data)
input_data = preprocess.makeup_sentence(input_data, 5000)
print("input_data :", input_data)
print("input_data shape is :", input_data.shape)
print("vocabulary num : ", np.shape(word_vec))
label = preprocess.one_hot_encoder(y1)
print("label is :", label)
print("label shape is :", np.shape(label))

# split the input_data into train_data and test_data
train_x, test_x, train_y, test_y = train_test_split(input_data, label, test_size=0.2, random_state=0)

batch = data_helper.generate_batch(train_x, train_y, 600, 10)


# define the global initializer
init = tf.global_variables_initializer()
session.run(init)
print(session.run(embedding_init, feed_dict={embedding_placeholder: word_vec}))


for i in range(600):
    batch_x, batch_y = batch.__next__()
    print("batch_x shape is : ", np.shape(batch_x))
    print("batch_y shape is : ", np.shape(batch_y))
    session.run([train_step, out_pooling1], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

end_time = time.time()

print("test accuracy is : ", accuracy.eval(session=session, feed_dict={x: test_x, y: test_y, keep_prob: 1.0}))

print("运行时间 : ", (end_time - start_time) / 60, ' m')









