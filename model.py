import os
import yaml
import numpy as np
import tensorflow as tf
import pre_process
import nltk
from sklearn.utils import shuffle
import math

def change_word_to_id(vocab, sentences, max_document_length, not_exist_index):
	current_batch_x_id=[]
	for sentence in sentences:
		sentence_word_ids = []
		for word in nltk.word_tokenize(sentence):
			if word in vocab:
				index = vocab.index(word)
			else:
				index = not_exist_index
			sentence_word_ids.append(index)
			#zero padding
		if len(sentence_word_ids) <= max_document_length:
			sentence_word_ids = np.concatenate((sentence_word_ids, np.full((max_document_length-len(sentence_word_ids)), not_exist_index)), axis=0)

		current_batch_x_id.append(sentence_word_ids)
	return current_batch_x_id


def glove_embeddings(glove_dir, dim):
	"""
	this function returns glove word vectors in a dict of the form {'word' : <vector>}
	parameters : glove_dir is the directory location of the glove file for word embeddings/vectos
	"""
	vocab = []
	embeddings = []
	f = open(os.path.join(glove_dir, 'glove.6B.' + str(dim) + 'd.txt'))
	for line in f:
		values = line.strip().split(' ')
		vocab.append(values[0])
		embeddings.append(values[1:])
	print 'loaded glove'
	f.close()

	return vocab, embeddings

def batchnorm(y, is_test, iteration, offset):
	exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
	bnepsilon = 1e-5
	mean, variance = tf.nn.moments(y, [0, 1, 2])
	update_moving_averages = exp_moving_avg.apply([mean, variance])
	m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
	v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
	y_batch_normalised = tf.nn.batch_normalization(y, m, v, offset, None, bnepsilon)
	return y_batch_normalised, update_moving_averages

if __name__ == '__main__':

	# Data loading params
	tf.flags.DEFINE_float("dev_test_percentage", .1, "Percentage of the training data to use for testing")
	tf.flags.DEFINE_float("dev_validation_percentage", .1, "Percentage of the training data to use for validation")
	tf.flags.DEFINE_string("positive_data_file", "./data/pos.txt", "Data source for the positive data.")
	tf.flags.DEFINE_string("negative_data_file", "./data/neg.txt", "Data source for the negative data.")

	tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
	tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
	tf.flags.DEFINE_integer("report_error_freq", 50, "How often to log error in the terminal")

	FLAGS = tf.flags.FLAGS
	FLAGS._parse_flags()

	print 'loading data ...'
	# y_real structure - [neg_label, pos_label] -> eg. positive post is [0,1]
	x_test, y_real = pre_process.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
	max_document_length = max([len(x.split(" ")) for x in x_test])

	# Randomly shuffle data
	np.random.seed(10)
	shuffle_indices = np.random.permutation(len(y_real))
	x_shuffled, y_shuffled = shuffle(x_test, y_real, random_state=0)

	#split train/test set
	dev_sample_index = -1 * int(FLAGS.dev_test_percentage * float(len(y_shuffled)))
	x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
	y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

	#load config file and parameters in it
	with open('config.yml') as config_file:
		config = yaml.load(config_file)
	glove_dir = config['glove_dir']
	data_file = config['data_file']
	num_classes = config['num_classes']
	embedding_size = config['embedding_size']
	learning_rate_file = config['learning_rate']
	dropout_file = config['dropout']
	num_filters = config['num_filters']
	config_file.close()

	#session variable
	sess = tf.Session()

	#get glove vectors
	vocab, embeddings = glove_embeddings(glove_dir, embedding_size)
	vocab.append('not_exist')
	not_exist_index = len(vocab) - 1
	embeddings.append(np.zeros(embedding_size))

	#get new vectors into embedding/vocab
	vocab_train_data = set(nltk.word_tokenize(' '.join(x_train)))
	only_in_train = vocab_train_data - set(vocab)
	print 'not in glove embeddings, length - ', len(only_in_train)
	for word in only_in_train:
		vocab.append(word)
		embeddings.append(np.zeros(embedding_size))
	
	vocab_size = len(vocab)
	embedding_dim = len(embeddings[0])
	emb = np.asarray(embeddings)

	#W is the tensorflow variable that would hold the word vectors according to the id's which we can search 
	#using embedding_lookup function
	W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=True, name="W")
	embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
	embedding_init = W.assign(embedding_placeholder)
	sess.run(embedding_init, feed_dict={embedding_placeholder: emb})

	#-------- MODEL GRAPH NODES --------
	#inputs
	input_x = tf.placeholder(tf.int32, [None, max_document_length], name='input_x')
	input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
	dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
	is_test = tf.placeholder(tf.bool)

	#get vector embedding of a sentence
	input_sentence_vec = tf.nn.embedding_lookup(W, input_x)

	#icreasing dimension
	input_sentence_vec = tf.expand_dims(input_sentence_vec, -1)

	pooled_outputs = []
	filter_sizes = [3,4,5]
	#num_filters is read from the file
	for i, filter_size in enumerate(filter_sizes):
		with tf.name_scope("conv-maxpool-%s" % filter_size):
			# Convolution Layer
			filter_shape = [filter_size, embedding_size, 1, num_filters]
			W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
			b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
			conv = tf.nn.conv2d(
				input_sentence_vec,
				W,
				strides=[1, 1, 1, 1],
				padding="VALID",
				name="conv")
			#batch normalisation
			# conv, update_moving_averages = batchnorm(conv, is_test, i, b)

			# Apply nonlinearity
			h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

			# Max-pooling over the outputs
			pooled = tf.nn.max_pool(
				h,
				ksize=[1, max_document_length - filter_size + 1, 1, 1],
				strides=[1, 1, 1, 1],
				padding='VALID',
				name="pool")
			pooled_outputs.append(pooled)
	 
	# Combine all the pooled features
	num_filters_total = num_filters * len(filter_sizes)
	h_pool = tf.concat(pooled_outputs, 3)
	h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

	# Add dropout
	with tf.name_scope("dropout"):
		h_drop = tf.nn.dropout(h_pool_flat, dropout_prob)

	with tf.name_scope("output"):
		W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
		scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
		predictions = tf.argmax(scores, 1, name="predictions")

	# Calculate mean cross-entropy loss
	with tf.name_scope("loss"):
		losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
		losses = tf.reduce_mean(losses)

	# Calculate Accuracy
	with tf.name_scope("accuracy"):
		correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


	learning_rate = tf.placeholder(tf.float32)
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(losses)
	init = tf.global_variables_initializer()
	sess.run(init)

	#-------- INPUTS ---------
	for x in range(1, FLAGS.num_epochs):
		print '---------- EPOCH - ', x, ' ----------'
		batch_x = pre_process.chunks(x_train, FLAGS.batch_size)
		batch_y = pre_process.chunks(y_train, FLAGS.batch_size)
		for i in range(0, int(math.ceil(len(y_train)/FLAGS.batch_size)-1) ):
			try:
				current_batch_x = next(batch_x)
				current_batch_y = next(batch_y)
			except StopIteration:
				break

			#get word id list for each word in each sentence of the batch_x
			current_batch_x_id = change_word_to_id(vocab, current_batch_x, max_document_length, not_exist_index)
			_, accuracy_report = sess.run([train_step, accuracy], feed_dict={input_x:current_batch_x_id, input_y:current_batch_y, learning_rate : learning_rate_file, dropout_prob : dropout_file, is_test: False})
			# sess.run(update_moving_averages, feed_dict={input_x:current_batch_x_id, input_y:current_batch_y, learning_rate : learning_rate_file, dropout_prob : 1.0, is_test: False})

			if i % FLAGS.report_error_freq == 0:
				print 'accuracy - ', i, accuracy_report
		x_test_acc = change_word_to_id(vocab, x_dev, max_document_length, not_exist_index)
		test_accuracy,_ = sess.run([accuracy, predictions], feed_dict={input_x:x_test_acc, input_y:y_dev, dropout_prob : np.float32(1.0), is_test: True})
		print 'test accuracy - ', test_accuracy

	



