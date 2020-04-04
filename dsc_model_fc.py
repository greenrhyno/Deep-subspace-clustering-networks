# Code Authors: Pan Ji,     University of Adelaide,         pan.ji@adelaide.edu.au
#               Tong Zhang, Australian National University, tong.zhang@anu.edu.au
# Copyright Reserved!

# edits made Mar 2019 by Ryan Green, Northeastern University, green.ry@husky.neu.edu

from __future__ import division, print_function, absolute_import

import os
import numpy as np
import warnings
# filter all FutureWarnings in tensorflow
with warnings.catch_warnings():  
	warnings.filterwarnings("ignore",category=FutureWarning)
	import tensorflow as tf
	from tensorflow.contrib import layers
import matplotlib.pyplot as plt
from sklearn import cluster
from munkres import Munkres
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import ipdb # TODO - remove

## Functions for building autoencoder

def encoder(x, weights, n_layers):
	shapes = []
	shapes.append(x.get_shape().as_list())
	
	layeri = tf.nn.bias_add(tf.matmul(x, weights['enc_w0']), weights['enc_b0'])
	layeri = tf.nn.relu(layeri)
	shapes.append(layeri.get_shape().as_list())
	
	iter_i = 1
	while iter_i < n_layers:
		layeri = tf.nn.bias_add(tf.matmul(layeri, weights['enc_w' + str(iter_i)]), weights['enc_b' + str(iter_i)])
		layeri = tf.nn.relu(layeri)
		shapes.append(layeri.get_shape().as_list())
		iter_i = iter_i + 1
	
	layer3 = layeri
	return  layer3, shapes

# Building the decoder
def decoder(z, weights, shapes, n_layers):	
	layer3 = z
	iter_i = 0
	while iter_i < n_layers:
		layer3 = tf.nn.bias_add(tf.matmul(layer3, weights['dec_w' + str(iter_i)], transpose_b=True), weights['dec_b' + str(iter_i)])
		layer3 = tf.nn.relu(layer3)
		iter_i = iter_i + 1
	return layer3


def initialize_weights(input_dim, n_hidden, reg):
	all_weights = dict()
	n_layers = len(n_hidden)
	
	all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[input_dim, n_hidden[0]], initializer=layers.xavier_initializer(), regularizer = reg)
	all_weights['enc_b0'] = tf.Variable(tf.zeros([n_hidden[0]], dtype = tf.float32)) # , name = 'enc_b0'
	
	iter_i = 1
	while iter_i < n_layers:
		enc_name_wi = 'enc_w' + str(iter_i)
		all_weights[enc_name_wi] = tf.get_variable(enc_name_wi, shape=[n_hidden[iter_i-1], n_hidden[iter_i]], initializer=layers.xavier_initializer(),regularizer = reg)
		enc_name_bi = 'enc_b' + str(iter_i)
		all_weights[enc_name_bi] = tf.Variable(tf.zeros([n_hidden[iter_i]], dtype = tf.float32)) # , name = enc_name_bi
		iter_i = iter_i + 1
			
	iter_i = 1
	while iter_i < n_layers:	
		dec_name_wi = 'dec_w' + str(iter_i - 1)
		all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[n_hidden[n_layers-iter_i-1],n_hidden[n_layers-iter_i]], initializer=layers.xavier_initializer(),regularizer = reg)
		dec_name_bi = 'dec_b' + str(iter_i - 1)
		all_weights[dec_name_bi] = tf.Variable(tf.zeros([n_hidden[n_layers-iter_i-1]], dtype = tf.float32)) # , name = dec_name_bi
		iter_i = iter_i + 1
		
	dec_name_wi = 'dec_w' + str(iter_i - 1)
	all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[input_dim, n_hidden[0]],
		initializer=layers.xavier_initializer(),regularizer = reg)
	dec_name_bi = 'dec_b' + str(iter_i - 1)
	# all_weights[dec_name_bi] = tf.Variable(tf.zeros([1], dtype = tf.float32)) # TODO - why a single bias here?
	all_weights[dec_name_bi] = tf.Variable(tf.zeros([n_hidden[0]], dtype = tf.float32))
	return all_weights	


def self_expressive_module(encoder_out, batch_size):
	Coef = tf.Variable(1.0e-4 * tf.ones([batch_size, batch_size], tf.float32), name = 'Coef')			
	z_ssc = tf.matmul(Coef,	encoder_out)
	return z_ssc, Coef


######## MODEL CLASSES #########

# Basic AutoEncoder - for pretraining
class DSCAutoEncoder(object):
	def __init__(self, *, input_dim, n_hidden, learning_rate, batch_size, logs_path, save_dir, reg = None, denoise = False):	
		
		self.input_dim = input_dim 
		self.n_hidden = n_hidden # array containing the numbera of neurals for each layer
		self.reg = reg
		self.save_dir = save_dir
		self.iter = 0
		self.batch_size = batch_size
		weights = initialize_weights(self.input_dim, self.n_hidden, self.reg)
		self.learning_rate = tf.placeholder(tf.float32, [])

		# model input
		self.x = tf.placeholder(tf.float32, [None, self.input_dim])
		if denoise == False:
			x_input = self.x
		else:
			x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x), mean = 0, stddev = 0.2, dtype=tf.float32))
		
		# encoder
		self.enc_out, shape = encoder(x_input, weights, len(n_hidden))					
		self.dec_out = decoder(self.enc_out, weights, shape, len(n_hidden))
		
		self.saver = tf.train.Saver() 
		# cost for reconstruction
		# l_2 loss 
		self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.dec_out, self.x), 2.0))
		tf.summary.scalar("l2_loss", self.cost)          
		self.merged_summary_op = tf.summary.merge_all()        
        
		self.loss = self.cost
		self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss) #GradientDescentOptimizer #AdamOptimizer
		init = tf.global_variables_initializer()
		self.sess = tf.InteractiveSession()
		self.sess.run(init)
		self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

	def partial_fit(self, X): 
		cost, summary, _ = self.sess.run((self.cost, self.merged_summary_op, self.optimizer), feed_dict = {self.x: X})
		self.summary_writer.add_summary(summary, self.iter)
		self.iter = self.iter + 1
		return cost 

	def transform(self, X):
		return self.sess.run(self.enc_out, feed_dict = {self.x:X})

	def save_model(self, epoch):
		save_path = self.saver.save(self.sess, os.path.join(self.save_dir, 'checkpoint_ep{}.ckpt'.format(str(epoch).zfill(5))))
		print("model saved in file: %s" % save_path)

	def restore(self, restore_file):
		self.saver.restore(self.sess, restore_file)
		print ("model restored")


# MODEL CLASS - for post pretraining with self expressive layer
class DSCModelFull(object):
	def __init__(self, *, input_dim, n_hidden, logs_path, save_dir, reg_const1 = 1.0, reg_const2 = 1.0, \
				 reg = None, batch_size = 256, denoise = False):	
		
		self.input_dim = input_dim 
		self.n_hidden = n_hidden # array containing the numbera of neurals for each layer
		self.reg = reg
		self.save_dir = save_dir
		self.iter = 0
		self.batch_size = batch_size
		weights = initialize_weights(self.input_dim, self.n_hidden, self.reg)
		self.learning_rate = tf.placeholder(tf.float32, [])

		# model input
		self.x = tf.placeholder(tf.float32, [None, self.input_dim])
		if denoise == False:
			x_input = self.x
		else:
			x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x), mean = 0, stddev = 0.2, dtype=tf.float32))
		
		# encoder
		self.enc_out, shape = encoder(x_input, weights, len(n_hidden))
		# self expressive middle module
		self.z_ssc, self.Coef = self_expressive_module(self.enc_out, batch_size)						
		self.dec_out = decoder(self.z_ssc, weights, shape, len(n_hidden))
		
		self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))]) 
		
		self.cost_ssc = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.enc_out,self.z_ssc), 2))
		self.recon =  tf.reduce_sum(tf.pow(tf.subtract(self.dec_out, self.x), 2.0))
		self.reg_ssc = tf.reduce_sum(tf.pow(self.Coef,2))
		tf.summary.scalar("self_expressive_loss", self.cost_ssc)
		tf.summary.scalar("coefficient_lose", self.reg_ssc)			
		self.loss_ssc = self.cost_ssc * reg_const2 + reg_const1 * self.reg_ssc + self.recon

		self.merged_summary_op = tf.summary.merge_all()		
		self.optimizer_ssc = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss_ssc)
		self.init = tf.global_variables_initializer()
		self.sess = tf.InteractiveSession()
		self.sess.run(self.init)
		self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())


	def finetune_fit(self, X, lr):
		C,l1_cost, l2_cost, summary, _ = self.sess.run((self.Coef, self.reg_ssc, self.cost_ssc, self.merged_summary_op, self.optimizer_ssc), \
													feed_dict = {self.x: X, self.learning_rate: lr})
		self.summary_writer.add_summary(summary, self.iter)
		self.iter = self.iter + 1
		return C, l1_cost, l2_cost 

	def fit(self, X):
		return self.sess.run((self.Coef), feed_dict= { self.x: X })

	def transform(self, X):
		return self.sess.run(self.enc_out, feed_dict = {self.x:X})

	def save_model(self, epoch):
		save_path = self.saver.save(self.sess, os.path.join(self.save_dir, 'checkpoint_ep{}.ckpt'.format(str(epoch).zfill(5))))
		print("model saved in file: %s" % save_path)

	def restore(self, restore_file):
		print ("Restoring model from {}".format(restore_file))
		self.saver.restore(self.sess, restore_file)
		print ("model restored")


def best_map(L1,L2):
	#L1 should be the labels and L2 should be the clustering number we got

	Label1 = np.unique(L1)
	nClass1 = len(Label1)
	Label2 = np.unique(L2)
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1,nClass2)
	G = np.zeros((nClass,nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		ind_cla1 = ind_cla1.astype(float)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			ind_cla2 = ind_cla2.astype(float)
			G[i,j] = np.sum(ind_cla2 * ind_cla1)
	m = Munkres()
	# ipdb.set_trace()
	index = m.compute(- G.T)
	index = np.array(index)
	c = index[:,1]
	newL2 = np.zeros(L2.shape)
	for i in range(nClass2):
		if c[i] < nClass1: # only replace if there is a label to replace it with
			newL2[L2 == Label2[i]] = Label1[c[i]]
	return newL2

def thrC(C,ro):
	if ro < 1:
		N = C.shape[1]
		Cp = np.zeros((N,N))
		S = np.abs(np.sort(-np.abs(C),axis=0))
		Ind = np.argsort(-np.abs(C),axis=0)
		for i in range(N):
			cL1 = np.sum(S[:,i]).astype(float)
			stop = False
			csum = 0
			t = 0
			while(stop == False):
				csum = csum + S[t,i]
				if csum > ro*cL1:
					stop = True
					Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
				t = t + 1
	else:
		Cp = C

	return Cp

def post_proC(C, K, d, alpha):
	# C: coefficient matrix, K: number of clusters, d: dimension of each subspace
	C = 0.5*(C + C.T)
	r = d*K + 1	
	U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
	U = U[:,::-1] 
	S = np.sqrt(S[::-1])
	S = np.diag(S)
	U = U.dot(S)
	U = normalize(U, norm='l2', axis = 1)  
	Z = U.dot(U.T)
	Z = Z * (Z>0)
	L = np.abs(Z ** alpha)
	L = L/L.max()
	L = 0.5 * (L + L.T)	
	spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
	spectral.fit(L)
	grp = spectral.fit_predict(L) + 1 
	return grp, L

def err_rate(gt_s, s):
	c_x = best_map(gt_s,s)
	err_x = np.sum(gt_s[:] != c_x[:])
	missrate = err_x.astype(float) / (gt_s.shape[0])
	return missrate  



		
	
