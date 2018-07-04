import tensorflow as tf
import numpy as np
import time
import scipy.io.wavfile as wav
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
from skimage.util.shape import view_as_windows
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
import librosa
import librosa.display
from scipy import misc
import pandas as pd
from matplotlib.pyplot import specgram
import math
import matplotlib.image as mpimg
import os
import scipy.io.wavfile as wav
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
from skimage.util.shape import view_as_windows
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
import tensorflow as tf
import librosa
import librosa.display
from scipy import misc,stats
import pandas as pd
from matplotlib.pyplot import specgram
import math
import matplotlib.image as mpimg
import sugartensor as tf
import numpy as np
import librosa
from model import *
import data
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

conversion_number = 100001
source_destination = "/content/drive/App/VCC_Dataset/vcc2016_training/SF1/"+str(conversion_number)+".wav"
target_destination = "/content/drive/App/VCC_Dataset/vcc2016_training/TF1/"+str(conversion_number)+".wav"
voice_files = [source_destination,target_destination]
voice_data = []
for i in range(len(voice_files)):
	wav, _ = librosa.load(voice_files[i], mono=True, sr=16000)
	mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, 16000), axis=0), [0, 2, 1])
	mfcc = mfcc.reshape(-1,20)
	new_len = (20-mfcc.shape[0]%20)%20
	mfcc = np.vstack((mfcc,np.zeros((new_len, 20))))                                         # n_steps = 20
	voice_data.append(mfcc)
#print(voice_data)

ALPHA = 0
BETA = 1-ALPHA
np.random.seed(1)
batch_size = 1
n_steps = 20
epochs = 30
input_dim = 20
hidden_dim = 100
output_dim = 10

content_size = (voice_data[0]).shape[0] // n_steps
style_size = (voice_data[1]).shape[0] // n_steps

#g1 = tf.Graph()
#with g1.as_default(), tf.Session() as session:
seq_input = tf.placeholder(tf.float32, [None, input_dim])

with tf.variable_scope("SpeakerRecognition") :
	#  Sequences we will provide at runtime

	#Y_batch = tf.placeholder(tf.float32, [output_dim,])
	pseudo_batch_size = tf.placeholder(tf.int32)
	#  What timestep we want to stop at
	#early_stop = tf.placeholder(tf.int32)
	# seq_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(x,axis=2), 0.), tf.int32),axis=1)
	initializer = tf.contrib.layers.xavier_initializer()

	seq_input_1 = tf.transpose(tf.reshape(seq_input, (-1,n_steps,input_dim)), perm = [1,0,2])

	#  Inputs for rnn needs to be a list, each item being a timestep.
	#  we need to split our input into each timestep, and reshape it because
	#  split keeps dims by default
	inputs = [tf.reshape(i, (-1, input_dim)) for i in tf.split(seq_input_1, n_steps, axis = 0)]    
	#Y_batch_1 = tf.reshape(tf.tile(Y_batch, tf.reshape(pseudo_batch_size, (1,))), (-1,output_dim))

	#tf.get_variable_scope().reuse_variables()

	cell1 = tf.contrib.rnn.LSTMCell(hidden_dim, input_dim, initializer=initializer)
	cell1 = tf.contrib.rnn.DropoutWrapper(cell1, output_keep_prob=0.25)
	initial_state1 = cell1.zero_state(pseudo_batch_size, tf.float32)
	outputs1, states1 = tf.contrib.rnn.static_rnn(cell1, inputs, initial_state=initial_state1, scope="RNN1")#sequence_length=early_stop


	cell2 = tf.contrib.rnn.LSTMCell(hidden_dim, hidden_dim, initializer=initializer)
	cell2 = tf.contrib.rnn.DropoutWrapper(cell2, output_keep_prob=0.25)
	initial_state2 = cell2.zero_state(pseudo_batch_size, tf.float32)
	outputs2, states2 = tf.contrib.rnn.static_rnn(cell2, outputs1, initial_state=initial_state2, scope="RNN2")#sequence_length=early_stop

	cell3 = tf.contrib.rnn.LSTMCell(hidden_dim, hidden_dim, initializer=initializer)
	cell3 = tf.contrib.rnn.DropoutWrapper(cell3, output_keep_prob=0.25)
	initial_state3 = cell3.zero_state(pseudo_batch_size, tf.float32)
	outputs3, states3 = tf.contrib.rnn.static_rnn(cell3, outputs2, initial_state=initial_state3,scope="RNN3")#sequence_length=early_stop

	dense_out = tf.layers.dense(outputs3[-1], output_dim)
	soft_prob = tf.nn.softmax(dense_out)
	#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_batch_1, logits=dense_out))
	#train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

	# Create initialize op, this needs to be run by the session!
	iop = tf.initialize_all_variables()
	#tf.get_variable_scope().reuse_variables()

############################################################################################
### Wavenet

#tf.sg_verbosity(10)

voca_size = data.voca_size

# mfcc feature of audio
x = tf.reshape(seq_input, (1,-1,input_dim))
# sequence length except zero-padding
seq_len = tf.not_equal(x.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1)

logit = get_logit(x, voca_size=voca_size)

# ctc decoding
decoded, _ = tf.nn.ctc_beam_search_decoder(logit.sg_transpose(perm=[1, 0, 2]), seq_len, merge_repeated=False)

# to dense tensor
y = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values) + 1



############################################################################################

# Actually initialize, if you don't do this you get errors about uninitialized
session = tf.Session()
session.run(iop)
tf.sg_init(session)

all_vars = tf.global_variables()
model_one_vars = [k for k in all_vars if k.name.startswith("SpeakerRecognition")]
set_1 = set(model_one_vars)
model_two_vars = [o for o in all_vars if o not in set_1]

tf.train.Saver(model_two_vars).restore(session, tf.train.latest_checkpoint('asset/train'))
tf.train.Saver(model_one_vars).save(session, "asset/LSTM_train/model.ckpt")

content_Wavenet_out = session.run(decoded, feed_dict={seq_input: voice_data[0]})
style_y_pred = session.run(soft_prob, feed_dict={seq_input: voice_data[1] , pseudo_batch_size: (voice_data[1]).shape[0] // n_steps})

#print(content_Wavenet_out)
#print(style_y_pred)
#Wavenet_out, y_pred = session.run(decoded,soft_prob, feed_dict={seq_input: mix_input.astype('float32'), pseudo_batch_size: mix_input.shape[0] // n_steps})

with tf.variable_scope("StyleTransfer") :
	up_seq_input = tf.get_variable(name = "input_seq", dtype = tf.float32, shape = voice_data[0].shape, initializer = tf.contrib.layers.xavier_initializer())
	#up_seq_input = tf.Variable(voice_data[0].astype(np.float32), name="input_seq")

	with tf.variable_scope("SpeakerRecognition") :
		#  Sequences we will provide at runtime

		#Y_batch = tf.placeholder(tf.float32, [output_dim,])
		up_pseudo_batch_size = tf.constant(name = "input_batch", dtype =tf.int32, value = (voice_data[0]).shape[0] // n_steps)
		#  What timestep we want to stop at
		#early_stop = tf.placeholder(tf.int32)
		# seq_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(x,axis=2), 0.), tf.int32),axis=1)
		initializer = tf.contrib.layers.xavier_initializer()

		up_seq_input_1 = tf.transpose(tf.reshape(up_seq_input, (-1,n_steps,input_dim)), perm = [1,0,2])

		#  Inputs for rnn needs to be a list, each item being a timestep.
		#  we need to split our input into each timestep, and reshape it because
		#  split keeps dims by default
		up_inputs = [tf.reshape(i, (-1, input_dim)) for i in tf.split(up_seq_input_1, n_steps, axis = 0)]    
		#Y_batch_1 = tf.reshape(tf.tile(Y_batch, tf.reshape(pseudo_batch_size, (1,))), (-1,output_dim))

		#tf.get_variable_scope().reuse_variables()

		up_cell1 = tf.contrib.rnn.LSTMCell(hidden_dim, input_dim, initializer=initializer)
		up_cell1 = tf.contrib.rnn.DropoutWrapper(up_cell1, output_keep_prob=0.25)
		up_initial_state1 = up_cell1.zero_state(up_pseudo_batch_size, tf.float32)
		up_outputs1, up_states1 = tf.contrib.rnn.static_rnn(up_cell1, up_inputs, initial_state=up_initial_state1, scope="RNN1")#sequence_length=early_stop


		up_cell2 = tf.contrib.rnn.LSTMCell(hidden_dim, hidden_dim, initializer=initializer)
		up_cell2 = tf.contrib.rnn.DropoutWrapper(up_cell2, output_keep_prob=0.25)
		up_initial_state2 = up_cell2.zero_state(up_pseudo_batch_size, tf.float32)
		up_outputs2, up_states2 = tf.contrib.rnn.static_rnn(up_cell2, up_outputs1, initial_state=up_initial_state2, scope="RNN2")#sequence_length=early_stop

		up_cell3 = tf.contrib.rnn.LSTMCell(hidden_dim, hidden_dim, initializer=initializer)
		up_cell3 = tf.contrib.rnn.DropoutWrapper(up_cell3, output_keep_prob=0.25)
		up_initial_state3 = up_cell3.zero_state(up_pseudo_batch_size, tf.float32)
		up_outputs3, up_states3 = tf.contrib.rnn.static_rnn(up_cell3, up_outputs2, initial_state=up_initial_state3,scope="RNN3")#sequence_length=early_stop

		up_dense_out = tf.layers.dense(up_outputs3[-1], output_dim)
		up_soft_prob = tf.nn.softmax(up_dense_out)
		#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_batch_1, logits=dense_out))
		#train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

		# Create initialize op, this needs to be run by the session!
		iop = tf.initialize_all_variables()
		#tf.get_variable_scope().reuse_variables()

	############################################################################################
	### Wavenet

	#tf.sg_verbosity(10)

	up_voca_size = data.voca_size

	# mfcc feature of audio
	up_x = tf.reshape(up_seq_input, (1,-1,input_dim))

	# sequence length except zero-padding
	up_seq_len = tf.not_equal(up_x.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1)

	up_logit = get_logit(up_x, voca_size=up_voca_size)

	# ctc decoding
	up_decoded, _ = tf.nn.ctc_beam_search_decoder(up_logit.sg_transpose(perm=[1, 0, 2]), up_seq_len, merge_repeated=False)

	# to dense tensor
	up_y = tf.sparse_to_dense(up_decoded[0].indices, up_decoded[0].dense_shape, up_decoded[0].values) + 1


	content_loss = tf.norm(tf.cast(up_decoded[0].values - content_Wavenet_out[0].values, dtype = tf.float32))

	style_loss = 0
	if content_size > style_size :
		step_size = int(round(content_size/style_size))
		for i in range(style_size-1):
			content_prob = tf.reduce_mean(up_soft_prob[i*step_size:(i+1)*step_size],axis = 0)
			style_loss += tf.norm(content_prob - style_y_pred[i])
		content_prob = tf.reduce_mean(up_soft_prob[(style_size-1)*step_size:],axis = 0)
		style_loss += tf.norm(content_prob - style_y_pred[style_size-1])
	else :
		step_size = int(round(style_size/content_size))
		for i in range(content_size-1):
			style_prob = np.mean(style_y_pred[i*step_size:(i+1)*step_size],axis = 0)
			style_loss += tf.norm(up_soft_prob[i] - style_prob)
		style_prob = np.mean(style_y_pred[(content_size-1)*step_size:],axis = 0)
		style_loss += tf.norm(up_soft_prob[content_size-1] - style_prob)

	loss = ALPHA*content_loss + BETA*style_loss
	opt = tf.train.AdamOptimizer(0.001).minimize(loss, var_list = [up_seq_input])
  ############################################################################################

# Actually initialize, if you don't do this you get errors about uninitialized
#session = tf.Session()
session.run(iop)
tf.sg_init(session)

all_vars = tf.global_variables()
update_vars = [k for k in all_vars if k.name.startswith("StyleTransfer")]
update_model_one_vars = [k for k in update_vars if k.name.startswith("StyleTransfer/SpeakerRecognition")]
update_set_1 = set(update_model_one_vars)
update_model_two_vars = [o for o in update_vars if o not in update_set_1]

update_model_two_vars.pop(0)            # pop input_seq tensor variable


constant_vars = [o for o in all_vars if o not in update_vars]
constant_model_one_vars = [k for k in constant_vars if k.name.startswith("SpeakerRecognition")]
constant_set_1 = set(constant_model_one_vars)
constant_model_two_vars = [o for o in constant_vars if o not in constant_set_1]

constant_model_two_vars.pop(0)          # pop global_step variable


#tf.train.Saver(constant_model_two_vars).restore(session, tf.train.latest_checkpoint('asset/train'))
#tf.train.Saver(constant_model_one_vars).restore(session, "asset/LSTM_train/model.ckpt")

#for i in constant_model_one_vars:
#	print(i.name)

#content_Wavenet_out = session.run(decoded, feed_dict={seq_input: voice_data[0]})
#style_y_pred = session.run(soft_prob, feed_dict={seq_input: voice_data[1] , pseudo_batch_size: (voice_data[1]).shape[0] // n_steps})

cons_one_name = []
cons_two_name = []
for i in constant_model_one_vars :
	cons_one_name.append(i.name[:-2])
for i in constant_model_two_vars :
	cons_two_name.append(i.name[:-2])

dict_one_vars = dict(zip(cons_one_name, update_model_one_vars))
dict_two_vars = dict(zip(cons_two_name, update_model_two_vars))

#print_tensors_in_checkpoint_file(file_name="asset/LSTM_train/model.ckpt", tensor_name='', all_tensors=False)

tf.train.Saver(var_list = dict_one_vars).restore(session, "asset/LSTM_train/model.ckpt")
tf.train.Saver(var_list = dict_two_vars).restore(session, tf.train.latest_checkpoint('asset/train'))

Wavenet_out, y_pred = session.run([up_decoded,up_soft_prob])
c_loss,s_loss = session.run([content_loss, style_loss])

'''print("Content speech to text")
print(content_Wavenet_out[0].dense_shape)
print(content_Wavenet_out[0].values)
print("Random speech to text")
print(Wavenet_out[0].dense_shape)
print(Wavenet_out[0].values)
print("Style SpeakerRecognition")
print(style_y_pred.shape)
print(style_y_pred)
print("Random SpeakerRecognition")
print(y_pred.shape)
print(y_pred)
print("Content Loss")
print(c_loss)
print("Style Loss")
print(s_loss)'''

for i in range(10000):
	_, c_loss, s_loss, tot_loss = session.run([opt, content_loss, style_loss, loss])
	if(i%10==0):
		print("Interation : "+str(i) + " - Loss : " + str(tot_loss) + " - Content Loss : " + str(c_loss) + " - Style Loss : " + str(s_loss))

transfered_input = session.run(up_seq_input)
np.save("/content/drive/App/Wavenet_Style_Transfer/asset", transfered_input)