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
import tensorflow as tf
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
from scipy import misc
import pandas as pd
from matplotlib.pyplot import specgram
import math
import matplotlib.image as mpimg

SF1 = os.listdir("/content/drive/App/VCC_Dataset/vcc2016_training/SF1")
for i in range(162) :
  SF1[i] = "/content/drive/App/VCC_Dataset/vcc2016_training/SF1/"+SF1[i]
SF2 = os.listdir("/content/drive/App/VCC_Dataset/vcc2016_training/SF2")
for i in range(162) :
  SF2[i] = "/content/drive/App/VCC_Dataset/vcc2016_training/SF2/"+SF2[i]
SF3 = os.listdir("/content/drive/App/VCC_Dataset/vcc2016_training/SF3")
for i in range(162) :
  SF3[i] = "/content/drive/App/VCC_Dataset/vcc2016_training/SF3/"+SF3[i]
SM1 = os.listdir("/content/drive/App/VCC_Dataset/vcc2016_training/SM1")
for i in range(162) :
  SM1[i] = "/content/drive/App/VCC_Dataset/vcc2016_training/SM1/"+SM1[i]
SM2 = os.listdir("/content/drive/App/VCC_Dataset/vcc2016_training/SM2")
for i in range(162) :
  SM2[i] = "/content/drive/App/VCC_Dataset/vcc2016_training/SM2/"+SM2[i]
TF1 = os.listdir("/content/drive/App/VCC_Dataset/vcc2016_training/TF1")
for i in range(162) :
  TF1[i] = "/content/drive/App/VCC_Dataset/vcc2016_training/TF1/"+TF1[i]
TF2 = os.listdir("/content/drive/App/VCC_Dataset/vcc2016_training/TF2")
for i in range(162) :
  TF2[i] = "/content/drive/App/VCC_Dataset/vcc2016_training/TF2/"+TF2[i]
TM1 = os.listdir("/content/drive/App/VCC_Dataset/vcc2016_training/TM1")
for i in range(162) :
  TM1[i] = "/content/drive/App/VCC_Dataset/vcc2016_training/TM1/"+TM1[i]
TM2 = os.listdir("/content/drive/App/VCC_Dataset/vcc2016_training/TM2")
for i in range(162) :
  TM2[i] = "/content/drive/App/VCC_Dataset/vcc2016_training/TM2/"+TM2[i]
TM3 = os.listdir("/content/drive/App/VCC_Dataset/vcc2016_training/TM3")
for i in range(162) :
  TM3[i] = "/content/drive/App/VCC_Dataset/vcc2016_training/TM3/"+TM3[i]
voice_files = SF1+SF2+SF3+SM1+SM2+TF1+TF2+TM1+TM2+TM3

labels = np.ndarray(len(voice_files))
voice_data = []
for i in range(len(voice_files)):
  labels[i] = i//162
  wav, _ = librosa.load(voice_files[i], mono=True, sr=16000)
  # get mfcc feature
  mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, 16000), axis=0), [0, 2, 1])
  mfcc = mfcc.reshape(-1,20)
  voice_data.append(mfcc)

minimum = 1000
maximum = 0
for i in range(1620):
  tem = voice_data[i].shape[1]
  if tem < minimum :
    minimum = tem
    minindex = i
  if tem > maximum :
    maximum = tem
    maxindex = i

from sklearn.preprocessing import OneHotEncoder
label_onehot_encoder = OneHotEncoder(sparse=False)
labels = labels.reshape(len(labels), 1)
label_hot = label_onehot_encoder.fit_transform(labels)

random_seed = 2
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(voice_data, labels, test_size = 0.1, random_state=random_seed)

epochs = 100
temp_zero = [0.0 for i in range(20)]
temp_zero = [temp_zero]

for i in range(len(X_train)):
  ini = len(X_train[i])
  for j in range(ini, 370):
    X_train[i] = np.append(X_train[i], temp_zero, axis=0)

for i in range(len(X_val)):
  ini = len(X_val[i])
  for j in range(ini, 370):
    X_val[i] = np.append(X_val[i], temp_zero, axis=0)

X_train = np.transpose(X_train, [1, 0, 2])
X_val = np.transpose(X_val, [1, 0, 2])

log_device_placement = True
np.random.seed(1)
batch_size = 10
n_steps = 370

input_dim = 20
hidden_dim = 100
output_dim = 10

with tf.variable_scope("SpeakerRecognition") :
    #  Sequences we will provide at runtime
    seq_input = tf.placeholder(tf.float32, [n_steps, None, input_dim])
    Y_batch = tf.placeholder(tf.int32, [None,1])

    #  What timestep we want to stop at
    #early_stop = tf.placeholder(tf.int32)
        
    initializer = tf.random_uniform_initializer(-1, 1)

    #  Inputs for rnn needs to be a list, each item being a timestep.
    #  we need to split our input into each timestep, and reshape it because
    #  split keeps dims by default
    inputs = [tf.reshape(i, (-1, input_dim)) for i in tf.split(seq_input, n_steps, axis = 0)]

    
    #tf.get_variable_scope().reuse_variables()
    
    cell1 = tf.contrib.rnn.LSTMCell(hidden_dim, input_dim, initializer=initializer)
#    initial_state1 = cell1.zero_state(None, tf.float32)
    outputs1, states1 = tf.contrib.rnn.static_rnn(cell1, inputs, dtype = tf.float32, scope="RNN1")#sequence_length=early_stop


    cell2 = tf.contrib.rnn.LSTMCell(hidden_dim, hidden_dim, initializer=initializer)
#    initial_state2 = cell2.zero_state(None, tf.float32)
    outputs2, states2 = tf.contrib.rnn.static_rnn(cell2, outputs1, dtype = tf.float32, scope="RNN2")#sequence_length=early_stop

    cell3 = tf.contrib.rnn.LSTMCell(hidden_dim, hidden_dim, initializer=initializer)
#    initial_state3 = cell3.zero_state(None, tf.float32)
    outputs3, states3 = tf.contrib.rnn.static_rnn(cell3, outputs2, dtype = tf.float32, scope="RNN3")#sequence_length=early_stop

    dense_out = tf.layers.dense(outputs3[-1], output_dim)
    soft_prob = tf.nn.softmax(dense_out)
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=Y_batch, logits=dense_out)
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    # Create initialize op, this needs to be run by the session!
    iop = tf.initialize_all_variables()    
    #tf.get_variable_scope().reuse_variables()

# Create session with device logging
session = tf.Session(
    config=tf.ConfigProto(log_device_placement=log_device_placement))

# Actually initialize, if you don't do this you get errors about uninitialized
session.run(iop)
session.run(tf.global_variables_initializer())

num_batches = X_train.shape[1]//batch_size - 1
for i in range(epochs) :
  
  for batch in range(num_batches):
    batch_input = X_train[:, batch*batch_size:batch*batch_size+batch_size, :]
    y_label = Y_train[batch*batch_size:batch*batch_size+batch_size, :]
#    batch_input = list(batch_input)
    feed = {seq_input: batch_input.astype('float32'), Y_batch: y_label}
    #print(Y_batch.shape)
    session.run(train_step, feed_dict=feed)
  print("epoch : "+str(i))
  y_pred = session.run(soft_prob, feed_dict={seq_input: X_val.astype('float32')})
  y_pred_classes = np.argmax(y_pred, axis = 1)
  count = 0
  for it in range(len(y_pred_classes)):
    if(y_pred_classes[it] == Y_val[it]):
        count+=1
  print("Accuracy : "+str(100*count/len(y_pred_classes)))
        
'''
    t1 = time.time()
    feed = {early_stop: 2,
            seq_input: np.random.rand(
                n_steps, batch_size, input_dim).astype('float32')}
    outs = session.run(outputs2, feed_dict=feed)
    t2 = time.time()
    print("Time for first call to session.run %f" % (t2 - t1))

    for e_s in [10, 50, 100, 150, 200, 250]:
        feed = {early_stop: e_s,
                seq_input: np.random.rand(
                    n_steps, batch_size, input_dim).astype('float32')}
        t1 = time.time()
        #  Early_stop can be varied, but seq_input needs to match the earlier shape
        outs = session.run(outputs2, feed_dict=feed)
        t2 = time.time()
        #  Output is a list, each item being a single timestep.
        #  Items at t>early_stop are all 0s
        print("Time for %i: %f" % (e_s, t2 - t1))'''