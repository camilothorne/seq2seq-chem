'''
Created on 10 June 2022
@author: Camilo Thorne

Create a character table for one-hot character encoding of training and test
sets from a large corpus. This corpus should be large enough to cover *all* the characters C
and the *maximum* sentence length L of training and test corpora.

Train an encoder-decoder ses2seq model on an arbitrary training set with maximum sentence length < L
and characters included in C, and serialize it for testing
'''

from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')  # ignore all warnings

import sys
sys.path.insert(0,'/home/jovyan/workbench-shared-folder/retro-syn/seq2seq-chem')

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate
#import numpy as np
from utils.onehotencode import OneHotEncode
import tensorflow as tf, os, codecs
import keras as ks

# tensorboard
from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='../logs/bilstm-uspto-50k', histogram_freq=0,
                          write_graph=True, write_images=True)
# checkpoints
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('../dnns/best_ckp_encoder_decoder-bilstm-uspto-50k.h5', 
                             save_weights_only=True, monitor='val_acc', 
                             save_best_only=True)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# mute warnings
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
Hyperparameters
'''

batch_size  = 64       # Batch size for training.
epochs      = 100      # Number of epochs to train for.
latent_dim  = 128      # Latent dimensionality of the encoding space.
num_samples = 40000    # Max number of samples to train on.

'''
Create character table(s)
'''

# Path to the data file on disk.
data_path  = '../data/training-s2s-uspto-50k.tsv'     # for char table
data_encoding = OneHotEncode()
data_encoding.build_char_table(data_path, xrows=['reactants','product'])
data_encoding.corpus_stats()

num_encoder_tokens                                               = data_encoding.num_encoder_tokens
num_decoder_tokens                                               = data_encoding.num_decoder_tokens
(encoder_input_data, decoder_input_data, decoder_target_data, _) = data_encoding.select_sample(0, num_samples)

'''
Define encoder/decoder inputs
'''

# Define input encoder/decoder sequences
encoder_inputs = Input(shape=(None, num_encoder_tokens))
decoder_inputs = Input(shape=(None, num_decoder_tokens))

'''
Define encoder (training)
'''

# Define encoder (training)
encoder_lstm1                            = LSTM(latent_dim, return_state=True)
encoder_lstm2                            = LSTM(latent_dim, return_state=True, go_backwards=True)
# We discard encoder outputs and only keep the h and c states.
_, forward_h, forward_c   = encoder_lstm1(encoder_inputs)
_, backward_h, backward_c = encoder_lstm2(encoder_inputs)
state_h                                  = Concatenate()([forward_h, backward_h])
state_c                                  = Concatenate()([forward_c, backward_c])
# We merge the states.
encoder_states                           = [state_h, state_c]

'''
Define decoder (training), using `encoder_states` as initial state.
We set up our decoder to return full output sequences,
and to return internal states as well. We don't use the
return states in the training model, but we will use them in inference.
Note that the latent dimensions *double*.
'''

decoder_lstm              = LSTM(2*latent_dim, return_sequences=True, return_state=True)
# We keep `encoder_outputs` and discard the rest.
decoder_outputs, _, _     = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense             = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs           = decoder_dense(decoder_outputs)

'''
Define and train the encoder-decoder model that will turn
`encoder_input_data` & `decoder_input_data` into `decoder_target_data`.
We will extract from it the models for testing/prediction.
'''

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[tensorboard, checkpoint])
print()
print("==> finished training model\n")

'''
Save results.
'''

def get_model_summary(model):
    '''
    convert summary to string
    '''
    string_list = []
    model.summary(line_length=80, print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)

def myprint(s, filename):
    '''
    save model summary to file
    '''
    with open('../plots/'+filename+'.txt','w') as fi:
        fi.write(s)

# plot training & validation accuracy curves
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model accuracy (biLSTM)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='upper left')
plt.savefig('../plots/bilstm-100-128-training-acc-uspto-50k.png', bbox_inches='tight')
plt.close()

# plot training & validation loss curves
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model loss (biLSTM)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='upper left')
plt.savefig('../plots/bilstm-100-128-training-loss-uspto-50k.png', bbox_inches='tight')
plt.close()

# print model
summ = get_model_summary(model)
myprint(summ, "encoder_decoder-bilstm-100-128")

# plot model
ks.utils.plot_model(model, to_file='../plots/encoder-decoder-bilstm-100-128-uspto-50k.png', show_shapes=True)
print("==> saved training stats\n")

# serialize encoder-decoder
with codecs.open('../dnns/encoder_decoder_bilstm-100-128-uspto-50k.json', 'w', encoding='utf8') as f:
    f.write(model.to_json())
model.save_weights('../dnns/encoder_decoder_bilstm_weights-100-128-uspto-50k.h5')
model.save("../dnns/encoder_decoder-bilstm-100-128-uspto-50k.h5")

'''
Project encoder-decoder models from trained architecture and serialiaze
'''

# encoder - note that the latent dimensions *double*.
encoder_model           = Model(encoder_inputs, encoder_states)
decoder_state_input_h   = Input(shape=(2*latent_dim,))

# serialize encoder
with codecs.open('../dnns/encoder_bilstm-100-128-uspto-50k.json', 'w', encoding='utf8') as f:
    f.write(encoder_model.to_json())
encoder_model.save_weights('../dnns/encoder_bilstm_weights-100-128-uspto-50k.h5')
encoder_model.save("../dnns/enc-bilstm-100-128-uspto-50k.h5")

# print model
summ_e = get_model_summary(encoder_model)
myprint(summ_e, "encoder-bilstm-100-128-uspto-50k")

# plot model
ks.utils.plot_model(encoder_model, to_file='../plots/encoder-bilstm-100-128-uspto-50k.png', show_shapes=True)
print("==> saved encoder model to disk\n")

# decoder - note that the latent dimensions *double*.
decoder_state_input_c               = Input(shape=(2*latent_dim,))
decoder_states_inputs               = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c   = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states                      = [state_h, state_c]
decoder_outputs                     = decoder_dense(decoder_outputs)
decoder_model                       = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# serialize decoder
with codecs.open('../dnns/decoder_bilstm-100-128-uspto-50k.json', 'w', encoding='utf8') as f:
    f.write(decoder_model.to_json())
decoder_model.save_weights('../dnns/decoder_bilstm_weights-100-128-uspto-50k.h5')
decoder_model.save("../dnns/dec-bilstm-100-128.h5")

# print model
summ_d = get_model_summary(decoder_model)
myprint(summ_d, "decoder-bilstm-100-128-uspto-50k")

# plot model
ks.utils.plot_model(decoder_model, to_file='../plots/decoder-bilstm-100-128-uspto-50k.png', show_shapes=True)
print("==> saved decoder model to disk")