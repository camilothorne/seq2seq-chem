'''
Created on 10 May 2022
@author: camilo thorne

Create a character table for one-hot character encoding of training and test
sets from a large corpus. This corpus should be large enough to cover *all* the characters C
and the *maximum* sentence length L of training and test corpora.

Train an encoder-decoder ses2seq model on an arbitrary training set with maximum sentence length < L
and characters included in C, and serialize it for testing
'''

from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')  # ignore all warnings

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate
import numpy as np
import tensorflow as tf, os, codecs
import keras as ks

from keras.callbacks import TensorBoard
# tensorboard
tensorboard = TensorBoard(log_dir='../logs/bilstm', histogram_freq=10,
                          write_graph=True, write_images=True, write_grads=True)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# mute warnings
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
Hyperparameters
'''

batch_size  = 64        # Batch size for training.
epochs      = 100       # Number of epochs to train for.
latent_dim  = 128       # Latent dimensionality of the encoding space.
num_samples = 10000     # Max number of samples to train on.

'''
Encode data into one-hot character vectors
'''

# Path to the data file on disk.
train_path          = '../data/train-smi2smi.tsv'
data_path           = '../data/all-smi2smi.tsv'     # for char table

'''
Create character table(s)
'''

# Vectorize the data.
input_texts         = []
target_texts        = []
input_characters    = set()
target_characters   = set()

with codecs.open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
    
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    '''
    We use "tab" as the "start sequence" character
    for the targets, and "\n" as "end sequence" character.
    '''
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print()
print('Number of unique input tokens/chars: ', num_encoder_tokens)
print('Number of unique output tokens/chars:', num_decoder_tokens)
print('Max sequence length for inputs:      ', max_encoder_seq_length)
print('Max sequence length for outputs:     ', max_decoder_seq_length)
print()

input_token_index  = dict(
    [(char, i) for i, char in enumerate(input_characters)])

target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

'''
Load train set.
We encode sources and targets using the character table defined
above.
'''

train_input          = []
train_target         = []  

# open file
with codecs.open(train_path, 'r', encoding='utf-8') as f:
    lines_t = f.read().split('\n')

# load (source, target) pairs
for line in lines_t[: min(num_samples, len(lines_t) - 1)]:
    tinput_text, ttarget_text = line.split('\t')
    '''
    We use "tab" as the "start sequence" character
    for the targets, and "\n" as "end sequence" character.
    '''
    ttarget_text = '\t' + ttarget_text + '\n' 
    train_input.append(tinput_text)
    train_target.append(ttarget_text)   

print('Number of training samples: %s\n' %len(train_input)) 

'''
We work with one-hot char/token vectors. Note that input data dimensions 
depend on the maximum possible theoretical inputs lengths in both the 
target and the source data and on the maximum possible number of distinct characters (256). 
'''

encoder_input_data  = np.zeros((len(train_input), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data  = np.zeros((len(train_input), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(train_input), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

# We populate one-hot vectors.
for i, (tinput_text, ttarget_text) in enumerate(zip(train_input, train_target)):
    # loop on source
    for t, char in enumerate(tinput_text):
        #print(i, t, input_token_index[char], len(tinput_text))
        encoder_input_data[i, t, input_token_index[char]] = 1.    
    # loop on target
    for t, char in enumerate(ttarget_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

# Define input encoder/decoder sequences
encoder_inputs = Input(shape=(None, num_encoder_tokens))
decoder_inputs = Input(shape=(None, num_decoder_tokens))

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
We will extract from it the models for testing/prediction
'''

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[tensorboard])
print("==> finished training model\n")

# plot training & validation accuracy curves
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model accuracy (biLSTM)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='upper left')
plt.savefig('../plots/bilstm-100-128-training-loss.png', bbox_inches='tight')
plt.close()

# plot training & validation loss curves
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model loss (biLSTM)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='upper left')
plt.savefig('../plots/bilstm-100-128-training-acc.png', bbox_inches='tight')
plt.close()

# print model
model.summary()

# plot model
ks.utils.plot_model(model, to_file='../plots/encoder-decoder-bilstm-100-128.png', show_shapes=True)
print("==> saved training stats\n")

# serialize encoder-decoder
with codecs.open('../dnns/encoder_decoder_bilstm-100-128.json', 'w', encoding='utf8') as f:
    f.write(model.to_json())
model.save_weights('../dnns/encoder_decoder_bilstm_weights-100-128.h5')
model.save("../dnns/encoder_decoder-bilstm-100-128.h5")

'''
Project encoder-decoder models from trained architecture and serialiaze
'''

# encoder - note that the latent dimensions *double*.
encoder_model           = Model(encoder_inputs, encoder_states)
decoder_state_input_h   = Input(shape=(2*latent_dim,))

# serialize encoder
with codecs.open('../dnns/encoder_bilstm-100-128.json', 'w', encoding='utf8') as f:
    f.write(encoder_model.to_json())
encoder_model.save_weights('../dnns/encoder_bilstm_weights-100-128.h5')
encoder_model.save("../dnns/enc-bilstm-100-128.h5")

# print model
encoder_model.summary()

# plot model
ks.utils.plot_model(encoder_model, to_file='../plots/encoder-bilstm-100-128.png', show_shapes=True)
print("==> saved encoder model to disk\n")

# decoder - note that the latent dimensions *double*.
decoder_state_input_c               = Input(shape=(2*latent_dim,))
decoder_states_inputs               = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c   = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states                      = [state_h, state_c]
decoder_outputs                     = decoder_dense(decoder_outputs)
decoder_model                       = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# serialize decoder
with codecs.open('../dnns/decoder_bilstm-100-128.json', 'w', encoding='utf8') as f:
    f.write(decoder_model.to_json())
decoder_model.save_weights('../dnns/decoder_bilstm_weights-100-128.h5')
decoder_model.save("../dnns/dec-bilstm-100-128.h5")

# print model
decoder_model.summary()

# plot model
ks.utils.plot_model(decoder_model, to_file='../plots/decoder-bilstm-100-128.png', show_shapes=True)
print("==> saved decoder model to disk")