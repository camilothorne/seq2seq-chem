'''
Created on 10 June 2022
@author: Camilo Thorne

Create a character table for one-hot character encoding of training and test
sets from a large corpus -- this corpus should be large enough to cover *all* the characters C
and the *maximum* sentence length L of training and test corpora!

Train an encoder-decoder ses2seq model on an arbitrary training set with maximum sentence length < L
and characters included in C, and serialize it for testing

We will estimate P(p_1,...p_n|r_1,...,r_m) as the product (chain rule):

    P(p_n|p_{n-1},...,p_1;r_1,...,r_m) x ... x P(p_2|p_1;r_1,...,r_m)
    
The model will be used to estimate each term P(p_i|p_{i-1},..,p_1,r_1,...,r_m)

See also: https://lorenlugosch.github.io/posts/2019/02/seq2seq/

This training script will train and save three models, using checkpoints:

- encoder (last)
- encoder-decoder (best)
- decoder (last)

It will also compute and save some performance statistics

'''

from __future__ import print_function

# Ignore all warnings
import warnings
warnings.filterwarnings('ignore')

# Assign timestamp to results
import datetime as dt
ts = dt.datetime.now().strftime("-%m:%d:%Y::%H:%M:%S")

# Plotting
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Load Keras and TF
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate
import tensorflow as tf, os, codecs
import keras as ks

# Mute TF warnings
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Lead repo into to Python path
import sys
sys.path.insert(0,'../../seq2seq-chem')

from utils.onehotencode import OneHotEncode, ExpConfig

'''
Hyperparameters (1)
'''

# Get experimental parameters
my_exp = ExpConfig('bilstm','./config_bilstm_uspto50-sample.json')
my_exp.print_config()

batch_size  = my_exp.batch_size     # Batch size for training
epochs      = my_exp.epochs         # Number of epochs to train for
latent_dim  = my_exp.latent_dim     # Latent dimensionality of the encoding space
data_name   = my_exp.data_name      # Dataset name
typ         = my_exp.typ            # Type

'''
Callabacks, logs and tensorboard
'''

# Tensorboard
from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='../logs/log_' + 
                          typ + '-' + str(batch_size) + '-' + str(latent_dim) + '-' + 
                          data_name + ts, histogram_freq=0,
                          write_graph=True, write_images=True)
# Checkpoints
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('../dnns/best_ckp_encoder_decoder_' + 
                             typ + '-' + str(batch_size) + '-' + str(latent_dim) + '-' + 
                             data_name + ts + '.h5', 
                             save_weights_only=True, monitor='val_acc', 
                             save_best_only=True)

'''
Create character table(s) and one-hot encodings for whole dataset (train + test)
'''

# Path to the data file on disk
data_path  = '../data/' + data_name + '.tsv' 
data_encoding = OneHotEncode()                                           # Char table + one-hot vectors (full corpus)
data_encoding.build_char_table(data_path, xrows=['reactants','product']) # We want to generate products from reactants
data_encoding.corpus_stats()
data_encoding.to_pickle('../data/' + data_name + '.pk') 

'''
Hyperparameters (2)
'''

tr_samples  = len(data_encoding.corpus[
                    data_encoding.corpus.split=="train"])   # Number of samples to train on

'''
Get one-hot vectors for training (= slice one-hot vector matrix to training set for training)
'''

num_encoder_tokens     = data_encoding.num_encoder_tokens
num_decoder_tokens     = data_encoding.num_decoder_tokens

(encoder_input_data, 
    decoder_input_data, 
    decoder_target_data, _) = data_encoding.select_sample(0, 
                                                          tr_samples, 
                                                          print_stats=False)

print('Shape of encoder inputs  (train):    ', encoder_input_data.shape)
print('Shape of decoder inputs  (train):    ', decoder_input_data.shape)
print('Shape of decoder targets (train):    ', decoder_target_data.shape)
print()

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
# We discard encoder outputs and only keep the h and c states
_, forward_h, forward_c   = encoder_lstm1(encoder_inputs)
_, backward_h, backward_c = encoder_lstm2(encoder_inputs)
state_h                                  = Concatenate()([forward_h, backward_h])
state_c                                  = Concatenate()([forward_c, backward_c])
# We merge the states
encoder_states                           = [state_h, state_c]

'''
Define decoder (training), using `encoder_states` as initial state.
We set up our decoder to return full output sequences,
and to return internal states as well. We don't use the
return states in the training model, but we will use them in inference.
Note that the latent dimensions *double*.
'''

decoder_lstm              = LSTM(2*latent_dim, return_sequences=True, return_state=True)
# We keep `encoder_outputs` and discard the rest
decoder_outputs, _, _     = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense             = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs           = decoder_dense(decoder_outputs)

'''
Define and train the encoder-decoder model that will turn
`encoder_input_data` & `decoder_input_data` into `decoder_target_data`.
We will extract from it the models for testing/prediction.
'''

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# Training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[tensorboard, checkpoint])
print()
print('==> finished training model')

'''
Save results
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
    with open('../plots/' + filename + '.txt','w') as fi:
        fi.write(s)

# plot training & validation accuracy curves
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model accuracy (biLSTM)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='upper left')
plt.savefig('../plots/train_' + 
            typ + '-' + str(batch_size) + '-' + str(latent_dim) + '-training-acc-' + 
            data_name +ts + '.png', bbox_inches='tight')
plt.close()

# Plot training & validation loss curves
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model loss (biLSTM)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='upper left')
plt.savefig('../plots/train_' + 
            typ + '-' + str(batch_size) + '-' + str(latent_dim) + '-training-loss-' + 
            data_name +ts + '.png', bbox_inches='tight')
plt.close()

# Print model
summ = get_model_summary(model)
myprint(summ, 'encoder_decoder_' + 
        typ + '-' + str(batch_size) + 
        '-' + str(latent_dim) + '-' + data_name + ts)

# Plot model
ks.utils.plot_model(model, to_file='../plots/encoder_decoder_' +
                    typ + '-' + str(batch_size) + '-' + str(latent_dim) + 
                    '-' + data_name + ts + 
                    '.png', show_shapes=True)
print('==> saved training stats')

# Serialize encoder-decoder
with codecs.open('../dnns/encoder_decoder_' + 
                 typ + '-' + str(batch_size) + '-' + str(latent_dim) +
                 data_name + ts +'.json', 'w', encoding='utf8') as f:
    f.write(model.to_json())
model.save_weights('../dnns/encoder_decoder_weights_' +
                   typ + '-' + str(batch_size) + '-' + str(latent_dim) +
                   data_name + ts + '.h5')
model.save('../dnns/encoder_decoder_' + 
           typ + '-' + str(batch_size) + '-' + str(latent_dim) +
           data_name + ts + '.h5')

'''
Project encoder-decoder models from trained architecture and serialiaze
'''

# Encoder - note that the latent dimensions *double*.
encoder_model           = Model(encoder_inputs, encoder_states)
decoder_state_input_h   = Input(shape=(2*latent_dim,))

# Serialize encoder
with codecs.open('../dnns/encoder_' +
                 typ + '-' + str(batch_size) + '-' + str(latent_dim) +
                 data_name + ts + '.json', 'w', encoding='utf8') as f:
    f.write(encoder_model.to_json())
encoder_model.save_weights('../dnns/encoder_weights_' +
                           typ + '-' + str(batch_size) + '-' + str(latent_dim) +
                           data_name + ts +'.h5')
encoder_model.save('../dnns/encoder_' +
                   typ + '-' + str(batch_size) + '-' + str(latent_dim) +
                   data_name + ts + '.h5')

# Print model
summ_e = get_model_summary(encoder_model)
myprint(summ_e, 'encoder_' + 
        typ + '-' + str(batch_size) + 
        '-' + str(latent_dim) + '-' + data_name + ts)

# Plot model
ks.utils.plot_model(encoder_model, to_file='../plots/encoder_' +
                    typ + '-' + str(batch_size) + '-' + str(latent_dim) + 
                    '-' + data_name + ts + 
                    '.png', show_shapes=True)
print('==> saved encoder model to disk')

# Decoder - note that the latent dimensions *double*.
decoder_state_input_c               = Input(shape=(2*latent_dim,))
decoder_states_inputs               = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c   = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states                      = [state_h, state_c]
decoder_outputs                     = decoder_dense(decoder_outputs)
decoder_model                       = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Serialize decoder
with codecs.open('../dnns/decoder_' +
                 typ + '-' + str(batch_size) + '-' + str(latent_dim) +
                 data_name + ts + '.json', 'w', encoding='utf8') as f:
    f.write(decoder_model.to_json())
decoder_model.save_weights('../dnns/decoder_weights_' +
                           typ + '-' + str(batch_size) + '-' + str(latent_dim) +
                           data_name + ts + '.h5')
decoder_model.save('../dnns/decoder_' +
                   typ + '-' + str(batch_size) + '-' + str(latent_dim) +
                   data_name + ts + '.h5')

# Print model
summ_d = get_model_summary(decoder_model)
myprint(summ_d, 'decoder_' + 
        typ + '-' + str(batch_size) + 
        '-' + str(latent_dim) + '-' + data_name + ts)

# Plot model
ks.utils.plot_model(decoder_model, to_file='../plots/decoder_' 
                    + typ + '-' + str(batch_size) + '-' + str(latent_dim) + 
                    '-' + data_name + ts + 
                    '.png', show_shapes=True)
print('==> saved decoder model to disk')