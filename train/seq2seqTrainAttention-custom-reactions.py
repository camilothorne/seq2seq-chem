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

# Plotting / stats
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd

# Load Keras and TF
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate
from keras.layers import Permute, RepeatVector, Flatten, Activation, merge, Lambda
from keras.layers.wrappers import Bidirectional, TimeDistributed

import tensorflow as tf, os, codecs
import keras as ks
import keras.backend as K

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

# get model / experiment name and dataset name from CLI
eval_path = sys.argv[-1]

# display experiment/model trained
print('\nModel: ' + eval_path)

# Get experimental parameters
my_exp = ExpConfig('../dnns/' + eval_path + '.json')
my_exp.print_config()

batch_size  = my_exp.batch_size     # Batch size for training
epochs      = my_exp.epochs         # Number of epochs to train for
latent_dim  = my_exp.latent_dim     # Latent dimensionality of the encoding space
data_name   = my_exp.data_name      # Dataset name
typ         = my_exp.typ            # Type
expr        = typ + '-' + str(batch_size) + '-' + str(epochs) + '-' + str(latent_dim) + '_' + data_name + '_' # Path
ts          = dt.datetime.now().strftime("%m:%d:%Y::%H:%M:%S") # Timestamp

'''
Create plotting, logging and model directories        
'''   
  
#os.mkdir('../plots/' + expr + ts)
os.mkdir('../dnns/'  + expr + ts)
#os.mkdir('../logs/'  + expr + ts)

'''
Callabacks, logs and tensorboard
'''

# Tensorboard
from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='../logs/' + 
                          expr + ts, histogram_freq=0,
                          write_graph=True, write_images=True)
# Checkpoints
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('../dnns/' + 
                             expr + ts + '/best_ckp_encoder_decoder.h5', 
                             save_weights_only=True, monitor='val_acc', 
                             save_best_only=True)

'''
Create character table(s) and one-hot encodings for whole dataset (train + test)
'''

# Path to the data file on disk
data_path  = '../data/' + data_name + '.tsv' 
data_encoding = OneHotEncode()                                           # Char table + one-hot vectors (full corpus)
#data_encoding.build_char_table(data_path, xrows=['reactants','product'], norm_len=True) # We want to generate products from reactants
data_encoding.build_char_table(data_path, xrows=['product','reactants'], norm_len=True) # We want to generate reactants from products
data_encoding.corpus_stats()
data_encoding.to_pickle('../dnns/' + expr + ts + '/' + data_name + '.pk') 

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
seq_len                = max([data_encoding.max_encoder_seq_length,
                              data_encoding.max_decoder_seq_length])

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
encoder_inputs = Input(shape=(seq_len, num_encoder_tokens))
decoder_inputs = Input(shape=(seq_len, num_decoder_tokens))

'''
Define encoder (training)
'''

# Define encoder (training)
f_encoder, f_h, f_s            = LSTM(latent_dim, return_sequences=True, 
                                      return_state=True)(encoder_inputs)
b_encoder, b_h, b_s            = LSTM(latent_dim, return_sequences=True, 
                                      return_state=True, go_backwards=True)(encoder_inputs)
h                              = Concatenate()([f_h, b_h])
s                              = Concatenate()([f_s, b_s])
encoder                        = Concatenate()([f_encoder, b_encoder])
                                           
'''
Define decoder
'''

# Base decoder is a LSTM
initial_state = [h,s]
decoder ,_ ,_ = LSTM(2*latent_dim, return_sequences=True,
                     return_state=True)(decoder_inputs, initial_state=initial_state)

# Compute importance for each time step
attention = TimeDistributed(Dense(1, activation='tanh'))(encoder)
attention = Flatten()(attention)
attention = Activation('softmax')(attention)
attention = RepeatVector(2*latent_dim)(attention)
attention = Permute([2, 1])(attention)

# Merge branches via cross-attention
sent_representation       = merge([attention, decoder], mode='mul')
#sent_representation       = Lambda(lambda xin: K.sum(xin, axis=-2),output_shape=(,latent_dim,))(sent_representation)
sent_representation       = Dense(num_decoder_tokens, activation='softmax')(sent_representation)

# Define encoder-decoder
model                     = Model([encoder_inputs, decoder_inputs], sent_representation)

'''
Define branches
'''

encoder_model            = Model(inputs=encoder_inputs, outputs=encoder)

'''
Visualize summary
'''

def summary_to_table(model):
    table=pd.DataFrame(columns=["Name","Type","OutShape", "Params"])
    for layer in model.layers:
        table = table.append({"Name": layer.name, 
                              "Type": layer.__class__.__name__,
                              "OutShape": layer.output_shape,
                              "Params": layer.count_params()
                             }, 
                             ignore_index=True)
    return table

print("Encoder-decoder:\n")                    
df = summary_to_table(model)
print(df)    
ks.utils.plot_model(model, to_file='./encoder_decoder_attention.png', show_shapes=True)
print("\nEncoder:\n")                    
df = summary_to_table(encoder_model)
print(df) 
print()

'''
Train the encoder-decoder model that will turn
`encoder_input_data` & `decoder_input_data` into `decoder_target_data`.
We will extract from it the models for testing/prediction.
'''

# Training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[tensorboard, checkpoint])
print()
print('==> finished training model')

sys.exit()

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

# Plot training & validation accuracy curves
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model accuracy (biLSTM)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='upper left')
plt.savefig('../plots/' + 
            expr + ts + '/acc_train.png', bbox_inches='tight')
plt.close()

# Plot training & validation loss curves
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model loss (biLSTM)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='upper left')
plt.savefig('../plots/' + 
            expr + ts + '/loss_train.png', bbox_inches='tight')
plt.close()

# Print model
summ = get_model_summary(model)
myprint(summ, expr + ts + '/encoder_decoder_plot')

# Plot model
ks.utils.plot_model(model, to_file='../plots/' +
                    expr + ts + '/encoder_decoder.png', show_shapes=True)
print('==> saved training stats')

# Serialize encoder-decoder
with codecs.open('../dnns/' + 
                 expr + ts + '/encoder_decoder.json', 'w', encoding='utf8') as f:
    f.write(model.to_json())
model.save_weights('../dnns/' + 
                   expr + ts + '/encoder_decoder_weights.h5')
model.save('../dnns/' + 
           expr + ts + '/encoder_decoder.h5')

'''
Project encoder-decoder models from trained architecture and serialiaze
'''

# Encoder - note that the latent dimensions *double*.
encoder_model           = Model(encoder_inputs, encoder_states)
decoder_state_input_h   = Input(shape=(2*latent_dim,))

# Serialize encoder
with codecs.open('../dnns/' +
                 expr + ts + '/encoder.json', 'w', encoding='utf8') as f:
    f.write(encoder_model.to_json())
encoder_model.save_weights('../dnns/' +
                           expr + ts + '/encoder_weights.h5')
encoder_model.save('../dnns/' +
                   expr + ts + '/encoder.h5')

# Print model
summ_e = get_model_summary(encoder_model)
myprint(summ_e, expr + ts + '/encoder_plot')

# Plot model
ks.utils.plot_model(encoder_model, to_file='../plots/' +
                    expr + ts + '/encoder.png', show_shapes=True)
print('==> saved encoder model to disk')

# Decoder - note that the latent dimensions *double*.
decoder_state_input_c               = Input(shape=(2*latent_dim,))
decoder_states_inputs               = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c   = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states                      = [state_h, state_c]
decoder_outputs                     = decoder_dense(decoder_outputs)
decoder_model                       = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Serialize decoder
with codecs.open('../dnns/' +
                 expr + ts + '/decoder.json', 'w', encoding='utf8') as f:
    f.write(decoder_model.to_json())
decoder_model.save_weights('../dnns/' +
                           expr + ts + '/decoder_weights.h5')
decoder_model.save('../dnns/' +
                   expr + ts + '/decoder.h5')

# Print model
summ_d = get_model_summary(decoder_model)
myprint(summ_d, expr + ts + '/decoder_plot')

# Plot model
ks.utils.plot_model(decoder_model, to_file='../plots/' +
                    expr + ts + '/decoder.png', show_shapes=True)
print('==> saved decoder model to disk')