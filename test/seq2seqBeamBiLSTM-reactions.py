'''
Created on 10 June 2022
@author: Camilo Thorne

Restore a character-level sequence to sequence model from disk and use it
to generate predictions.

This script loads the model saved and generates
sequences from it.  It assumes that no changes have been made (for example:
latent_dim is unchanged, and the input data and model architecture are unchanged).

It relies on "beam" search.
'''

from __future__ import print_function

import sys
sys.path.insert(0,'../../seq2seq-chem')

from utils.onehotencode import OneHotEncode, ExpConfig
from utils.beamsearch import State, BeamSearch

import warnings
warnings.filterwarnings('ignore')  # ignore all warnings

import numpy as np
import pandas as pd
from keras.models import model_from_json
import tensorflow as tf
import os, sys
from collections import OrderedDict
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu

# mute TF warnings
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
Get experiment path from CLI
'''

# get model / experiment name and dataset name from CLI
eval_path = sys.argv[-1]
data_name = eval_path.split('_')[1]

# display experiment/model trained
print('\nModel: ' + eval_path)

'''
Retrieve corpus and one-hot encodings:
'''

data_encoding = OneHotEncode()
data_encoding.from_pickle('../dnns/' + eval_path + '/' + data_name + '.pk')
data_encoding.corpus_stats()
df = data_encoding.corpus

input_characters          = data_encoding.input_characters
target_characters         = data_encoding.target_characters
input_token_index         = data_encoding.input_token_index
target_token_index        = data_encoding.target_token_index 
num_encoder_tokens        = data_encoding.num_encoder_tokens
num_decoder_tokens        = data_encoding.num_decoder_tokens
max_encoder_seq_length    = data_encoding.max_encoder_seq_length
max_decoder_seq_length    = data_encoding.max_decoder_seq_length

# hyperparameters
train_samples = len(df[df.split=='train'])             # number of samples to test on
num_samples   =  len(df) - len(df[df.split=='train'])  # number of samples to test on

# get the test partition
(encoder_input_data, 
    decoder_input_data, 
    decoder_target_data, 
    corpus) = data_encoding.select_sample(train_samples, 
                                          train_samples+num_samples, 
                                          print_stats=True)

'''
Build test corpus
'''

test_input          = []
test_target         = []
for i in range(0, corpus.shape[0]):
    test_input.append(corpus[i,0])
    test_target.append(corpus[i,1]) 

'''
Create alternatives for BLEU-4
'''

test_dict = {}    
for i in range(0,len(test_input)):
    '''
    create dict of alternative targets for every source sentence
    '''
    if test_input[i] in test_dict:
        test_dict[test_input[i]].append(test_target[i])
    else:
        test_dict[test_input[i]] = [test_target[i]] 
        
# Display test dataset example        
test_key = list(test_dict.keys())[0]
print('\nTest set example:')
print('- source:    %s' %test_key)
print('- target(s): %s' %test_dict[test_key])

'''
Restore the model and construct the encoder and decoder.
'''

# restore the models
def my_load_model(model_filename, model_weights_filename):
    '''
    restore models from serializations
    '''
    with open(model_filename, 'r', encoding='utf8') as fil:
        model = model_from_json(fil.read())
    model.load_weights(model_weights_filename)
    return model

# load models
decoder_model = my_load_model('../dnns/' + eval_path + '/decoder.json', 
                              '../dnns/' + eval_path + '/decoder_weights.h5')
encoder_model = my_load_model('../dnns/' + eval_path + '/encoder.json', 
                              '../dnns/' + eval_path + '/encoder_weights.h5')

# visualize models
print("\nENCODER:")
encoder_model.summary()
print("\nDECODER:")
decoder_model.summary()

'''
Reverse-lookup token index to decode sequences back to
something readable.
'''
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())

reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

#print(reverse_input_char_index)
#print(reverse_target_char_index)

'''
Beam-search decoding, where 
a beam size K=1 reverts to best-first decoding.
'''

# decodes an input sequence using "beam search".
def decode_sequence_beam(xinput_seq, max_len, k):
    '''
    decodes seq2seq prediction using beam search
    ''' 
    # result
    res = OrderedDict()
    s_queue = []
    # intialize root with root prediction
    states_value = encoder_model.predict(xinput_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.
    output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
    #
    samp_tok_indx       = np.argsort(output_tokens)[0, -1, :][-k:]
    samp_chars          = [reverse_target_char_index[token_ind] for token_ind in samp_tok_indx]
    samp_tok_prob       = sorted(output_tokens[0, -1, :])[:]
    #
    for j in range(0,len(samp_chars)):
        tok_ind     = samp_tok_indx[j]
        token       = samp_chars[j]
        cost        = samp_tok_prob[j]
        target_seq  = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, tok_ind] = 1.
        root = State(token=token, 
                     hidden=[h,c], 
                     cost=0 + np.log(cost), 
                     mod=decoder_model, 
                     targ_seq=target_seq, 
                     pred=output_tokens,
                     reverse_target_char_index=reverse_target_char_index,
                     num_decoder_tokens=num_decoder_tokens)
        s_queue.append(root)
    # do beam search
    beam = BeamSearch(input_seq=xinput_seq, 
                      queue=s_queue,
                      epochs=max_len,
                      beam_size=k)
    beam.traverse()
    # update results with results of beam search
    for state in beam.queue:
        res[state.token]=state.cost
    # return result
    return res

'''
We create structures to save the results in disk
'''

df_bleu = pd.DataFrame(columns=['bleu@k', 
                        'bleu', 'num_samples', 
                        'beam_size', 'method'])
df_res  = pd.DataFrame(columns=['source', 
                        'predicted', 'target', 'beam_size'])

print()
print(df_res)
print()
print(df_bleu)

'''
We evaluate predictions on N samples on a beam of size K
using BLEU-4 in different variations
'''

def evaluate(N, K, method="corpus-variants", df_bleu=df_bleu, df_res=df_res):
    '''
    Measure BLEU sentence by sentence, and then return average
    For each prediction in the beam, pick the highest
    '''
    print()
    # we use BLEU
    chc  = SmoothingFunction()
    smooth = chc.method2 # Laplace (+1) smoothing for proportion estimators
    #
    inputs = []
    preds = []
    targets = []
    #
    if method=="sentence":
        bleus = []    
        for seqs_index in range(N):
            # populate predictions vs. targets for BLEU
            xinput_seq          = encoder_input_data[seqs_index: seqs_index + 1]
            decoded_sentences   = decode_sequence_beam(xinput_seq, max_decoder_seq_length, K)
            best_decoding       = max(decoded_sentences).replace('\n','')
            #print(input_texts[seqs_index], '\t', decoded_sentences)
            pred_sent   = list(best_decoding)
            inp_sent    = test_input[seqs_index]
            tar_sent    = test_target[seqs_index]
            ref_sent    = list(tar_sent)
            bleu        = sentence_bleu([ref_sent], pred_sent, smoothing_function=smooth)
            bleus.append(bleu)
            preds.append(best_decoding)    
            targets.append(tar_sent)
            inputs.append(inp_sent)
            df_res = df_res.append(pd.DataFrame({'source': inp_sent, 'predicted':best_decoding,
                        'target':str(test_dict[inp_sent]), 'beam_size':K},index=[0]),
                               ignore_index=True)
        print('results:     BLEU-4 (macro): %s' %(np.mean(bleus)), '\tbeam size: %s' %K, '\tsamples: %s' %N)
        df_bleu = df_bleu.append(pd.DataFrame({'bleu@k':'bleu@'+str(K), 'bleu':np.mean(bleus), 
                        'num_samples':N, 'beam_size':K, 'method':'sentence'},index=[0]),
                               ignore_index=True)
    #        
    elif method == "sentence-variants":   
        bleus = []
        for seqs_index in range(N):
            # populate predictions vs. targets for BLEU
            xinput_seq          = encoder_input_data[seqs_index: seqs_index + 1]
            decoded_sentences   = decode_sequence_beam(xinput_seq, max_decoder_seq_length, K)
            best_decoding       = max(decoded_sentences).replace('\n','')
            pred_sent   = list(best_decoding)
            inp_sent    = test_input[seqs_index]
            ref_sents   = [list(s) for s in test_dict[inp_sent]]
            bleu        = sentence_bleu(ref_sents, pred_sent, smoothing_function=smooth)
            bleus.append(bleu)
            preds.append(best_decoding)    
            targets.append(test_dict[inp_sent])
            inputs.append(inp_sent)
            df_res = df_res.append(pd.DataFrame({'source': inp_sent, 'predicted':best_decoding, 
                        'target':str(test_dict[inp_sent]),'beam_size':K},index=[0]),
                               ignore_index=True)
        print('results:     BLEU-4: (macro w. variants) %s' %(np.mean(bleus)), '\tbeam size: %s' %K, '\tsamples: %s' %N)
        df_bleu = df_bleu.append(pd.DataFrame({'bleu@k':'bleu@'+str(K), 'bleu':np.mean(bleus), 
                        'num_samples':N, 'beam_size':K, 'method':'sentence-variants'},index=[0]),
                               ignore_index=True)
    #
    elif method == "corpus":   
        refs = []
        pred = []
        for seqs_index in range(N): 
            # populate predictions vs. targets for BLEU
            xinput_seq          = encoder_input_data[seqs_index: seqs_index + 1]
            decoded_sentences   = decode_sequence_beam(xinput_seq, max_decoder_seq_length, K)
            best_decoding       = max(decoded_sentences).replace('\n','')
            pred_sent   = list(best_decoding)
            inp_sent    = test_input[seqs_index]
            tar_sent    = test_target[seqs_index]
            ref_sent    = list(tar_sent)
            pred.append(pred_sent)
            refs.append([ref_sent])
            preds.append(best_decoding)    
            targets.append(tar_sent)
            inputs.append(inp_sent)
            df_res = df_res.append(pd.DataFrame({'source': inp_sent, 'predicted':best_decoding, 
                        'target':str(test_dict[inp_sent]),'beam_size':K},index=[0]),
                                ignore_index=True)
        bleu  = corpus_bleu(refs, pred, smoothing_function=smooth)
        print('results:     BLEU-4: (micro) %s' %bleu, '\tbeam size: %s' %K, '\tsamples: %s' %N)
        df_bleu = df_bleu.append(pd.DataFrame({'bleu@k':'bleu@'+str(K), 'bleu':bleus, 
                        'num_samples':N, 'beam_size':K, 'method':'corpus'},index=[0]),
                                 ignore_index=True)
    #
    elif method == "corpus-variants":   
        refs = []
        pred = []        
        for seqs_index in range(N):
            # populate predictions vs. targets for BLEU
            xinput_seq          = encoder_input_data[seqs_index: seqs_index + 1]
            decoded_sentences   = decode_sequence_beam(xinput_seq, max_decoder_seq_length, K)
            best_decoding       = max(decoded_sentences).replace('\n','')
            pred_sent   = list(best_decoding)
            inp_sent    = test_input[seqs_index]
            ref_sents   = [list(s) for s in test_dict[inp_sent]]
            pred.append(pred_sent)
            refs.append(ref_sents)            
            preds.append(best_decoding)    
            targets.append(test_dict[inp_sent])
            inputs.append(inp_sent)
            df_res = pd.concat([df_res, pd.DataFrame( {'source': inp_sent, 'predicted':best_decoding, 
                        'target':str(test_dict[inp_sent]),'beam_size':K}, index=[0])],
                                ignore_index=True)
        bleu  = corpus_bleu(refs, pred, smoothing_function=smooth)
        print('results:     BLEU-4: (macro w. variants) %s' %bleu, '\tbeam size: %s' %K, '\tsamples: %s' %N)
        df_bleu = df_bleu.append(pd.DataFrame({'bleu@k':'bleu@'+str(K), 'bleu':bleu, 
                        'num_samples':N, 'beam_size':K, 'method':'corpus-variants'},index=[0]),
                                ignore_index=True)
    #
    else:
        pass
    #
    print('source:      %s' %inputs[0:min(10,N)])
    print('predicted:   %s' %preds[0:min(10,N)])
    print('target:      %s' %targets[0:min(10,N)])
    print()
    return df_res, df_bleu

'''
Evaluation and inference w. beam size =< 5
'''
      
df_res_x, df_bleu_x = evaluate(num_samples,1)
df_bleu             = df_bleu.append(df_bleu_x, ignore_index=True)
df_res              = df_res.append(df_res_x, ignore_index=True)
df_res_x, df_bleu_x = evaluate(num_samples,2)
df_bleu             = df_bleu.append(df_bleu_x, ignore_index=True)
df_res              = df_res.append(df_res_x, ignore_index=True)
df_res_x, df_bleu_x = evaluate(num_samples,3)
df_bleu             = df_bleu.append(df_bleu_x, ignore_index=True)
df_res              = df_res.append(df_res_x, ignore_index=True)
df_res_x, df_bleu_x = evaluate(num_samples,4)
df_bleu             = df_bleu.append(df_bleu_x, ignore_index=True)
df_res              = df_res.append(df_res_x, ignore_index=True)
df_res_x, df_bleu_x = evaluate(num_samples,5)
df_bleu             = df_bleu.append(df_bleu_x, ignore_index=True)
df_res              = df_res.append(df_res_x, ignore_index=True)

# display results
print(df_res.head())
print(df_res.shape)
print()
print(df_bleu.head())
print(df_bleu.shape)
print()

'''
Save results
'''


df_bleu.to_csv('../dnns/' + eval_path + '/test_bleus.csv', sep='\t', index=False)
df_res.to_csv('../dnns/' + eval_path + '/test_preds.csv', sep='\t', index=False)

print('Saving results:')
print('- bleus:', '../dnns/' + eval_path + '/test_bleus.csv')
print('- predictions:', '../dnns/' + eval_path + '/test_preds.csv')