'''
Created on 10 May 2022
@author: camilo thorne

Restore a character-level sequence to sequence model from disk and use it
to generate predictions.

This script loads the model saved by seq2seqTrain.py and generates
sequences from it.  It assumes that no changes have been made (for example:
latent_dim is unchanged, and the input data and model architecture are unchanged).

It relies on "beam" search.
'''
from __future__ import print_function
from functools import total_ordering

import warnings
warnings.filterwarnings('ignore')  # ignore all warnings
#warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
from keras.models import model_from_json
import tensorflow as tf, os
from collections import OrderedDict
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu

# mute warnings
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

latent_dim  = 128       # Latent dimensionality of the encoding space.
num_samples = 10000     # Number of samples to test on.

# Path to the data txt file on disk.
data_path   = '../data/all-smi2smi.tsv'     # for char table
test_path   = '../data/test-smi2smi.tsv'    # for evaluation

'''
Recreate the char table used for training
NOTE: the data must be identical, in order for the character -> integer
mappings to be consistent.
We omit encoding target_texts since they are not needed.
'''
input_texts         = []
target_texts        = []
input_characters    = set()
target_characters   = set()

with open(data_path, 'r', encoding='utf-8') as f:
    '''
    open full corpus
    '''
    lines = f.read().split('\n')    
        
for line in lines[: min(num_samples, len(lines) - 1)]:
    '''
    parse full corpus, build char table
    '''
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

# Load test set
test_input          = []
test_target         = []  
         
with open(test_path, 'r', encoding='utf-8') as f:
    '''
    open test corpus
    '''
    lines_t = f.read().split('\n')
   
for line in lines_t[: min(num_samples, len(lines) - 1)]:
    '''
    parse test corpus
    '''
    input_text, target_text = line.split('\t') 
    test_input.append(input_text)
    test_target.append(target_text)               

input_characters        = sorted(list(input_characters))
target_characters       = sorted(list(target_characters))

num_encoder_tokens      = len(input_characters)
num_decoder_tokens      = len(target_characters)

max_encoder_seq_length  = max([len(txt) for txt in input_texts])
max_decoder_seq_length  = max([len(txt) for txt in target_texts])

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

print()
print('Number of samples:',                     len(test_input))
print('Number of unique input chars/tokens:',   num_encoder_tokens)
print('Number of unique output chars/tokens:',  num_decoder_tokens)
print('Max sequence length for inputs:',        max_encoder_seq_length)
print('Max sequence length for outputs:',       max_decoder_seq_length)
print()  

input_token_index  = dict(
    [(char, i) for i, char in enumerate(input_characters)])

target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(test_input), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')

for i, input_text in enumerate(test_input):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.

# restore the model and construct the encoder and decoder.
def my_load_model(model_filename, model_weights_filename):
    '''
    restore models from serializations
    '''
    with open(model_filename, 'r', encoding='utf8') as fil:
        model = model_from_json(fil.read())
    model.load_weights(model_weights_filename)
    return model

# load models
decoder_model = my_load_model('../dnns/decoder_lstm-100-128.json', 
                              '../dnns/decoder_lstm_weights-100-128.h5')
encoder_model = my_load_model('../dnns/encoder_lstm-100-128.json', 
                              '../dnns/encoder_lstm_weights-100-128.h5')

# visualize models
print("ENCODER:")
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

class BeamSearch:
    '''
    Class encapsulating beam search algorithm
    '''
    
    def __init__(self, input_seq=None, queue=None, beam_size=0, epochs=None):
        '''
        generate and search state space
        '''
        self.input_seq = input_seq  # start search with input sequence
        self.queue = queue          # start search with empty queue (will be a priority queue of sorts!)
        self.beam_size = beam_size  # start search with fixed beam size
        self.epochs = epochs        # max depth of search
    
    def sort_queue(self):
        '''
        sort queue from cheapest to most costly state
        '''
        self.queue = sorted(self.queue)
    
    def prune_queue(self):
        '''
        prune queue and keep the top K most costly states
        '''
        self.queue = self.queue[-self.beam_size:]
    
    def traverse(self):
        '''
        traverse beam
        '''
        # stop when True
        stop_condition = False
        # count iteration / search depth
        cnt_iter = 0
        while not stop_condition:
            '''
            LIFO traversal:
            
            - pick and remove the last K states in beam
            - a beam is a queue that has never more than K
              (unvisited) elements
            ''' 
            beam_states = []
            for _ in range(0, len(self.queue)):
                curr_state = self.queue[-1]             
                curr_state.get_successors()
                beam_states = beam_states + curr_state.states
                self.queue.remove(curr_state)
            for state in beam_states:
                self.queue.append(state)                    # add all successors
            self.sort_queue()                               # sort
            self.prune_queue()                              # prune            
            sample_toks = [s.token for s in self.queue]
            cnt_iter = cnt_iter + 1
            '''
            stop when stop character is predicted, or when
            max sequence length is reached
            '''
            if (
                ('\n' in " ".join(sample_toks)) or 
                (cnt_iter > self.epochs)
                ):
                stop_condition = True

@total_ordering
class State:
    '''
    Class encapsulating a decoding state, ordered by cost
    '''
    
    def __init__(self, token=None, hidden=None, cost=None, mod=None, targ_seq=None, pred=None):
        '''
        create state
        '''
        self.token = token              # current input token at time t of state
        self.hidden = hidden            # current hidden state(s) at time t
        self.cost = cost                # cost of path leading to state (from time t==0 to time t)
        self.mod = mod                  # seq2seq model
        self.targ_seq = targ_seq        # current decoded sequence leading to state (from time t==0 until time t)
        self.pred = pred                # current prediction at time t of state
        self.states = None              # successor states
        
    def __repr__(self):
        # display x and y instead of address
        return f'State(input_dims={self.targ_seq.shape}, cost={self.cost})'
        
    @property
    def rank(self):
        return self.cost        
        
    def __eq__(self, other):
        '''
        states are identical if costs are identical, and tokens are identical
        '''
        return self.rank == other.rank

    def __lt__(self, other):
        '''
        states are smaller if path cost is smaller
        '''
        return self.rank < other.rank
        
    def get_successors(self):
        '''
        expand state to all neighbors
        '''
        states              = []
        output_tokens, h, c = self.mod.predict([self.targ_seq] + self.hidden)
        samp_tok_indx       = np.argsort(output_tokens)[0, -1, :][:]
        samp_chars          = [reverse_target_char_index[token_ind] for token_ind in samp_tok_indx]
        samp_tok_prob       = sorted(output_tokens[0, -1, :])[:]
        for j in range(0,len(samp_chars)):
            tok_ind     = samp_tok_indx[j]
            token       = samp_chars[j]
            cost        = samp_tok_prob[j]
            target_seq  = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, tok_ind] = 1.
            state = State(token=self.token + token, hidden=[h,c], 
                          cost=(self.cost + np.log(cost)), 
                          mod=self.mod, 
                          targ_seq=target_seq, 
                          pred=self)
            states.append(state)
        self.states = states

# Decodes an input sequence using "beam search".
def decode_sequence_beam(xinput_seq, max_len, k, num):
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
                     pred=output_tokens)
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
       
def evaluate(N, K, method="corpus-variants"):
    '''
    Measure BLEU sentence by sentence, and then return average
    For each prediction in the beam, pick the highest
    '''
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
            decoded_sentences   = decode_sequence_beam(xinput_seq, max_decoder_seq_length, K, seqs_index)
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
        print('results:     BLEU-4 (macro): %s' %(np.mean(bleus)), '\tbeam size: %s' %K, '\tsamples: %s' %N)
    #        
    elif method == "sentence-variants":   
        bleus = []
        for seqs_index in range(N):
            # populate predictions vs. targets for BLEU
            xinput_seq          = encoder_input_data[seqs_index: seqs_index + 1]
            decoded_sentences   = decode_sequence_beam(xinput_seq, max_decoder_seq_length, K, seqs_index)
            best_decoding       = max(decoded_sentences).replace('\n','')
            #print(input_texts[seqs_index], '\t', decoded_sentences)
            pred_sent   = list(best_decoding)
            inp_sent    = test_input[seqs_index]
            ref_sents   = [list(s) for s in test_dict[inp_sent]]
            bleu        = sentence_bleu(ref_sents, pred_sent, smoothing_function=smooth)
            bleus.append(bleu)
            preds.append(best_decoding)    
            targets.append(test_dict[inp_sent])
            inputs.append(inp_sent)
        print('results:     BLEU-4: (macro w. variants) %s' %(np.mean(bleus)), '\tbeam size: %s' %K, '\tsamples: %s' %N)
    #
    elif method == "corpus":   
        refs = []
        pred = []
        for seqs_index in range(N): 
            # populate predictions vs. targets for BLEU
            xinput_seq          = encoder_input_data[seqs_index: seqs_index + 1]
            decoded_sentences   = decode_sequence_beam(xinput_seq, max_decoder_seq_length, K, seqs_index)
            best_decoding       = max(decoded_sentences).replace('\n','')
            #print(input_texts[seqs_index], '\t', decoded_sentences)
            pred_sent   = list(best_decoding)
            inp_sent    = test_input[seqs_index]
            tar_sent    = test_target[seqs_index]
            ref_sent    = list(tar_sent)
            pred.append(pred_sent)
            refs.append([ref_sent])
            preds.append(best_decoding)    
            targets.append(tar_sent)
            inputs.append(inp_sent)
        bleu  = corpus_bleu(refs, pred, smoothing_function=smooth)
        print('results:     BLEU-4: (micro) %s' %bleu, '\tbeam size: %s' %K, '\tsamples: %s' %N)
    #
    elif method == "corpus-variants":   
        refs = []
        pred = []        
        for seqs_index in range(N):
            # populate predictions vs. targets for BLEU
            xinput_seq          = encoder_input_data[seqs_index: seqs_index + 1]
            decoded_sentences   = decode_sequence_beam(xinput_seq, max_decoder_seq_length, K, seqs_index)
            best_decoding       = max(decoded_sentences).replace('\n','')
            #print(input_texts[seqs_index], '\t', decoded_sentences)
            pred_sent   = list(best_decoding)
            inp_sent    = test_input[seqs_index]
            ref_sents   = [list(s) for s in test_dict[inp_sent]]
            pred.append(pred_sent)
            refs.append(ref_sents)            
            preds.append(best_decoding)    
            targets.append(test_dict[inp_sent])
            inputs.append(inp_sent)
        bleu  = corpus_bleu(refs, pred, smoothing_function=smooth)
        print('results:     BLEU-4: (macro w. variants) %s' %bleu, '\tbeam size: %s' %K, '\tsamples: %s' %N)
    #
    else:
        pass
    #
    print('source:      %s' %inputs[0:min(10,N)])
    print('predicted:   %s' %preds[0:min(10,N)])
    print('target:      %s' %targets[0:min(10,N)])
    print()
      
evaluate(10,1)
evaluate(10,2)
evaluate(10,3)
evaluate(10,4)
evaluate(10,5)
evaluate(10,6)
evaluate(10,7)
evaluate(10,8)
evaluate(10,9)
evaluate(10,10)        
evaluate(10,11)
evaluate(10,12)
evaluate(10,13)
evaluate(10,14)
evaluate(10,15)

# evaluate(20,1,method="corpus")
# evaluate(20,2,method="corpus")
# evaluate(20,3,method="corpus")
# evaluate(20,4,method="corpus")
# evaluate(20,5,method="corpus")
# evaluate(20,6,method="corpus")
# evaluate(20,7,method="corpus")
# evaluate(20,8,method="corpus")
# evaluate(20,9,method="corpus")
# evaluate(20,10,method="corpus") 

# evaluate(20,1,method="sentence")
# evaluate(20,2,method="sentence")
# evaluate(20,3,method="sentence")
# evaluate(20,4,method="sentence")
# evaluate(20,5,method="sentence")
# evaluate(20,6,method="sentence")
# evaluate(20,7,method="sentence")
# evaluate(20,8,method="sentence")
# evaluate(20,9,method="sentence")
# evaluate(20,10,method="sentence")  
      
# evaluate(200,1,method="corpus-variants")
# evaluate(200,2,method="corpus-variants")
# evaluate(200,3,method="corpus-variants")
# evaluate(200,4,method="corpus-variants")
# evaluate(200,5,method="corpus-variants")
# evaluate(200,6,method="corpus-variants")
# evaluate(200,7,method="corpus-variants")
# evaluate(200,8,method="corpus-variants")
# evaluate(200,9,method="corpus-variants")
# evaluate(200,10,method="corpus-variants")

# evaluate(20,1,method="sentence-variants")
# evaluate(20,2,method="sentence-variants")
# evaluate(20,3,method="sentence-variants")
# evaluate(20,4,method="sentence-variants")
# evaluate(20,5,method="sentence-variants")
# evaluate(20,6,method="sentence-variants")
# evaluate(20,7,method="sentence-variants")
# evaluate(20,8,method="sentence-variants")
# evaluate(20,9,method="sentence-variants")
# evaluate(20,10,method="sentence-variants")