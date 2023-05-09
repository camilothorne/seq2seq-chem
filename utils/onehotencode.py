'''
Created on 5 Jun 2022
@author: Camilo Thorne

Corpus preprocessor to build character one-hot encodings for each
data point.
'''


import numpy as np
import pandas as pd
import pickle as pk
import json


class ExpConfig(object):
    '''
    Generic reader of JSON config files
    and expects a file of the form:
    {
      'key1' : value1,
           ...
      'keyN' : valueN
    }
    '''
    
    def __init__(self, path):
        '''
        Init experiment hyperparameters
        '''
        self.data_name   = None      # Dataset name
        self.batch_size  = None      # Batch size for training
        self.epochs      = None      # Number of epochs to train for
        self.typ         = None      # Type of model
        self.latent_dim  = None      # Latent dimensionality of the encoding space
        self._read_config(path) # Read config
        
    def _read_config(self, path):
        '''
        Read JSON configuration, and set values
        '''
        with open(path) as handle:
            dict_config = json.loads(handle.read())
        self.data_name   = dict_config['data_name']
        self.batch_size  = dict_config['batch_size']
        self.epochs      = dict_config['epochs']
        self.typ         = dict_config['typ'] 
        self.latent_dim  = dict_config['latent_dim']
        
    def print_config(self):
        '''
        Print values
        '''
        print("\nHyperparameters")
        print("- dataset:     %s\n" %self.data_name +  
              "- batch size:  %s\n" %self.batch_size + 
              "- epochs:      %s\n" %self.epochs +
              "- type:        %s\n" %self.typ +
              "- latent dims: %s"   %self.latent_dim)

class OneHotEncode(object):
    '''
    Class to build one-hot character encoders
    for Seq2Seq experiments
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.input_texts         = []
        self.target_texts        = []
        self.input_characters    = set()
        self.target_characters   = set()
        self.num_encoder_tokens     = None
        self.num_decoder_tokens     = None
        self.max_encoder_seq_length = None
        self.max_decoder_seq_length = None
        self.encoder_input_data     = None
        self.decoder_input_data     = None
        self.decoder_target_data    = None
        self.input_token_index      = None
        self.target_token_index     = None
        self.corpus                 = None
    

    def build_char_table(self, data_path, xrows, norm_len=False):
        '''
        Read (full) dataset, and:
        
        - create char table of M characters for inputs (resp. target) 
        - for each input (resp. target) of length N, create a N x M table
        - create a K x N x M table for the K inputs (resp. targets)
        - results are saved in class-internal variables
        '''
        self.corpus = pd.read_csv(data_path, sep="\t", encoding='utf-8')
        for _, row in self.corpus.iterrows():
            input_text  = row[xrows[0]]
            target_text = row[xrows[1]]
            '''
            We use "tab" as the "start sequence" character
            for the targets, and "\n" as "end sequence" character.
            '''
            target_text = '\t' + target_text + '\n'
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            for char in input_text:
                if char not in self.input_characters:
                    self.input_characters.add(char)
            for char in target_text:
                if char not in self.target_characters:
                    self.target_characters.add(char)
        self._set_input_chars(norm_len)
        self._build_embeddings()
        
                    
    def _set_input_chars(self, norm_len):
        '''
        Set internal variables (private method)
    
        - lists of input and target chars
        - max number of chars in inputs and targets
        - max length of inputs anf targets
        - input token indexes (char dictionary of positions to chars)
        - target token indexes (char dictionary of positions to chars)
        - if norm_len==True, all sequences are defined to
        be of the same length
        '''
        self.input_characters = sorted(list(self.input_characters))
        self.target_characters = sorted(list(self.target_characters))
        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])
        if norm_len:
            seq_len = max([self.max_encoder_seq_length, self.max_decoder_seq_length])
            self.max_encoder_seq_length = seq_len
            self.max_encoder_seq_length = seq_len
        # We need dictionaries to decode predictions. The `argmax(...)` function
        # applied to a softmax layer returns the position in the output M-dimensional probability 
        # distribution vector with the highest softmax value.
        # This position is mapped back to a character with the help of these two dictionaries.
        self.input_token_index  = dict(
            [(char, j) for j, char in enumerate(self.input_characters)])
        self.target_token_index = dict(
            [(char, j) for j, char in enumerate(self.target_characters)])
        
        
    def _build_embeddings(self):
        '''
        Create one-hot char embeddings (private method)
        '''
        # we work with one-hot char/token vectors
        self.encoder_input_data  = np.zeros((len(self.input_texts), 
                                        self.max_encoder_seq_length, 
                                        self.num_encoder_tokens), dtype='float32')
        self.decoder_input_data  = np.zeros((len(self.input_texts), 
                                        self.max_decoder_seq_length, 
                                        self.num_decoder_tokens), dtype='float32')
        self.decoder_target_data = np.zeros((len(self.input_texts), 
                                        self.max_decoder_seq_length, 
                                        self.num_decoder_tokens), dtype='float32')
        # loop over data
        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            # loop
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, self.input_token_index[char]] = 1.
            # loop
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                self.decoder_input_data[i, t, self.target_token_index[char]] = 1.
                if t > 0:
                    '''
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    '''
                    self.decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.
        

    def corpus_stats(self):
        '''
        Display dataset stats
        '''
        print()
        print('Total number of examples:            ', len(self.input_texts))
        print('Number of unique input tokens/chars: ', self.num_encoder_tokens)
        print('Number of unique output tokens/chars:', self.num_decoder_tokens)
        print('Max sequence length for inputs:      ', self.max_encoder_seq_length)
        print('Max sequence length for outputs:     ', self.max_decoder_seq_length)
        print('Shape of encoder inputs  (all):      ', self.encoder_input_data.shape)
        print('Shape of decoder inputs  (all):      ', self.decoder_input_data.shape)
        print('Shape of decoder targets (all):      ', self.decoder_target_data.shape)
        
        
    def select_sample(self, begin, end, print_stats=False):
        '''
        Restrict dataset to subset of points
        '''
        input_enc  = self.encoder_input_data[begin:end,:,:]
        input_dec  = self.decoder_input_data[begin:end,:,:]
        target_dec = self.decoder_target_data[begin:end,:,:]
        corpus     = self.corpus.iloc[begin:end].values
        if print_stats:
            print()
            print('Sample size:                   ', end-begin)  
            print('Begin:                         ', begin)  
            print('End:                           ', end)        
            print('Shape of encoder inputs  (res):', input_enc.shape)
            print('Shape of decoder inputs  (res):', input_dec.shape)
            print('Shape of decoder targets (res):', target_dec.shape)
            print('Corpus (res):                  ', corpus.shape)
            print('Corpus (res):\n%s' %corpus[:min(corpus.shape[0],5),:])
        else:
            pass
        return (input_enc, input_dec, target_dec, corpus)

                    
    def to_pickle(self, path):
        '''
        Save class to Pickle:
        
        - Serialization must be repeated each time the class definition is changed
        '''
        with open(path, 'wb') as f:
            pk.dump(self, f)


    def from_pickle(self, path):
        '''
        Read class from Pickle:
        
        - Sets fields in empty object to the values in the serialized object
        - Serialization must be repeated each time the class definition is changed
        '''
        with open(path, 'rb') as f:
            self.__dict__.update(pk.load(f).__dict__)