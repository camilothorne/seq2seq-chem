'''
Created on 5 Jun 2022
@author: Camilo Thorne

Corpus preprocessor to build character one-hot encodings for each
data point.

'''


import numpy as np
import pandas as pd


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
    

    def build_char_table(self, data_path, xrows):
        '''
        Read (full) dataset
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
        self._set_input_chars()
        self._build_embeddings()
        
                    
    def _set_input_chars(self):
        '''
        Set internal variables (private method)
        '''
        self.input_characters = sorted(list(self.input_characters))
        self.target_characters = sorted(list(self.target_characters))
        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])
        self.input_token_index  = dict(
            [(char, i) for i, char in enumerate(self.input_characters)])
        self.target_token_index = dict(
            [(char, i) for i, char in enumerate(self.target_characters)])
        

    def corpus_stats(self):
        '''
        Display dataset stats
        '''
        print('Total number of examples:            ', len(self.input_texts))
        print('Number of unique input tokens/chars: ', self.num_encoder_tokens)
        print('Number of unique output tokens/chars:', self.num_decoder_tokens)
        print('Max sequence length for inputs:      ', self.max_encoder_seq_length)
        print('Max sequence length for outputs:     ', self.max_decoder_seq_length)
        print('Shape of encoder inputs (all):       ', self.encoder_input_data.shape)
        print('Shape of decoder inputs (all):       ', self.decoder_input_data.shape)
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
            print('Sample size:                   ', end-begin)  
            print('Begin:                         ', begin)  
            print('End:                           ', end)        
            print('Shape of encoder inputs (res): ', input_enc.shape)
            print('Shape of decoder inputs (res): ', input_dec.shape)
            print('Shape of decoder targets (res):', target_dec.shape)
            print('Corpus (res):                  ', corpus.shape)
        else:
            pass
        return (input_enc, input_dec, target_dec, corpus)
        
        
    def _build_embeddings(self):
        '''
        Read train/test data and create embeddings
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

                    
if __name__ == "__main__":
     
    # Path to the data file on disk.
    xdata_path           = '../data/all-smi2smi.tsv'
    xnum_samples = 1000     # Max number of samples to train on.
 
    print('----------------')
 
    char_table1 = OneHotEncode()
    char_table1.build_char_table(xdata_path, ['smiles','smiles-out'])
    char_table1.corpus_stats()
     
    print('----------------')      
     
    (x,y,z,w) = char_table1.select_sample(0, xnum_samples*3, print_stats=True)
    print('source','\t','target')
    for i in range(0, min(w.shape[0],10)):
        print(w[i,0],'\t',w[i,1])
     
#     print('----------------')      
#    
#     char_table1.select_sample(xnum_samples, xnum_samples*2, print_stats=True)
#                     
#     print('----------------')                    
#                     
#     char_table2 = OneHotEncode()
#     char_table2.build_char_table(xdata_path, ['smiles','smiles-out'])
#     char_table2.corpus_stats()
#     
#     print('----------------')         
#     
#     char_table2.select_sample(0, xnum_samples, print_stats=True)
#     
#     print('----------------')      
#     
#     char_table2.select_sample(xnum_samples, xnum_samples+200, print_stats=True)
