'''
Created on 8 Jun 2022
@author: Camilo Thorne

Beam search implementation for "vanilla" seq2seq model derived from this paper:

    Nicolas Boulanger-Lewandowski, Yoshua Bengio and Pascal Vincent.
    "Audio chord recognition system with recurrent neural networks", ISMIR 2013.

    https://archives.ismir.net/ismir2013/paper/000243.pdf

'''

import numpy as np
from functools import total_ordering

@total_ordering
class State:
    '''
    Class encapsulating a decoding state, ordered by cost
    '''
    
    def __init__(self,  token=None, 
                        hidden=None, 
                        cost=None, 
                        mod=None, 
                        targ_seq=None, 
                        pred=None,
                        reverse_target_char_index=None,
                        num_decoder_tokens=None):
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
        self.reverse_target_char_index = reverse_target_char_index  # reverse char index
        self.num_decoder_tokens = num_decoder_tokens                # number of characters for softmax
        
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
        samp_chars          = [self.reverse_target_char_index[token_ind] for token_ind in samp_tok_indx]
        samp_tok_prob       = sorted(output_tokens[0, -1, :])[:]
        for j in range(0,len(samp_chars)):
            tok_ind     = samp_tok_indx[j]
            token       = samp_chars[j]
            cost        = samp_tok_prob[j]
            target_seq  = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, tok_ind] = 1.
            state = State(token=self.token + token, hidden=[h,c], 
                          cost=(self.cost + np.log(cost)), 
                          mod=self.mod, 
                          targ_seq=target_seq, 
                          pred=self)
            states.append(state)
        self.states = states
       
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
