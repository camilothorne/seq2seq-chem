'''
Created on 21 Apr 2023

@author: camilo thorne

Self attention layers for RNNs.
Works with Keras and TF 1.x, CuDNN 8.x, and Python 3.6.x
'''

# Ignore all warnings
import warnings
warnings.filterwarnings('ignore')

# Mute TF warnings
import tensorflow as tf, os
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Keras
from keras.layers import Layer
import keras.backend as K
from keras import backend as K, initializers, regularizers, constraints


class SelfAttention(Layer):
    '''
    Class implementing self-attention from [https://arxiv.org/pdf/1512.08756.pdf]
    
    See: 
    - https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137
    - https://machinelearningmastery.com/adding-a-custom-attention-layer-to-recurrent-neural-network-in-keras/
    '''
    def __init__(self,**kwargs):
        '''
        constructor:
        inherits params from Layer
        '''
        super(SelfAttention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        '''
        intitializes the attention weight matrices in
        the layer
        '''
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(SelfAttention, self).build(input_shape)
 
    def call(self,x):
        '''
        forward pass for back-propagation
        '''
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention(Layer):
    """
        Keras Layer that implements an (self) attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        
        See: https://gist.github.com/cbaziotis/6428df359af27d58078ca5ed9792bd6d
        
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Note: The layer has been tested with Keras 1.x
        Example:
        
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2 - Get the attention scores
            hidden = LSTM(64, return_sequences=True)(words)
            sentence, word_scores = Attention(return_attention=True)(hidden)
    """
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):

        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]


# check to see if it compiles
if __name__ == '__main__':
    
    # Mute TF warnings
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    from keras.layers import Input, LSTM, Dense
    from keras.models import Model
    #from keras.layers.wrappers import Bidirectional
    
    def create_RNN_with_self_attention(hidden_units, dense_units, input_shape, activation):
        x=Input(shape=input_shape)
        RNN_layer = LSTM(hidden_units, return_sequences=True, activation=activation)(x)
        attention_layer = SelfAttention()(RNN_layer)
        outputs=Dense(dense_units, trainable=True, activation='softmax')(attention_layer)
        model=Model(x,outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])    
        return model    

    def create_RNN_with_attention(hidden_units, dense_units, input_shape, activation):
        x=Input(shape=input_shape)
        RNN_layer = LSTM(hidden_units, return_sequences=True, activation=activation)(x)
        _, attention_layer = Attention(return_attention=True)(RNN_layer)
        outputs=Dense(dense_units, trainable=True, activation='softmax')(attention_layer)
        model=Model(x,outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])    
        return model     
    
    model_self_attention = create_RNN_with_self_attention(hidden_units=128, dense_units=1, 
                                  input_shape=(10,1), activation='tanh')
    model_self_attention.summary()
    
    model_attention = create_RNN_with_attention(hidden_units=128, dense_units=1, 
                                  input_shape=(10,1), activation='tanh')
    model_attention.summary()