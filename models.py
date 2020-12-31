import string, shutil, os, pickle

import numpy as np
np.random.seed(11)

import pandas as pd

from time import time

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras import optimizers

def text_2_token_seq(tokenizer, corpus):
    '''
    In case of tokenizer has been trained, use it to convert corpus into sequences (valid or test)
    '''
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences
    
def get_sequence_of_tokens(corpus):
    '''
    tokenize training corpus into word (n-gram) sequence
    '''
    tokenizer = Tokenizer(lower=False, filters='', char_level=False, oov_token='OOV')
    tokenizer.fit_on_texts(corpus)
    
    total_words = len(tokenizer.word_index) + 1
    
    ## convert data to sequence of tokens 
    input_sequences = text_2_token_seq(tokenizer, corpus)
    
    return tokenizer, input_sequences, total_words

def generate_padded_sequences(input_sequences, total_words, max_sequence_len):
    '''
    generate pair of (input word sequence, next word)
    '''
    if max_sequence_len is None: # suggesting it is "train" and need to be inferred from input_sequences
        max_sequence_len  = max([len(x) for x in input_sequences])
        
    input_sequences   = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

def data_generator(inputPath, tokenizer, total_words, max_sequence_len, gs=dnn_config['gen_size']):
    '''
    data generator which can produce (x,y) pairs with desired batch size against very large raw data file, to avoid Out-Of-Memory issue
    we expect the tokenizer has been fit
    '''
    # leverage Pandas built-in function to produce generator
    f  = pd.read_csv(inputPath, iterator=True, chunksize=gs)    
    while True: # generator is supposed to loop infinitely, and the start and stop of epoch will be controlled by steps_per_epoch later on by tf
        try:
            corpus                 = f.get_chunk(gs)['dnn_text'].apply(space_punctuation).tolist()
            input_sequences        = text_2_token_seq(tokenizer, corpus)
            x, y, max_sequence_len = generate_padded_sequences(input_sequences, total_words, max_sequence_len=max_sequence_len)
            yield (x, y)
        except: # if data exhausted, reset to the beginning of the data
            f  = pd.read_csv(inputPath, iterator=True, chunksize=gs)
        
def generate_text(seed_text, tokenizer, model, max_sequence_len, stop_word='GfsStop', max_words=3500, verbose=0):
    
    while (seed_text.split(' ')[-1] != stop_word) and (max_words > 0):
    
        # tokenizer the seed_text
        seed_tokens = tokenizer.texts_to_sequences([seed_text])[0]
        
        # padd the token seq
        seed_pad    = np.array(pad_sequences([seed_tokens], maxlen=max_sequence_len-1, padding='pre'))
        
        # predict the next word
        predicted   = model.predict_classes(seed_pad, verbose=0)

        next_word = ''
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                next_word = word
                break
        seed_text = seed_text + ' ' + next_word

        max_words = max_words - 1
        
        if verbose > 1:
            print('max_words: {}, next_word: {}'.format(max_words, next_word))
        elif verbose > 0:
            print(max_words)
    
    return seed_text

def get_model(max_sequence_len, total_words):
    '''
    Define DNN structure
    '''
    input_len = max_sequence_len - 1
    
    model     = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, dnn_config['embed_dim'], input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(dnn_config['lstm_units']))
    
    model.add(Dropout(dnn_config['dropout']))
    
    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    # model compile
    optimizer = optimizers.Adam(learning_rate=dnn_config['lr_start'])
    model.compile(loss=dnn_config['loss'], optimizer=optimizer, metrics=['acc'])
    
    return model