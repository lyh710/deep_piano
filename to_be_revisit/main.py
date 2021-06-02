import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import tensorflow as tf

from utils import split_midi_train_test, midi_2_notes, notes_2_midi, midi_2_csv, df_split
from models import *

if __name__ == "__main__":
    # if gpu ready
    assert tf.test.is_gpu_available()

    # midi_dir
    midi_dir = os.path.join(os.getcwd(), 'midi')
    print(midi_dir)

    # identify piano-like midi from all the downloads
    # leave one of the piano files out as Test, using all others as Train
    split_midi_train_test(midi_dir)

    # convert all the piano files in train folder into one csv corpus
    train_dir = os.path.join(os.getcwd(), 'piano', 'train')

    data_dir = os.path.join(os.getcwd(), 'data')
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)

    out_fn    = os.path.join(data_dir, 'corpus.csv')
    midi_2_csv(train_dir, out_fn, small_f=False)

    # split corpus into df_train & df_valid
    corpus_fn = os.path.join(data_dir, 'corpus.csv')
    df_corpus = pd.read_csv(corpus_fn, header=None, names=['notes'])
    print(df_corpus.shape)
    
    # df_dnn = df_corpus[df_corpus['notes'].apply(lambda x: len(x.split(' ')))<=5000][['notes']]
    df_dnn = df_split(df_corpus, split = 0.1)

    df_train = df_dnn[df_dnn['split']=='train']
    df_train.to_csv(os.path.join(data_dir, 'df_train.csv'), index=False)

    df_valid = df_dnn[df_dnn['split']=='valid']
    df_valid.to_csv(os.path.join(data_dir, 'df_valid.csv'), index=False)

    import dnn_data_ready
    import dnn_model_train
