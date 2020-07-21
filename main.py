import warnings
warnings.filterwarnings('ignore')

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

from utils import * 
from models import *

if __name__ == "__main__":
    # if gpu ready
    assert tf.test.is_gpu_available()

    # data_dir
    data_dir = os.path.join(os.getcwd(),'data')
    print(data_dir)

    # convert mp3 to wav within data_dir
    mp3_to_wave(data_dir)

    # leave one of the music files out as Test, using all others as Train
    split_music_train_test(data_dir)

    # convert all music files in 'train" folder into (x,y) training pairs
    train_to_x_y_all(data_dir)
    
    # prepare data
    # 1. load all data
    # 2. split to Train & Valid
    x_train, y_train, x_valid, y_valid = load_data(data_dir, split_thr=split_thr)

    # train the DNN
    model = model_dnn(x_train, y_train, x_valid, y_valid)

    # music generate
    # out_seq = music_generate(data_dir, model_version, start_seq, sr=sr, full_len=full_len, target_size=target_size)

    


