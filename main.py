import warnings
warnings.filterwarnings('ignore')

import os
import tensorflow as tf

from utils import * 
from models import *
from music_config import music_config

sr          = music_config['sr']
win         = music_config['win']
stride      = music_config['stride']
target_size = music_config['target_size']
split_rate  = music_config['split_rate']
full_len    = music_config['full_len']

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
    # 3. normalize via L2
    x_train_norm, y_train, x_valid_norm, y_valid = load_data(data_dir, split_thr=0.9)

