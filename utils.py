import subprocess, os, shutil
import numpy as np
import librosa, math, pickle
from tensorflow.keras.utils import normalize
import matplotlib.pyplot as plt
from models import *
from music_config import music_config
from scipy.ndimage.interpolation import shift

sr          = music_config['sr']
win         = music_config['win']
stride      = music_config['stride']
target_size = music_config['target_size']
split_rate  = music_config['split_rate']
full_len    = music_config['full_len']

def mp3_to_wave(data_dir, bat_dir=os.path.join(os.getcwd(),'ffmpeg_mp3_to_wav.bat')):
    '''
    Input:
    - data_dir: mp3 files location
    - bat_dir: windows batch file that calls ffmpeg.exe
    Output:
    mp3 files will be converted to wav and saved in the same directory, i.e. filename.mp3 --> filename.wav
    '''
    
    subprocess.call([bat_dir, data_dir])

def split_music_train_test(data_dir):
    '''
    Input:
    - data_dir: wav files location
    Output:
    - a folder named "train", which contains all except one wav file
    - a folder named "test", which contains only one random selected wav file
    '''

    music_all   = [fn for fn in os.listdir(data_dir) if fn.endswith('.wav')]
    music_test  = [np.random.choice(music_all)]
    music_train = music_all.copy()
    music_train.remove(music_test[0])

    test_dir = os.path.join(data_dir, 'test')
    if os.path.isdir(test_dir):
        shutil.rmtree(test_dir)
        os.mkdir(test_dir)
    else:
        os.mkdir(test_dir)

    for fn in music_test:
        src_dir = os.path.join(data_dir,fn)
        dst_dir = os.path.join(test_dir, fn)
        shutil.move(src_dir, dst_dir)

    train_dir = os.path.join(data_dir, 'train')
    if os.path.isdir(train_dir):
        shutil.rmtree(train_dir)
        os.mkdir(train_dir)
    else:
        os.mkdir(train_dir)
        
    for fn in music_train:
        src_dir = os.path.join(data_dir,fn)
        dst_dir = os.path.join(train_dir, fn)
        shutil.move(src_dir, dst_dir)

def music_to_x_y(fn):
    '''
    Input:
    - fn: full path to one wav file
    - sr: sample rate of the music
    - win: historical window size to predict the next note, note this is NOT seconds
    - stride: step length to move the hisotrical window from the beginning to the end of the music file, note this is NOT seconds
    - target_size: how many numerical sample to predict, note this is numpy sample, not seconds
    Output:
    - data: numpy array with shape of (n, win)
    - labels: numpy array with shape of (n, 1)
    Note:
    - ignore the first 10 seconds of fn
    - ignore the last 10 seconds of fn
    '''
    data   = []
    labels = []
    
    # load music file into numpy array
    music_len = librosa.core.get_duration(filename=fn) # in seconds
    dataset,_ = librosa.load(fn, sr=sr)
    
    # the first and last 10 seconds is to be ignored, music should be longer than 20 seconds
    if (dataset.shape[0] == math.ceil(music_len*sr)) & (music_len > 20):
        
        print('%s is %s seconds long, sample rate at %s, leads to 1-D numpy array at length of %s' %(fn, music_len, sr, dataset.shape[0]))
    
        # ignore the first 10 seconds
        start_index = 10*sr+win

        # ignore the last 10 seconds
        end_index   = dataset.shape[0] - 10*sr - target_size
        # print('start from %s, end at %s' %(start_index, end_index))

        for i in range(start_index, end_index, stride):
            indices = range(i-win, i)
            data.append(dataset[indices,])
            labels.append(dataset[i-1+target_size])
            # print('start at %s, end at %s' % (i-win, i))
    else:
        
        print('%s is not used' %(fn))
    
    return np.array(data), np.array(labels)

def train_to_x_y_all(data_dir):
    '''
    assume a "train" folder under data_dir
    '''
    x = []
    y = []
    for fn in os.listdir(os.path.join(data_dir,'train')):
        if fn.endswith('.wav'):
            data, labels = music_to_x_y(os.path.join(data_dir,'train',fn))
            if (data.shape[0]>0) & (data.shape[0] == labels.shape[0]):
                x.append(data)
                y.append(labels)
    
    x_all  = np.concatenate(x, axis=0)
    fn     = os.path.join(data_dir,'train','x_all.pkl')
    if os.path.exists(fn):
        os.remove(fn)
    with open(fn, 'wb') as f:
        pickle.dump(x_all, f, protocol=4)
    
    y_all  = np.concatenate(y, axis=0)
    fn     = os.path.join(data_dir,'train','y_all.pkl')
    if os.path.exists(fn):
        os.remove(fn)
    with open(fn, 'wb') as f:
        pickle.dump(y_all, f, protocol=4)

def norm_data(x):
    '''
    x : numpy array
    return L2 norm
    '''
    return normalize(x, order=2)

def prep_data(data_dir, split_thr=split_rate):
    '''
    assume a "train" folder under data_dir
    '''
    # x_all
    fn     = os.path.join(data_dir,'train','x_all.pkl')
    if not os.path.exists(fn):
        print('training data not available, check')
        exit()
    else:
        with open(fn, 'rb') as f:
            x_all = pickle.load(f)
    
    # y_all
    fn     = os.path.join(data_dir,'train','y_all.pkl')
    if not os.path.exists(fn):
        print('training data not available, check')
        exit()
    else:
        with open(fn, 'rb') as f:
            y_all = pickle.load(f)
            
    assert x_all.shape[0] == y_all.shape[0]

    # split to Train & Valid
    # split rate
    train_split = int(x_all.shape[0]*split_thr)
    x_train = x_all[:train_split, :]
    y_train = y_all[:train_split]
    assert x_train.shape[0] == y_train.shape[0]
    x_valid = x_all[train_split:, :]
    y_valid = y_all[train_split:]
    assert x_valid.shape[0] == y_valid.shape[0]
    assert x_train.shape[0] + x_valid.shape[0] == x_all.shape[0]

    # normalize x
    x_train_norm = norm_data(x_train)
    x_valid_norm = norm_data(x_valid)
    
    # dump to disk
    # x_train
    fn     = os.path.join(data_dir,'train','x_train.pkl')
    if os.path.exists(fn):
        os.remove(fn)
    with open(fn, 'wb') as f:
        pickle.dump(x_train, f, protocol=4)
    
    # y_train
    fn     = os.path.join(data_dir,'train','y_train.pkl')
    if os.path.exists(fn):
        os.remove(fn)
    with open(fn, 'wb') as f:
        pickle.dump(y_train, f, protocol=4)
    
    # x_valid
    fn     = os.path.join(data_dir,'train','x_valid.pkl')
    if os.path.exists(fn):
        os.remove(fn)
    with open(fn, 'wb') as f:
        pickle.dump(x_valid, f, protocol=4)
    
    # y_valid
    fn     = os.path.join(data_dir,'train','y_valid.pkl')
    if os.path.exists(fn):
        os.remove(fn)
    with open(fn, 'wb') as f:
        pickle.dump(y_valid, f, protocol=4)

    # x_train_norm
    fn     = os.path.join(data_dir,'train','x_train_norm.pkl')
    if os.path.exists(fn):
        os.remove(fn)
    with open(fn, 'wb') as f:
        pickle.dump(x_train_norm, f, protocol=4)

    # x_valid_norm
    fn     = os.path.join(data_dir,'train','x_valid_norm.pkl')
    if os.path.exists(fn):
        os.remove(fn)
    with open(fn, 'wb') as f:
        pickle.dump(x_valid_norm, f, protocol=4)

    return x_train_norm, y_train, x_valid_norm, y_valid

def load_data(data_dir, split_thr=split_rate):
    
    # x_train_norm
    fn_x_train_norm = os.path.join(data_dir,'train','x_train_norm.pkl')

    # y_train
    fn_y_train      = os.path.join(data_dir,'train','y_train.pkl')

    # x_valid_norm
    fn_x_valid_norm = os.path.join(data_dir,'train','x_valid_norm.pkl')

    # y_valid
    fn_y_valid      = os.path.join(data_dir,'train','y_valid.pkl')

    if (os.path.exists(fn_x_train_norm)) and (os.path.exists(fn_y_train)) and (os.path.exists(fn_x_valid_norm)) and (os.path.exists(fn_y_valid)):
        with open(fn_x_train_norm, 'rb') as f:
            x_train_norm = pickle.load(f)
        
        with open(fn_y_train, 'rb') as f:
            y_train = pickle.load(f)

        with open(fn_x_valid_norm, 'rb') as f:
            x_valid_norm = pickle.load(f)

        with open(fn_y_valid, 'rb') as f:
            y_valid = pickle.load(f)
    else:
        x_train_norm, y_train, x_valid_norm, y_valid = prep_data(data_dir, split_thr=split_thr)

    return x_train_norm, y_train, x_valid_norm, y_valid

def create_time_steps(length):
    return list(range(-length, 0))

def show_plot(plot_data, delta, title):
    '''
    plot_data = [x, y_true, y_pred]
    '''
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])

    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                    label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

def music_generate(start_seq, sr=sr, full_len=full_len):
    '''
    Input:
    - start_seq: numpy array of initial music fragment
    - sr       : sample rate
    - full_len : desired output music length in seconds
    Output:
    - out_seq  : numpy array of generated music sequence
    '''

    # the start_seq should has longer than the "win"
    assert start_seq.shape[0] >= win

    # total length of music to be generated (length of the output numpy array)
    total_len = full_len * sr
    out_seq = np.zeros((total_len))

    # generate one sample per time by using the last win
    x = start_seq[-win:]
    for i in range(total_len):
        out_seq[i] = model_base_mean(x)
        x          = shift(x, -1)
        x[-1]      = out_seq[i]
    
    return out_seq