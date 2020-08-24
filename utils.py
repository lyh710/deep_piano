import subprocess, os, shutil, random
import numpy as np
import librosa, math, pickle, progressbar

import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from music_config import music_config
from scipy.ndimage.interpolation import shift
from sklearn.preprocessing import MinMaxScaler

sr          = music_config['sr']
win         = music_config['win']
stride      = music_config['stride']
single_step = music_config['single_step']
target_size = music_config['target_size']
norm_data   = music_config['norm_data']
split_thr   = music_config['split_thr']
full_len    = music_config['full_len']
lr_start    = music_config['lr_start']
batch_size  = music_config['batch_size']
epochs      = music_config['epochs']
verbose     = music_config['verbose']
loss        = music_config['loss']
reg_l2      = music_config['reg_l2']
dropout     = music_config['dropout']
clipvalue   = music_config['clipvalue']

def show_config(music_config):
    for key in music_config:
        print(key, ': ', music_config[key])

def mp3_to_wave(data_dir, bat_dir=os.path.join(os.getcwd(),'ffmpeg_mp3_to_wav.bat')):
    '''
    Input:
    - data_dir: mp3 files location
    - bat_dir: windows batch file that calls ffmpeg.exe
    Output:
    mp3 files will be converted to wav and saved in the same directory, i.e. filename.mp3 --> filename.wav
    '''
    
    subprocess.call([bat_dir, data_dir])

def gen_scaler(data_dir):
    '''
    randomly select one music file from 'train' folder, to fit the standariser
    which will be applied to all other music file (train & test) as preparation step
    
    note: this is different to standarise each music file to its own min_max,
          but normalise all music to the random target. 
          Such way may lead to certain music data still NOT within (0, 1) after normalisation, 
          but may provide a better cotrast across different music files, that might be helpful for DNN training
    '''

    target_fn  = np.random.choice([fn for fn in os.listdir(os.path.join(data_dir, 'train')) if fn.endswith('.wav')])
    target_dir = os.path.join(data_dir, 'train', target_fn)

    x,_        = librosa.load(target_dir, sr=sr)

    scaler     = MinMaxScaler()
    scaler.fit(x.reshape(-1,1))

    fn     = os.path.join(data_dir,'train','scaler.pkl')
    if os.path.exists(fn):
        os.remove(fn)
    with open(fn, 'wb') as f:
        pickle.dump(scaler, f, protocol=4)

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

    # generate the (0, 1) range standariser with one random train file
    gen_scaler(data_dir)

def data_normalise(data_dir, x):
    '''
    scale to (0,1)
    '''
    # scaler = MinMaxScaler()
    # x_norm = scaler.fit_transform(x.reshape(-1,1))

    # scaler
    fn     = os.path.join(data_dir,'train','scaler.pkl')
    if not os.path.exists(fn):
        print('scaler not available, check')
        exit()
    else:
        with open(fn, 'rb') as f:
            scaler = pickle.load(f)
            x_norm = scaler.transform(x.reshape(-1,1))

            return x_norm.flatten()

def shuffle_together(a,b):
    '''
    shuffle list a and list b at the same time (same order)
    '''
    c  = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c) 
    return list(a), list(b)

def music_to_x_y(data_dir, fn, norm_data=norm_data):
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

    if norm_data:
        dataset = data_normalise(data_dir, dataset)
    
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

            if single_step:
                labels.append(dataset[i])
            else:
                labels.append(dataset[i:i+target_size])
            # print('start at %s, end at %s' % (i-win, i))

        # shuffle x and y correspondingly
        data, labels = shuffle_together(data,labels)

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
            data, labels = music_to_x_y(data_dir, os.path.join(data_dir,'train',fn))
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

def prep_data(data_dir, split_thr=split_thr):
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

    return x_train, y_train, x_valid, y_valid

def load_data(data_dir, split_thr=split_thr):
    
    # x_train
    fn_x_train = os.path.join(data_dir,'train','x_train.pkl')

    # y_train
    fn_y_train      = os.path.join(data_dir,'train','y_train.pkl')

    # x_valid
    fn_x_valid = os.path.join(data_dir,'train','x_valid.pkl')

    # y_valid
    fn_y_valid      = os.path.join(data_dir,'train','y_valid.pkl')

    if (os.path.exists(fn_x_train)) and (os.path.exists(fn_y_train)) and (os.path.exists(fn_x_valid)) and (os.path.exists(fn_y_valid)):
        with open(fn_x_train, 'rb') as f:
            x_train = pickle.load(f)
        
        with open(fn_y_train, 'rb') as f:
            y_train = pickle.load(f)

        with open(fn_x_valid, 'rb') as f:
            x_valid = pickle.load(f)

        with open(fn_y_valid, 'rb') as f:
            y_valid = pickle.load(f)
    else:
        x_train, y_train, x_valid, y_valid = prep_data(data_dir, split_thr=split_thr)

    return x_train, y_train, x_valid, y_valid

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

def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in  = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history), label='History')
    plt.plot(np.arange(num_out), np.array(true_future), 'bo', label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out), np.array(prediction), 'ro', label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()

def music_generate(data_dir, model_version, start_seq, sr=sr, full_len=full_len, target_size=target_size):
    '''
    Input:
    - start_seq  : numpy array of initial music fragment
    - sr         : sample rate
    - full_len   : desired output music length in seconds
    - target_size: number of time-steps to generate per step
    Output:
    - out_seq  : numpy array of generated music sequence
    '''

    # the start_seq should be no less than the "win"
    assert start_seq.shape[0] >= win

    # start_seq
    fn            = os.listdir(os.path.join(data_dir,'test'))[0]
    fn            = os.path.join(data_dir, 'test', fn)
    x_test,_      = music_to_x_y(data_dir, fn, norm_data=norm_data)
    i_rand        = np.random.choice(x_test.shape[0])
    start_seq     = x_test[i_rand,:]

    # model
    out_dir       = os.path.join(os.getcwd(), 'models', model_version)
    fn            = os.path.join(out_dir, model_version+'_checkpoint_epoch.hdf5')
    model         = load_model(fn)
    print(model.summary())

    # total length of music to be generated (length of the output numpy array)
    total_len      = full_len * sr
    generate_steps = math.ceil(total_len/target_size)
    # print(generate_steps)
    out_seq        = []

    # generate target_size samples per time by using the last win
    x     = start_seq[-win:]
    x_dnn = tf.expand_dims(x, 0)
    x_dnn = tf.expand_dims(x_dnn, 2)

    with progressbar.ProgressBar(max_value=generate_steps) as bar:
        for i in range(generate_steps):
            pred             = (model.predict(x_dnn)).tolist()[0]
            out_seq          = out_seq + pred
            x                = shift(x, -target_size)
            x[-target_size:] = pred
            x_dnn            = tf.expand_dims(x, 0)
            x_dnn            = tf.expand_dims(x_dnn, 2)
            bar.update(i)
    
    return np.array(out_seq)