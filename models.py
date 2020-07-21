import numpy as np
from time import time
from utils import * 
import shutil

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPool1D, LSTM, Dense, Flatten, Dropout, Bidirectional, BatchNormalization, Lambda
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import TensorBoard, Callback, ModelCheckpoint, LambdaCallback, LearningRateScheduler
from tensorflow.keras.losses import Huber
from tensorflow import squeeze, expand_dims

def model_base_naive(x):
    '''
    use the last know history (x[-1]) to predict the next output
    '''
    return x[-1]

def model_base_mean(x):
    '''
    use the mean of history (x) to predict the next output
    '''
    return np.mean(x)

def model_dnn(x_train, y_train, x_valid, y_valid):
    
    show_config(music_config)

    # reshape the input tensor (x) to be of shape=(sample, window, 1)
    x_train = tf.expand_dims(x_train, 2)
    x_valid = tf.expand_dims(x_valid, 2)

    # specify model version
    model_version  = 'model.'+str(time())

    # Prepar folder to output model
    out_dir = os.path.join(os.getcwd(), 'models', model_version)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    # Callbacks: checkpoint
    fn = os.path.join(out_dir, model_version+'_checkpoint_epoch.hdf5')
    checkpoint_epoch = ModelCheckpoint(fn, monitor='val_loss', save_best_only=True, mode = 'min')

    # set Tensorboard
    tensorboard = TensorBoard(log_dir=os.path.join(os.getcwd(), 'logs', model_version))

    # model architecture
    model   = Sequential()
    # model.add(Conv1D(filters=32, kernel_size=(3), activation='relu', kernel_regularizer=regularizers.l2(reg_l2), batch_input_shape=(None, win, 1)))
    # model.add(Conv1D(filters=32, kernel_size=(3), activation='relu', kernel_regularizer=regularizers.l2(reg_l2)))
    # model.add(BatchNormalization(axis=2))
    # model.add(MaxPool1D(pool_size=2))
    # model.add(Conv1D(filters=64, kernel_size=(3), activation='relu', kernel_regularizer=regularizers.l2(reg_l2)))
    # model.add(Conv1D(filters=64, kernel_size=(3), activation='relu', kernel_regularizer=regularizers.l2(reg_l2)))
    # model.add(BatchNormalization(axis=2))
    # model.add(MaxPool1D(pool_size=2))
    # model.add(Conv1D(filters=128, kernel_size=(3), activation='relu', kernel_regularizer=regularizers.l2(reg_l2)))
    # model.add(Conv1D(filters=128, kernel_size=(3), activation='relu', kernel_regularizer=regularizers.l2(reg_l2)))
    # model.add(BatchNormalization(axis=2))
    # model.add(MaxPool1D(pool_size=2))
    # model.add(Conv1D(filters=256, kernel_size=(3), activation='relu', kernel_regularizer=regularizers.l2(reg_l2)))
    # model.add(Conv1D(filters=256, kernel_size=(3), activation='relu', kernel_regularizer=regularizers.l2(reg_l2)))
    # model.add(BatchNormalization(axis=2))
    # model.add(MaxPool1D(pool_size=2))
    # model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=dropout)))
    # model.add(Bidirectional(LSTM(128, dropout=dropout)))
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(win, 1)))
    # model.add(LSTM(int(target_size), return_sequences=True, activation='relu'))
    model.add(LSTM(100, activation='relu'))
    # model.add(Dropout(dropout))
    model.add(Dense(target_size))
    print(model.summary())

    # optimizer & compile
    optimizer = optimizers.Adam(learning_rate=lr_start)
    # optimizer = optimizers.RMSprop(learning_rate=lr_start, clipvalue=clipvalue)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # train
    history = model.fit(  x_train
                        , y_train
                        , batch_size=batch_size
                        , epochs=epochs
                        , validation_data=(x_valid, y_valid)
                        , verbose=verbose
                        , callbacks=[checkpoint_epoch, tensorboard])

    # output the model to disk
    fn = os.path.join(out_dir, model_version+'.hdf5')
    if os.path.exists(fn):
        os.remove(fn)
    model.save(fn)

    # output the history to disk
    fn = os.path.join(out_dir, model_version+'.history.pkl')
    if os.path.exists(fn):
        os.remove(fn)
    with open(fn, 'wb') as f:
        pickle.dump(history.history, f)
    
    # Save the script
    src_fn = os.path.join(os.getcwd(), 'models.py')
    trg_fn = os.path.join(out_dir, 'models.py')
    if os.path.exists(trg_fn):
        os.remove(trg_fn)
    shutil.copy(src_fn, trg_fn)

    # Save the config (data)
    src_fn = os.path.join(os.getcwd(), 'music_config.py')
    trg_fn = os.path.join(out_dir, 'music_config.py')
    if os.path.exists(trg_fn):
        os.remove(trg_fn)
    shutil.copy(src_fn, trg_fn)

    return model