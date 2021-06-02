import warnings
warnings.filterwarnings('ignore')

import os, math
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from utils import *
from models import *

show_config(dnn_config)

# GPU ready
assert tf.test.is_gpu_available()

#----------------------------------------------------------------------------

# load tokenizer
fn        = os.path.join(os.getcwd(), 'data','tokenizer.pkl')
tokenizer = pkl_load(fn)
print('tokenizer loaded')

# load total_words
fn        = os.path.join(os.getcwd(), 'data','total_words.pkl')
total_words = pkl_load(fn)
print('total_words loaded')

# load max_sequence_len
fn        = os.path.join(os.getcwd(), 'data','max_sequence_len.pkl')
max_sequence_len = pkl_load(fn)
print('max_sequence_len loaded')

#----------------------------------------------------------------------------

# load train
fn = os.path.join(os.getcwd(), 'data', 'df_train.csv')
assert os.path.exists(fn)
df_train = pd.read_csv(fn)

corpus_train               = df_train['notes'].tolist()
input_sequences_train      = text_2_token_seq(tokenizer, corpus_train)
x_train, y_train, max_sequence_len_train = generate_padded_sequences(input_sequences_train, total_words, max_sequence_len=max_sequence_len)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[-1], 1)
# normalize input
# x_train = x_train / float(total_words)

# load valid
fn = os.path.join(os.getcwd(), 'data', 'df_valid.csv')
assert os.path.exists(fn)
df_valid = pd.read_csv(fn)

corpus_valid               = df_valid['notes'].tolist()
input_sequences_valid      = text_2_token_seq(tokenizer, corpus_valid)
x_valid, y_valid, max_sequence_len_valid = generate_padded_sequences(input_sequences_valid, total_words, max_sequence_len=max_sequence_len)
x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[-1], 1)
# x_valid = x_valid / float(total_words)

#----------------------------------------------------------------------------
# model define
model = get_model(max_sequence_len, total_words)
print(model.summary())

#----------------------------------------------------------------------------
# # Experimental train with 100 epochs, to find the optimal learning-rate with scheduler
# lr_start = 1e-08
# lr_schedule = LearningRateScheduler(lambda epoch: lr_start * 10**(epoch / 20))
# lr_optimizer = optimizers.Adam(learning_rate=lr_start)
# model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=lr_optimizer)
# lr_history = model.fit(  x_train, y_train
#                        , batch_size=128, epochs=100
#                        , validation_data=(x_valid, y_valid)
#                        , verbose = 1
#                        , callbacks=[lr_schedule])
# 
# import matplotlib.pyplot as plt
# plt.semilogx(lr_history['lr'], lr_history['loss'])
# plt.axis([1e-8, 1e-4, 0, 1])
# plt.savefig(os.path.join(os.getcwd(),'lr_schedule.png'))
# plt.show()

# ----------------------------------------------------------------------------
# Set model version
model_version = 'model.'+str(time())

# Setup folder to output model
model_dir = os.path.join(os.getcwd(), 'models')
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

out_dir = os.path.join(os.getcwd(), 'models', model_version)
if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)

# ----------------------------------------------------------------------------
# Callback: checkpoint
fn = os.path.join(out_dir, model_version+'_checkpoint_epoch.hdf5')
checkpoint_epoch = ModelCheckpoint(fn, monitor='val_loss', save_best_only=True, mode='min')

# Callback: Tensorboard (To-Do)

# Early Stop
es = EarlyStopping(  monitor='val_loss'
                   , min_delta=0
                   , patience=dnn_config['patience']
                   , verbose=0
                   , mode='auto'
                   , baseline=None
                   , restore_best_weights=False
                   )

# ----------------------------------------------------------------------------
history = model.fit(  x = x_train
                    , y = y_train
                    , batch_size=dnn_config['batch_size']
                    , epochs=dnn_config['epochs']
                    , validation_data=(x_valid, y_valid)
                    , verbose=dnn_config['verbose']
                    , callbacks=[checkpoint_epoch, es]
                    )

# ----------------------------------------------------------------------------
# output the model to disk
fn = os.path.join(out_dir, model_version+'.hdf5')
if os.path.exists(fn):
    os.remove(fn)
model.save(fn)

# output the history to disk
fn = os.path.join(out_dir, model_version+'.history.pkl')
pkl_dump(history.history, fn)

# Save the script
src_fn = os.path.join(os.getcwd(), 'models.py')
trg_fn = os.path.join(out_dir, 'models.py')
if os.path.exists(trg_fn):
    os.remove(trg_fn)
shutil.copy(src_fn, trg_fn)

# Save the config
src_fn = os.path.join(os.getcwd(), 'dnn_config.py')
trg_fn = os.path.join(out_dir, 'dnn_config.py')
if os.path.exists(trg_fn):
    os.remove(trg_fn)
shutil.copy(src_fn, trg_fn)