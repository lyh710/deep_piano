from music_config import music_config
import os, shutil, progressbar
import numpy as np
from music21 import converter, instrument, note, chord, stream

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

def midi_is_piano(midi_fn):
    '''
    return if the midi has Piano being one of the instrument partition
    '''
    isPiano = True
    
    try:
        midi = converter.parse(midi_fn)

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            if 'PIANO' not in [str(s.getInstrument()).upper() for s in s2]:
                isPiano = False
        except: # file has notes in a flat structure
            isPiano = True

    except:
        isPiano = False

    return isPiano

def split_midi_train_test(midi_dir):
    '''
    Input:
    - midi_dir: midi files location
    Output:
    - a folder named "train", which contains all except one midi file
    - a folder named "test", which contains only one random selected midi file
    - all the "train" and "test" midi files should contain 'PIANO' or being a flat structure
    '''

    # cache all the midi to a folder named "piano" if there is piano partition in the midi
    piano_dir = os.path.join(midi_dir, 'piano')
    if os.path.isdir(piano_dir):
        shutil.rmtree(piano_dir)
        os.mkdir(piano_dir)
    else:
        os.mkdir(piano_dir)
    
    i = 0
    with progressbar.ProgressBar(max_value=len(os.listdir(midi_dir))) as bar:
        for fn in os.listdir(midi_dir):
            if fn.endswith('.mid') and midi_is_piano(os.path.join(midi_dir,fn)):
                src_dir = os.path.join(midi_dir,fn)
                dst_dir = os.path.join(piano_dir, fn)
                shutil.move(src_dir, dst_dir)
            bar.update(i)
            i += 1

    # split all the piano midi into "train" and "test" by leave one out
    midi_all  = os.listdir(piano_dir)
    
    midi_test  = [np.random.choice(midi_all)]

    midi_train = midi_all.copy()
    midi_train.remove(midi_test[0])

    test_dir = os.path.join(piano_dir, 'test')
    if os.path.isdir(test_dir):
        shutil.rmtree(test_dir)
        os.mkdir(test_dir)
    else:
        os.mkdir(test_dir)

    for fn in midi_test:
        src_dir = os.path.join(piano_dir,fn)
        dst_dir = os.path.join(test_dir, fn)
        shutil.move(src_dir, dst_dir)

    train_dir = os.path.join(piano_dir, 'train')
    if os.path.isdir(train_dir):
        shutil.rmtree(train_dir)
        os.mkdir(train_dir)
    else:
        os.mkdir(train_dir)
        
    for fn in midi_train:
        src_dir = os.path.join(piano_dir,fn)
        dst_dir = os.path.join(train_dir, fn)
        shutil.move(src_dir, dst_dir)

def midi_2_notes(midi_fn):
    '''
    - midi_fn: full path to the midi music file
    - return notes, which is a list of all the elements of all the compositions:
        - If we process a note, we will store in the list a string representing the pitch (the note name) and the octave.
        - If we process a chord (Remember that chords are set of notes that are played at the same time) we will store 
          a different type of string with numbers separated by dots. Each number represents the pitch of a chord note.
        - We are not considering yet time offsets of each element. So all the notes and chords will have the same duration. 
    '''
    notes = []

    midi = converter.parse(midi_fn)

    notes_to_parse = None
    try: # file has instrument parts
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse() 
    except: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes