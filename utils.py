from music_config import music_config
import os, shutil, progressbar
import numpy as np
from music21 import converter, instrument, note, chord, stream, duration

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

def midi_is_valid(midi_fn):
    '''
    return True if:
    - The midi can be parsed by music21 package
    - And Piano being one of the instrument partition, or the midi has notes in a flat structure
    '''
    isValid = True
    
    try:
        midi = converter.parse(midi_fn)

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            if 'PIANO' not in [str(s.getInstrument()).upper() for s in s2]:
                isValid = False
        except: # file has notes in a flat structure
            isValid = True

    except:
        isValid = False

    return isValid

def split_midi_train_test(midi_dir):
    '''
    Input:
    - midi_dir: downloaded midi files
    Output:
    - a folder named "train", which contains all except one midi file
    - a folder named "test", which contains only one random selected midi file
    - all the "train" and "test" midi files should be "valid"
    '''

    # cache all the midi to a folder named "piano" if valid
    piano_dir = os.path.join(midi_dir, 'piano')
    if os.path.isdir(piano_dir):
        shutil.rmtree(piano_dir)
        os.mkdir(piano_dir)
    else:
        os.mkdir(piano_dir)
    
    i = 0
    with progressbar.ProgressBar(max_value=len(os.listdir(midi_dir))) as bar:
        for fn in os.listdir(midi_dir):
            if fn.endswith('.mid') and midi_is_valid(os.path.join(midi_dir,fn)):
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
    - return below format:
        - if single note: pitch+duration+offset
        - if single cord: note*note*note*chord_offset (note in above format) 
    '''
    
    notes = []

    midi = converter.parse(midi_fn)

    notes_to_parse = None
    try: # file has instrument parts
        s2 = instrument.partitionByInstrument(midi)
        all_instruments = [str(s.getInstrument()).upper() for s in s2]
        # print(all_instruments)
        piano_index = all_instruments.index('PIANO')
        # print(piano_index)
        notes_to_parse = s2.parts[piano_index].recurse() 
    except: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            single_note = '+'.join([str(element.pitch), str(element.duration.quarterLength), str(element.offset)])
            notes.append(single_note)
        elif isinstance(element, chord.Chord):
            single_cord = '*'.join(['+'.join([str(n.pitch), str(n.duration.quarterLength), str(n.offset)]) for n in element.notes])
            single_cord += '*'
            single_cord += str(element.offset)
            notes.append(single_cord)

    # return notes
    # assemble the list of notes into text-like format string, for subsequent text-based DNN process
    return ' '.join(notes)

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac

def pattern_2_note(note_pattern):
    '''
    pattern = pitch+duration+offset
    '''
    element = note_pattern.split('+')

    new_note = note.Note(element[0])
    new_note.storedInstrument = instrument.Piano()

    d = duration.Duration()
    d.quarterLength = convert_to_float(str(element[1]))
    new_note.duration = d

    new_note.offset = convert_to_float(str(element[2]))

    return new_note

def pattern_2_chord(chord_pattern):
    '''
    pattern = note*note*note*offset
    '''
    note_patterns = chord_pattern.split('*')[:-1]
    chord_offset  = chord_pattern.split('*')[-1]
    new_notes = []
    for note_pattern in note_patterns:
        new_notes.append(pattern_2_note(note_pattern))

    new_chord = chord.Chord(new_notes)
    new_chord.offset = convert_to_float(str(chord_offset))
    return new_chord

def notes_2_midi(prediction_output, midi_fn):
    '''
    prediction_output: list of notes generated, in below format:
        - if single note: pitch+duration+offset
        - if single cord: note*note*note (note in above format) 
    midi_fn          : absolute path of the midi file to write to
    output the notes to the midi_fn
    '''
    output_notes = []
    
    # notes = prediction_output
    patterns = prediction_output.split(' ')

    # create single_note and single_cord objects based on the values generated by the model
    for pattern in patterns:
        # pattern is a chord
        if ('*' in pattern):
            output_notes.append(pattern_2_chord(pattern))
        # pattern is a note
        else:
            output_notes.append(pattern_2_note(pattern))
    
    midi_stream = stream.Stream(output_notes)
    
    if os.path.exists(midi_fn):
        os.remove(midi_fn)
    midi_stream.write('midi', fp=midi_fn)

def midi_2_csv(train_dir, out_fn):
    '''
    given a folder of midi files (train_dir)
    convert each midi file into text-like notes, and output all notes to a csv file (out_fn), per midi file per line
    '''
    if os.path.exists(out_fn):
        os.remove(out_fn)

    with open(out_fn, 'a') as file:
        i = 0
        with progressbar.ProgressBar(max_value=len(os.listdir(train_dir))) as bar:
            for midi_fn in os.listdir(train_dir):
                if midi_fn.endswith('.mid'):
                    notes = midi_2_notes(os.path.join(train_dir, midi_fn))
                    file.write(notes)
                    file.write('\n')
                
                bar.update(i)
                i += 1

def df_split(df, split = 0.1):
    '''
    Split to 1-split : split
    '''
    if 'split' in df.columns:
        df.drop('split', axis=1, inplace=True)
    
    df['split'] = 'train'
    df.loc[np.random.choice(df[df['split']=='train'].index, int(df.shape[0]*valid_split)), 'split'] = 'valid'
    
    return df

def pkl_dump(data, fn):
    if os.path.exists(fn):
        os.remove(fn)
    with open(fn, 'wb') as f:
        pickle.dump(data, f, protocol=4)
        
def pkl_load(fn):
    if not os.path.exists(fn):
        print('data not available, check')
        exit()
    else:
        with open(fn, 'rb') as f:
            return pickle.load(f)