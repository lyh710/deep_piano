import os, shutil, progressbar, pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream, duration

from music_config import music_config

def show_config(config):
    for key in config:
        print(key, ': ', config[key])

def fn_is_midi(midi_fn):
    result = False

    if midi_fn.endswith('.mid') or midi_fn.endswith('.MID') or midi_fn.endswith('.midi') or midi_fn.endswith('.MIDI'):
        result = True
    
    return result

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

def split_midi_train_test(midi_dir, train_dir, test_dir):
    '''
    Input:
    - midi_dir: downloaded midi files
    Output:
    - a folder named "train", which contains all except one midi file
    - a folder named "test", which contains only one random selected midi file
    - all the "train" and "test" midi files should be "valid"
    '''

    if os.path.isdir(train_dir):
        shutil.rmtree(train_dir)
    os.mkdir(train_dir)

    if os.path.isdir(test_dir):
        shutil.rmtree(test_dir)
    os.mkdir(test_dir)
    
    valid_cnt = 0
    i = 0
    with progressbar.ProgressBar(max_value=len(os.listdir(midi_dir))) as bar:
        for fn in os.listdir(midi_dir):
            if fn_is_midi(fn) and midi_is_valid(os.path.join(midi_dir,fn)):
                src_dir = os.path.join(midi_dir,fn)
                dst_dir = os.path.join(train_dir, fn)
                shutil.copy(src_dir, dst_dir)
                valid_cnt = valid_cnt + 1
            bar.update(i)
            i += 1

    # split all the piano midi into "train" and "test" by leave one out
    midi_all  = os.listdir(train_dir)
    
    midi_test  = [np.random.choice(midi_all)]

    for fn in midi_test:
        src_dir = os.path.join(train_dir,fn)
        dst_dir = os.path.join(test_dir, fn)
        shutil.move(src_dir, dst_dir)

    if valid_cnt > 0:
        return True
    else:
        return False
    
def midi_2_notes(midi_fn, simple=True):
    '''
    - midi_fn: full path to the midi music file

    - return below format:

        - If simple==True: 
            - ignore duration and offset
            - chord will be the joint of each pitch (pitch*pitch*pitch)
            - Such simplification will:
                - significantly reduce the corpus/vocabulary size.
                - enable the program fit in lower memory and simple DNN schema.

        - If simple == False:
            - if single note: pitch+duration+offset
            - if single cord: note*note*note*chord_offset (note in above format) 
            - it will represent the music to higher quality, but demand significantly more coputing resource
            - cast bigger challenge to DNN schema/training/performance
        
        - Manually append 'STOP' at the end of the notes
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
            if simple:
                single_note = str(element.pitch)
            else:
                single_note = '+'.join([str(element.pitch), str(element.duration.quarterLength), str(element.offset)])

            notes.append(single_note)
        elif isinstance(element, chord.Chord):
            if simple:
                single_cord = str(element.notes[0].pitch)
                # single_cord = '*'.join([str(n.pitch) for n in element.notes])
            else:
                single_cord = '*'.join(['+'.join([str(n.pitch), str(n.duration.quarterLength), str(n.offset)]) for n in element.notes])
                single_cord += '*'
                single_cord += str(element.offset)
            
            notes.append(single_cord)

    # notes.append('STOP')
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

def pattern_2_note(note_pattern, simple=True):
    '''
    If simple==True:
        pattern = pitch
    else:
        pattern = pitch+duration+offset
    '''
    if simple:
        new_note                  = note.Note(note_pattern)
        new_note.storedInstrument = instrument.Piano()
        new_note.duration         = duration.Duration(quaterLength=music_config['quaterLength'])

    else:
        element = note_pattern.split('+')

        new_note = note.Note(element[0])
        new_note.storedInstrument = instrument.Piano()

        d = duration.Duration()
        d.quarterLength = convert_to_float(str(element[1]))
        new_note.duration = d

        new_note.offset = convert_to_float(str(element[2]))

    return new_note

def pattern_2_chord(chord_pattern, simple=True):
    '''
    If simple==True:
        pattern = note*note*note
    else:
        pattern = note*note*note*offset
    '''
    if simple:
        # new_chord                  = note.Note(chord_pattern)
        note_patterns = chord_pattern.split('*')
        new_notes = []
        for note_pattern in note_patterns:
            new_notes.append(pattern_2_note(note_pattern))
        
        new_chord = chord.Chord(new_notes)
        new_chord.storedInstrument = instrument.Piano()
        new_chord.duration         = duration.Duration(quaterLength=music_config['quaterLength'])
    else:
        note_patterns = chord_pattern.split('*')[:-1]
        chord_offset  = chord_pattern.split('*')[-1]
        new_notes = []
        for note_pattern in note_patterns:
            new_notes.append(pattern_2_note(note_pattern))

        new_chord = chord.Chord(new_notes)
        new_chord.offset = convert_to_float(str(chord_offset))
    return new_chord

def notes_2_midi(prediction_output, midi_fn, simple=True):
    '''
    prediction_output: list of notes generated, in below format:
    if simple == True:
        - if single note: pitch
        - if single cord: note
    else:
        - if single note: pitch+duration+offset
        - if single cord: note*note*note (note in above format) 
    midi_fn          : absolute path of the midi file to write to
    output the notes to the midi_fn
    '''
    output_notes = []
    offset       = 0
    
    patterns = prediction_output.split(' ')
    # remove the 'STOP' at the end, if any
    try:
        patterns.remove('STOP')
    except:
        None

    if simple:
        for pattern in patterns:
            # pattern is a chord
            if ('*' in pattern):
                note        = pattern_2_chord(pattern)
                note.offset = offset
                output_notes.append(note)
                offset      += music_config['offset']
            # pattern is a note
            else:
                note        = pattern_2_note(pattern)
                note.offset = offset
                output_notes.append(note)
                offset      += music_config['offset']
    else:
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

def midi_2_csv(train_dir, out_fn, small_f=True, small_size=music_config['small_size']):
    '''
    given a folder of midi files (train_dir)
    convert each midi file into text-like notes, and output all notes to a csv file (out_fn)
    all the notes from one single midi file will be broken down into multiple smaller notes, 100 notes each
    '''
    if os.path.exists(out_fn):
        os.remove(out_fn)

    with open(out_fn, 'a') as file:
        i = 0
        with progressbar.ProgressBar(max_value=len(os.listdir(train_dir))) as bar:
            for midi_fn in os.listdir(train_dir):
                if fn_is_midi(midi_fn):
                    notes = midi_2_notes(os.path.join(train_dir, midi_fn))
                    
                    # all the notes from one single midi file will be broken down into multiple smaller notes, 100 notes each
                    if small_f:
                        notes_big   = notes.split(' ')
                        small_notes = [notes_big[n:n+small_size] for n in range(0, len(notes_big), small_size)]
                        for sn in small_notes:
                            file.write(' '.join(sn))
                            file.write('\n')
                    else:
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
    df.loc[np.random.choice(df[df['split']=='train'].index, int(df.shape[0]*split)), 'split'] = 'valid'
    
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