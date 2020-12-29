# deep_piano
Piano music generation using deep learning.

## Aim
There are largely 2 ways to generate music: one is to generate the sound waves, the latter is to generate notation (sheet music). We tried the first and failed (https://github.com/lyh710/deep_piano/tree/music_as_time_series), and moved on to the latter as shown here.

## Dev env
1. Windows 10 Home, i7 Core, 16GB ram (most subsequent steps should still hold true if with Mac/Linux, but some will need to be modified, such as the conda env setup batch)
2. NVIDIA GeForce GTX 1080 with Max-Q, 8GB
3. conda env setup: dnn_gpu_setup_test\conda_dnn_gpu_setup.bat (Ananconda3)

## High level idea
Build a deep-learning model, with the "Transformer" architecture, which can take an input sequence of music notation and predict/generate the target sequence.

## Model work flow
1. music data acquired: midi_download.py
    - MIDI format
    - search online and download for study/research purpose only (http://midi.midicn.com/)

2. Convert MIDI file to sheet-music (notations), and vice versa
    - Employ the music21 package (https://web.mit.edu/music21/doc/index.html)
