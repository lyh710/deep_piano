# deep_piano
Piano music generation using deep learning.

## Aim
We believe there are largely 2 ways to generate music:
- Generate the sound waves, which we tried and failed (https://github.com/lyh710/deep_piano/tree/music_as_time_series).
- Generate notation (sheet music), which we focused on here.

## Dev env
1. Windows 10 Home, i7 Core, 16GB ram (most subsequent steps should still hold true if with Mac/Linux, but some will need to be modified, such as the conda env setup batch).
2. NVIDIA GeForce GTX 1080 with Max-Q, 8GB (Google CoLab would be a good alternative).
3. conda env setup: conda_dnn_gpu_setup.bat (Ananconda3).
4. We expect **tf.config.list_physical_devices('GPU')** should return **True** after above setup.

## High level idea
By converting music files into music-sheets (notations), we can empploy Deep-Learning technique that has been widely used in the NLP domain to generate music. The idea is largely similar to text generation with Deep-Learning, in which LSTM or other more sophisticated architecture can be employed.

## Music files (midi) handling
This is mainly done by the music21 package: https://web.mit.edu/music21/doc/index.html.

## Ref
- https://www.midiworld.com/classic.htm
- https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
- https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/