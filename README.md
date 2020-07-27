# deep_piano
Piano music generation using deep learning.

## Aim
Given a few piano notes as input (around 10 seconds), the program will generate a full piece of piano music (around 3 minutes).

## Dev env
1. Windows 10 Home, i7 Core, 16GB ram (most subsequent steps should still hold true if with Mac/Linux, but some will need to be modified, such as the conda env setup batch and the ffmpeg)
2. NVIDIA GeForce GTX 1080 with Max-Q, 8GB
3. conda env setup: dnn_gpu_setup_test\conda_dnn_gpu_setup.bat (Ananconda3)

## Model work flow
1. Music format ready:
    - convert mp3 to wav using ffmpeg: ffmpeg_mp3_to-wav.bat
    - ffmpeg executable file should be downloaded for win64 and placed in .\FFmpeg\bin\ffmpeg.exe
    - Note the conversion to wav is only required due to the piano music available happened to be mp3 format.
    - samples used being performance by Richard Clayderman (https://en.wikipedia.org/wiki/Richard_Clayderman)

2. Music to numeric data:
    - Python package librosa is employed, which will convert wav file into 1-D numpy array, given sample_rate (sr) as hyper-parameter
    - music length (in seconds) x sample_rate = numpy array length (1-D)

3. DNN based on Data
    - train towards the scenario of using the last N to generate the next 1 sample
    - LSTM + Dense
    - Loss = mse

4. DNN to generate music
    - short starting piece of piano melody (random segment from hold-out piece)

5. Data to Music
    - librosa

## Reference
1. https://www.coursera.org/learn/nlp-sequence-models by Andrew Ng from deeplearning.ai.
2. https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html
   https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-2.html
3. https://www.tensorflow.org/tutorials/structured_data/time_series
4. magenta: https://github.com/magenta/magenta
5. https://towardsdatascience.com/creating-a-pop-music-generator-with-the-transformer-5867511b382a 
   https://towardsdatascience.com/practical-tips-for-training-a-music-model-755c62560ec2
   https://towardsdatascience.com/a-multitask-music-model-with-bert-transformer-xl-and-seq2seq-3d80bd2ea08e
   https://towardsdatascience.com/how-to-remix-the-chainsmokers-with-a-music-bot-6b920359248c
