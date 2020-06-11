:: Win10
:: FFmpeg: https://ffmpeg.org/download.html#build-windows

SET data_dir=%1

for %%f in (%data_dir%\*.mp3) do FFmpeg\bin\ffmpeg -i %%f -acodec pcm_s16le -ac 1 -ar 16000 %data_dir%\%%~nf.wav