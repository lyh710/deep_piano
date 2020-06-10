:: Win10
:: FFmpeg: https://ffmpeg.org/download.html#build-windows

SET data_dir=data
for /r %%i in (%data_dir%\*.mp3) do FFmpeg\bin\ffmpeg -i %%i -acodec pcm_s16le -ac 1 -ar 16000 %data_dir%\%%~ni.wav