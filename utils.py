import subprocess, os

def mp3_to_wave(data_dir, bat_dir=os.path.join(os.getcwd(),'ffmpeg_mp3_to_wav.bat')):
    subprocess.call([bat_dir, data_dir])
