import warnings
warnings.filterwarnings('ignore')

from bs4 import BeautifulSoup, SoupStrainer
import requests, os, shutil, progressbar

def get_midi(input_url, midi_dir):
    
    # if output dir not exist, create one
    if not os.path.isdir(midi_dir):
        os.mkdir(midi_dir)
    try:
        page = requests.get(input_url)    
        data = page.text
        soup = BeautifulSoup(data)

        with progressbar.ProgressBar(max_value=progressbar.UnknownLength) as bar:
            midi_count = 0
            for link in soup.find_all('a'):
                sub_url = link.get('href')
                if sub_url is not None and sub_url[-4:] in ['.mid', '.MID', '.midi', '.MIDI']:
                    # print(sub_url)
                    r = requests.get(sub_url, allow_redirects=True)
                    midi_name = sub_url.split('/')[-1]
                    open(os.path.join(midi_dir, midi_name), 'wb').write(r.content)
                    midi_count += 1
                    bar.update()

        print(midi_count)
    except:
        None
