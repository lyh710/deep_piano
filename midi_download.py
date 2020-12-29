import warnings
warnings.filterwarnings('ignore')

from bs4 import BeautifulSoup, SoupStrainer
import requests, os, shutil, progressbar

midi_dir = os.path.join(os.getcwd(), 'midi')
if os.path.isdir(midi_dir):
        shutil.rmtree(midi_dir)
        os.mkdir(midi_dir)
else:
    os.mkdir(midi_dir)

# browse http://midi.midicn.com/ and manually determine a taqrget page
url = "http://midi.midicn.com/2000/06/06/%E5%8F%A4%E5%85%B8%E9%9F%B3%E4%B9%90MIDI"

page = requests.get(url)    
data = page.text
soup = BeautifulSoup(data)

with progressbar.ProgressBar(max_value=progressbar.UnknownLength) as bar:
    midi_count = 0
    for link in soup.find_all('a'):
        sub_url = link.get('href')
        if sub_url is not None and sub_url[-4:] == '.mid':
            r = requests.get(sub_url, allow_redirects=True)
            midi_name = sub_url.split('/')[-1]
            open(os.path.join(os.getcwd(), 'midi', midi_name), 'wb').write(r.content)
            midi_count += 1
            bar.update()

print(midi_count)