import warnings
warnings.filterwarnings('ignore')

from utils import *
from models import *

#--------- train -------------------------------------------
print('Train')
fn = os.path.join(os.getcwd(), 'data', 'df_train.csv')
assert os.path.exists(fn)
df_train = pd.read_csv(fn)

corpus_train = df_train['notes'].tolist()
print('corpus_train done')

# tokenizer
tokenizer, input_sequences_train, total_words = get_sequence_of_tokens(corpus_train)
fn     = os.path.join(os.getcwd(), 'data','tokenizer.pkl')
pkl_dump(tokenizer, fn)
print('tokenizer done')

# total_words
print('Vob size: {}'.format(total_words))
fn     = os.path.join(os.getcwd(), 'data','total_words.pkl')
pkl_dump(total_words, fn)
print('total_words done')

# pad sequence, and create (input, label) pairs
x_train, y_train, max_sequence_len = generate_padded_sequences(input_sequences_train, total_words, max_sequence_len=None)
print('Max input sequence length for train: {}'.format(max_sequence_len))
fn     = os.path.join(os.getcwd(), 'data','max_sequence_len.pkl')
pkl_dump(max_sequence_len, fn)