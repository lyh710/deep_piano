import numpy as np

def model_base_naive(x):
    '''
    use the last know history (x[-1]) to predict the next output
    '''
    return x[-1]

def model_base_mean(x):
    '''
    use the mean of history (x) to predict the next output
    '''
    return np.mean(x)