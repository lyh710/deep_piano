'''
- sr: sample rate of the music, default is 22050
- win: historical window size to predict the next note, default is 100 numpy sample (given sr), note this is NOT seconds
- stride: step length to move the hisotrical window from the beginning to the end of the music file, default is 10, note this is NOT seconds
- split_thr: split between Train & Test
- target_size: how many numerical sample to predict, default is 1 (note this is numpy sample, not seconds)
'''
music_config = {
                  'sr'          : 22050   
                , 'win'         : 100  
                , 'stride'      : 50
                , 'single_step' : False  
                , 'target_size' : 10
                , 'norm_data'   : True    
                , 'split_thr'   : 0.8
                , 'full_len'    : 180  # 180 seconds = 3 minutes
                , 'lr_start'    : 1e-05
                , 'batch_size'  : 128
                , 'epochs'      : 5
                , 'verbose'     : 1
                , 'loss'        : 'mse'
                , 'reg_l2'      : 0.001
                , 'dropout'     : 0.5
                , 'clipvalue'   : 1.0
                }
