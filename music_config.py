'''
- sr: sample rate of the music, default is 22050
- win: historical window size to predict the next note, default is 100 numpy sample (given sr), note this is NOT seconds
- stride: step length to move the hisotrical window from the beginning to the end of the music file, default is 10, note this is NOT seconds
- target_size: how many numerical sample to predict, default is 1 (note this is numpy sample, not seconds)
'''
music_config = {
                  'sr'          : 22050   
                , 'win'         : 1000    
                , 'stride'      : 100      
                , 'target_size' : 1      
                , 'split_rate'  : 0.9 
                , 'full_len'    : 180  # 180 seconds = 3 minutes
                }
