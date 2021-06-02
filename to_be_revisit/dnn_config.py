dnn_config = {  'seq_length' : 100
              , 'lr_start'   : 1e-03
              , 'batch_size' : 256
              , 'gen_size'   : 1
              , 'epochs'     : 100
              , 'verbose'    : 1
              , 'loss'       : 'categorical_crossentropy'
              , 'reg_l2'     : 0.001
              , 'dropout'    : 0.2
              , 'clipvalue'  : 1.0
              , 'embed_dim'  : 100
              , 'lstm_units' : 100
              , 'patience'   : 5
            }
