from utils.tools import dotdict
import torch


def set_args(model_num, data_name, seq_len, model_name, root_path, data_path, model_id_name, random_seed, pred_len, target, full_start_time):
    args = dotdict()

    #inputs to this function 
    args.seq_len = seq_len # input sequence length of Informer encoder, fixed window size, look back window
    args.model = model_name # model of experiment, options: [informer, informerstack, informerlight(TBD)]
    args.model_num = model_num
    args.root_path = root_path# root path of data file
    args.data_path = data_path # data file
    args.data_name = data_name # name of data set
    args.random_seed  = random_seed #****
    args.pred_len = pred_len # prediction sequence length, output window 
    args.target = target # target feature in S or MS task ****** based on the dataset
    args.model_id = model_id_name +str(args.seq_len)+str(args.pred_len)
    args.is_training = 1
    args.full_start_time = full_start_time
    
    # data 
    args.data = 'custom' # data type ****
    args.features = 'M' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
    args.freq = 'h'  # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
    args.checkpoints = 'checkpoints/' # location of model checkpoints
    #args.mix = True
    args.padding = 0
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]
    
    # forcasting 
    args.label_len = 48 # start token length of decoder
    
    # define model parameters
    if data_name == 'ECL':
        args.enc_in = 321 # encoder input size 321 based on the dataset 
        args.dec_in = 321 # decoder input size 321 
        args.c_out = 321 # output size 
    elif data_name == "PJM":
        args.enc_in = 7 # encoder input size 321 based on the dataset 
        args.dec_in = 7 # decoder input size 321 
        args.c_out = 7 # output size 

    args.d_model = 512 # dimension of model 128
    args.n_heads = 8 # num of heads 16
    args.e_layers = 2 # num of encoder layers 3
    args.d_layers = 1 # num of decoder layers 2
    args.d_ff = 2048 # dimension of fcn in model
    args.dropout = 0.05 # dropout
    args.fc_dropout =  0.2
    args.head_dropout = 0
    args.embed_type =2 # help='0: default, 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    args.moving_avg = 24 #[12, 24] # window size of moving average') #autoformer 25, FED -24
    args.factor = 3 # probsparse attn factor  5 
    args.distil = True # whether to use distilling in encoder
    args.attn = 'prob' # attention used in encoder, options:[prob, full]
    args.embed = 'timeF' # time features encoding, options:[timeF*, fixed, learned]
    args.activation = 'gelu' # activation
    args.output_attention = False # whether to output attention in encoder
    


    # training and optimsation 
    args.train_epochs = 20 #20 max 
    args.patience = 3 #3
    args.itr = 1
    args.batch_size = 32
    args.loss = 'mse'  
    args.patch_len = 16
    args.stride = 8
    args.des = 'Exp' # change to capital ****
    args.lradj = 'type1' # adjuseted learning rate ****
    args.pct_start = 0.2 #******
    args.learning_rate = 0.0001
    args.use_amp = False # whether to use automatic mixed precision training  
    args.num_workers = 10  #0
    args.padding_patch = 'end' # 'None: None; end: padding on the end')
    args.patch_modes = 32 # args.patch_len

    args.detect_nan = True

    # supplementary config for FEDformer model
    args.version ='Wavelets'  #'for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]'
    args.mode_select='random' #for FEDformer, there are two mode selection method, options: [random, low]
    args.modes=64 # help='modes to be selected random 64')
    args.L=3 #help='ignore level')
    args.base='legendre' # help='mwt base'
    args.cross_activation ='softmax' #help='mwt cross atention activation function tanh or softmax')

    #headflatten
    args.pretrain_head = False, 
    args.head_type = 'flatten'
    args.individual = False
    
    # gpu   
    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = '0,1,2,3' 
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    
    # setting 
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.model_num, args.data_name, args.features,
                    args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, 0)
    
    print(setting)
    
    return args, setting
    
    