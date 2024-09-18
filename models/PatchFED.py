from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchFED_layers import series_decomp, Flatten_Head, positional_encoding

from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.SelfAttention_Family import FullAttention, ProbAttention
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters from input args 
        self.n_vars = configs.enc_in  # variables in the dataset 
        self.seq_len = configs.seq_len # seq of data input into the model look back (L)
        self.pred_len = configs.pred_len # prediction length of model main input 
        self.e_layers = configs.e_layers # 2 number of encoder layers 
        self.n_heads = configs.n_heads # 8 number of heads 
        self.d_model = configs.d_model # 512 size of model
        self.d_ff = configs.d_ff # size of ff layer 
        self.dropout = configs.dropout #0.05 
        self.fc_dropout = configs.fc_dropout # 0.2 
        self.head_dropout = configs.head_dropout # 0 
        # patching
        self.patch_len = configs.patch_len # length of patch (P)
        self.stride = configs.stride # stride of patch (S)
        self.padding_patch = configs.padding_patch # pad the patch? end or None
        #fedformer from input args 
        self.version = configs.version #type of transform fourier or wavelets
        self.mode_select = configs.mode_select #random or low selection of transformation
        self.modes = configs.modes # number modes for transformation 
        self.label_len = configs.label_len # forcasting length in the decoder 
        self.output_attention = configs.output_attention # output attention in encoder 
        
        # Decomp for the decoder input 
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Patching 
        self.patch_num = int((self.seq_len - self.patch_len)/self.stride + 1)
        if self.padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
            self.patch_num += 1
        print("patch num = {}".format(self.patch_num))


        self.W_P = nn.Linear(self.patch_len, self.d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.model_proj = nn.Linear(self.n_vars, self.d_model)
        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, self.patch_num, self.d_model)
    
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, self.d_model, configs.embed, configs.freq,
                                                  self.dropout)


        # Residual dropout
        self.dropout_func = nn.Dropout(self.dropout)

        # FED transformation based on version wavelets or fourier
        if configs.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=self.d_model, L=configs.L, base=configs.base)
            decoder_self_att = MultiWaveletTransform(ich=self.d_model, L=configs.L, base=configs.base)
            decoder_cross_att = MultiWaveletCross(in_channels=self.d_model,
                                                  out_channels=self.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=self.modes,
                                                  ich=self.d_model,
                                                  base=configs.base,
                                                  activation=configs.cross_activation)
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len,
                                            #n_heads = self.n_heads,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select)
            decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len//2+self.pred_len,
                                            #n_heads = self.n_heads,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                      out_channels=configs.d_model,
                                                      seq_len_q=self.seq_len//2+self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      #n_heads = self.n_heads,
                                                      modes=configs.modes,
                                                      mode_select_method=configs.mode_select,
                                                      activation =configs.cross_activation)


        # Flatten Head from the PatchTST

        def create_pretrain_head(self, head_nf, vars, dropout):
            return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))
    
        self.head_nf = self.d_model * self.patch_num
        self.pretrain_head = False # input into the backbone of PatchTST set to default of False 
        self.head_type = head_type
        self.individual = False # input into the backbone of PatchTST set to default of False 

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, self.n_vars, self.fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, self.pred_len, head_dropout=self.head_dropout)


        #define the Encoder and decoder based on FEDformer
        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2)) # not used pnly for info - where does this come in 
        dec_modes = int(min(configs.modes, (configs.seq_len//2+configs.pred_len)//2)) # not used only for info - where does this come in 
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,         
            enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):                 # x_enc: [bs x nvars x seq_len], 
                                                                            

        # decomp x_enc for dec inputs 
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1).to(device)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len)).to(device)

        x_enc = x_enc.permute(0,2,1).to(device) 

        # do patching of the enc input
        if self.padding_patch == 'end':
            x_enc = self.padding_patch_layer(x_enc)
        # patching 

        x_enc = x_enc.unfold(dimension=-1, size=self.patch_len, step=self.stride).to(device)                    # x_enc: [bs x nvars x patch_num x patch_len]
        x_enc = x_enc.permute(0,1,3,2)                                                              # x_enc: [bs x nvars x patch_len x patch_num]
        


        n_vars = x_enc.shape[1]
        # Project x_enc into the model dimension space 
        x_enc = x_enc.permute(0,1,3,2)                                                              # x_enc: [bs x nvars x patch_num x patch_len]

        x_enc = self.W_P(x_enc)                                                                     # x_enc: [bs x nvars x patch_num x d_model]

        # channel independece for x_enc
        x_enc = torch.reshape(x_enc, (x_enc.shape[0]*x_enc.shape[1],x_enc.shape[2],x_enc.shape[3]))         # x_enc: [bs * nvars x patch_num x d_model]
        
        #x_enc embedding + dropout
        x_enc = self.dropout_func(x_enc + self.W_pos).to(device)                                                        # x_enc: [bs * nvars x patch_num x d_model]
        

        # Encoder
        enc_out, attns = self.encoder(x_enc, attn_mask=enc_self_mask)                             # attns may need to be reshaped if change sel.output_attention in args input


        enc_out = torch.reshape(enc_out, (-1,n_vars,enc_out.shape[-2],enc_out.shape[-1])).to(device)                 # enc_out: [bs x nvars x patch_num x d_model]
        

        enc_out = enc_out.permute(0,1,3,2)                                                               # enc_out: [bs x nvars x d_model x patch_num]

        # apply flatten head
        enc_out = self.head(enc_out)                                                                    # enc_out: [bs x nvars x target_window] 
 
        enc_out = enc_out.permute(0,2,1)

        enc_out = self.model_proj(enc_out)

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)


        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                trend=trend_init)
        
        # final
        dec_out = trend_part + seasonal_part


        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]




