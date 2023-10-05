import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):
    # Pytorch transformer module
    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5,args=None):
        super(TransformerModel, self).__init__()
        self.premlp = nn.Linear(ninp,nhid)
        self.pos_encoder = PositionalEncoding(ninp, dropout=0.0,max_len=75000)
        encoder_layers = TransformerEncoderLayer(nhid, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = nhid
        self.use_posemb = True

    def forward(self, src_inp, src_mask,timestamps=None):
        src = torch.cat([src_inp.new_zeros(1,*src_inp.shape[1:]),src_inp],0)
        src = src * math.sqrt(self.ninp)
        if self.use_posemb:
            src = self.pos_encoder(src,timestamps)
        src = self.premlp(src)
        output = self.transformer_encoder(src, src_mask)
        return output[0], output
    
class PositionalEncoding(nn.Module):
    # Postitional encoding to add timestamp information to the sequence
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        try:
            pe[:, 1::2] = torch.cos(position * div_term)
        except:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, timestamps):
        timestamps_clamped = torch.clip(timestamps, min=0, max=self.max_len-1).long()
        pem = self.pe[timestamps_clamped.permute(1,0)].squeeze(-2)
        L, B, D = pem.shape
        appended = torch.cat([torch.zeros(1,B,D).to(pem.device),pem],0)
        x = x + appended
        return self.dropout(x)


if __name__ == '__main__':
    pass