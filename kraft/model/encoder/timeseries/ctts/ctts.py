import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
# refer to Section 3.5 in the paper

    def __init__(self, device, max_len=512, d_model=16):
        super().__init__()
        self.pos_enc = torch.zeros(max_len,d_model,requires_grad=False, device=device)
        pos = torch.arange(0, max_len, 1, requires_grad=False, device=device).reshape(-1,1)
        w_vector = 10000**(-2*(torch.arange(0, (d_model // 2), 1, device=device))/d_model)

        self.pos_enc[:,0::2] = torch.cos(pos * w_vector)
        self.pos_enc[:,1::2] = torch.sin(pos * w_vector)


    def forward(self, x):
        """
        x.shape = [batch_size, seq_len, data_dim]
        """
        return x + self.pos_enc[:x.shape[1], :].unsqueeze(0)
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_model=16):
        super().__init__()
        self.d_model = d_model

    def forward(self, q, k, v, mask=None):
        """
        q, k, v = transformed query, key, value
        q.shape, k.shape, v.shpae = [batch_size, num_head, seq_len, d_ff=d_model/num_head]
        mask = masking matrix, if the index has value False, kill the value; else, leave the value
        """
        k_T = k.transpose(-1,-2)

        # 1. matmul Q @ K_T
        scores = (q @ k_T) / math.sqrt(q.shape[-1])

        # ( Optional ) masking
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))

        # 2. softmax
        attention_weight = F.softmax(scores, dim=-1)

        # 3. matmul attention_weight @ V
        attention_value = attention_weight @ v

        return attention_value
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model=16,num_head=4):
        super().__init__()
        assert d_model % num_head == 0, "check if d_model is divisible by num_head"

        # params
        self.d_model = d_model
        self.num_head = num_head
        self.d_ff = d_model//num_head

        # q, k, v's weight
        self.q_weight = nn.Linear(d_model, d_model)
        self.k_weight = nn.Linear(d_model, d_model)
        self.v_weight = nn.Linear(d_model, d_model)

        # output weight for concat
        self.output_weight =  nn.Linear(d_model, d_model)

        # set attention block
        self.attention = ScaledDotProductAttention(d_model=d_model)

    def forward(self, q, k, v, mask=None):
        # compute multi-head attention value
        """
        q, k, v = pre-transformed query, key, value
        q.shape, k.shape, v.shpae = [batch_size, seq_len, d_model]
        mask = masking matrix, if the index has value False, kill the value; else, leave the value
        """
        batch_size, seq_len, d_model = q.shape

        # make them learnable
        q, k, v = self.q_weight(q), self.k_weight(k), self.v_weight(v)

        # reshape [batch_size, seq_len, d_model] to [batch_size, num_head, seq_len, d_ff]
        def reshape(x):
            return x.view(batch_size, seq_len, self.num_head, self.d_ff).transpose(1,2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # calculate attention value
        attention_value = self.attention(q,k,v,mask=mask)

        # concat heads --> result :  [batch_size, seq_len, d_model]
        concated_value = attention_value.transpose(1,2).reshape(batch_size, seq_len, d_model)

        output = self.output_weight(concated_value)

        return output
    
class PositionwiseFeedForwardNetwork(nn.Module):
    def __init__(self,d_model=16,d_ff=32):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class LayerNormalization(nn.Module):
    def __init__(self,d_model=16,eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self,x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        normed = (x - mean)/torch.sqrt(var + self.eps) # 정규화
        normed = self.gamma * normed + self.beta # 파라미터 추가

        return normed
    
class EncoderLayer(nn.Module):
    # Pre-Norm 구조로 기존의 Post-Norm 구조와 다르다. 
    def __init__(self,d_model=16,num_head=4,d_ff=32,drop_prob=.1):
        super().__init__()
        self.norm1 = LayerNormalization(d_model)
        self.attention = MultiHeadAttention(d_model, num_head)

        self.norm2 = LayerNormalization(d_model)
        self.ffn = PositionwiseFeedForwardNetwork(d_model, d_ff)

        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self,enc):
        # multi head attention
        _x = enc
        x = self.norm1(_x)
        x = self.attention(q=x, k=x, v=x)

        # add and norm
        x = self.dropout(x)
        x = x + _x

        # feed forward
        _x = x
        x = self.norm2(_x)
        x = self.ffn(x)

        # add and norm
        x = self.dropout(x)
        x = x + _x

        return x
    
class Encoder(nn.Module):
    def __init__(self,device,input_dim=3,num_layer=3,max_len=512,d_model=16,num_head=4,d_ff=32,drop_prob=.1):
        super().__init__()
        self.positional_emb = PositionalEncoding(device=device,
                                                 max_len=max_len,
                                                 d_model=d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  num_head=num_head,
                                                  d_ff=d_ff,
                                                  drop_prob=drop_prob)
                                                  for _ in range(num_layer)])

        self.input_fc = nn.Linear(input_dim, d_model)


    def forward(self,x):
        # transform dimension : embedding이 없어서 필요한 부분
        x = self.input_fc(x)

        x = self.positional_emb(x)

        for layer in self.layers:
            hidden = layer(x)

        return hidden
    
class CNNTokenizer(nn.Module):
    def __init__(self, input_dim, embed_dim, kernel_size=4, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=embed_dim,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # x shape: (B, T, D) → (B, D, T)
        x = x.permute(0, 2, 1)
        x = self.conv(x)  # (B, embed_dim, N_tokens)
        x = x.permute(0, 2, 1)  # (B, N_tokens, embed_dim)
        return x
    

class CTTS(nn.Module):
    INIT_SEQ = ['input_dim', 'embed_dim', 
                'kernel_size', 'stride', 'device', 
                'num_layers', 'num_heads', 'd_ff', 'dropout']
    
    def __init__(self, 
                 input_dim,              # 입력 feature 수 (D)
                 embed_dim,              # CNN + Transformer 임베딩 차원 (d_model)
                 kernel_size,            # CNN 커널 사이즈
                 stride,                 # CNN stride
                 device,                 # positional encoding에 필요
                 num_layers=3,           # Transformer 층 수
                 num_heads=4,            # Multi-head attention 헤드 수
                 d_ff=64,                # FFN hidden size
                 dropout=0.1):           # dropout 비율
        super().__init__()

        self.tokenizer = CNNTokenizer(input_dim=input_dim, 
                                      embed_dim=embed_dim, 
                                      kernel_size=kernel_size, 
                                      stride=stride)

        self.encoder = Encoder(device=device,
                               input_dim=embed_dim,   # CNN output = Transformer input
                               num_layer=num_layers,
                               max_len=128,           # CNN 이후 토큰 개수의 upper bound
                               d_model=embed_dim,
                               num_head=num_heads,
                               d_ff=d_ff,
                               drop_prob=dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, T=30, D)
        tokens = self.tokenizer(x)              # (B, N_tokens, embed_dim)
        encoded = self.encoder(tokens)          # (B, N_tokens, embed_dim)

        # 대표 토큰을 사용 (e.g., 평균)
        pooled = encoded.mean(dim=1)            # (B, embed_dim)

        out = self.head(pooled)                 # (B, embed_dim)
        return out
    