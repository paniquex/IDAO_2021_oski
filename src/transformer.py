import torch
from torch import nn
from torch import Tensor


def split_last(x, shape):
    """Split the last dimension to given shape."""
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    """Merge the last n_dims to a dimension."""
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, dim, num_heads, drop_rate):
        super(MultiHeadSelfAttention, self).__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop_rate)
        self.num_heads = num_heads
        
        
    def forward(self, x, mask):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q, v, k = [split_last(x, (self.num_heads, -1)).transpose(1, 2) for x in [q, k, v]]
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.dropout(F.softmax(scores, dim=-1))
        scores = torch.matmul(v, scores).transpose(1, 2).contigous()
        scores = merge_last(scores, 2)
        self.scores = scores
        
        
class PointwiseFeedForward(nn.Module):
    def __init__(self, dim, hdim):
        super(PointwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hdim)
        self.fc2 = nn.Linear(hdim, dim)
        
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))
    

class TBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_rate, hdim):
        super(TBlock, self).__init__()
        self.attn = MultiHeadSelfAttention(dim, num_heads, drop_rate)
        self.ff = PointwiseFeedForward(dim, hdim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop_rate)
        
    def forward(self, x, mask):
        h = self.dropout(self.attn(self.norm1(x), mask))
        x = x + h
        h = self.dropout(self.proj(self.norm2(x)))
        x = x + h
        
        return x
    
    
def Transformer(nn.Module):
    def __init__(self, num_blocks, dim, num_heads, drop_rate, hdim):
        super(Transformer, self).__init__()
        self.num_blocks = num_blocks
        self.dim = dim
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.hdim = hdim
        
        self.blocks = nn.ModuleList([
            TBlock(dim, num_heads, drop_rate, hdim) for _ in range(num_blocks)
        ])
        
    def forward(self, x, mask):
        for block in self.blocks:
            x = block(x, mask)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, dim):
        super(PositionalEncoding, self).__init__()
        self.embeddings = nn.Parameter(Tensor(1, seq_len, dim))
    
    def forward(self, x, mask):
        return x + self.embeddings
    
    
