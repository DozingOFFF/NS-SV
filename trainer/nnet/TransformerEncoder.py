import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.nn import MultiheadAttention
import math


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()
        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        seq_len = x.size(1)

        return self.encoding[:seq_len, :]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v, mask=None):
        # |q| : (batch_size, num_heads, q_len, d_k), |k| : (batch_size, num_heads, k_len, d_k), |v| : (batch_size, num_heads, v_len, d_v)
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, float('-inf'))
        # |attn_score| : (batch_size, num_heads, q_len, k_len)
        attn_weights = nn.Softmax(dim=-1)(attn_score)
        # |attn_weights| : (batch_size, num_heads, q_len, k_len)
        output = torch.matmul(attn_weights, v)
        # |output| : (batch_size, num_heads, q_len, d_v)
        return output, attn_weights



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = self.d_v = embedding_dim // num_heads
        self.WQ = nn.Linear(embedding_dim, embedding_dim)
        self.WK = nn.Linear(embedding_dim, embedding_dim)
        self.WV = nn.Linear(embedding_dim, embedding_dim)
        self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k)
        self.linear = nn.Linear(num_heads * self.d_v, embedding_dim)

    def forward(self, Q, K, V, mask=None):
        # |Q| : (batch_size, q_len, embedding_dim), |K| : (batch_size, k_len, embedding_dim), |V| : (batch_size, v_len, embedding_dim)
        batch_size = Q.size(0)
        q_heads = self.WQ(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k_heads = self.WK(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v_heads = self.WV(V).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)
        # |q_heads| : (batch_size, num_heads, q_len, d_k), |k_heads| : (batch_size, num_heads, k_len, d_k), |v_heads| : (batch_size, num_heads, v_len, d_v)
        attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, mask)
        # |attn| : (batch_size, num_heads, q_len, d_v)
        # |attn_weights| : (batch_size, num_heads, q_len, k_len)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)
        # |attn| : (batch_size, q_len, num_heads * d_v)
        output = self.linear(attn)
        # |output| : (batch_size, q_len, embedding_dim)
        return output, attn_weights



class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, d_ff)
        self.linear2 = nn.Linear(d_ff, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len, embedding_dim)
        output = self.relu(self.linear1(inputs))
        # |output| : (batch_size, seq_len, d_ff)
        output = self.linear2(output)
        # |output| : (batch_size, seq_len, embedding_dim)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, p_drop, d_ff):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.ffn = PositionWiseFeedForwardNetwork(embedding_dim, d_ff)
        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(embedding_dim, eps=1e-6)

    def forward(self, inputs_q, inputs_k, inputs_v, mask=None):
        # |inputs| : (batch_size, seq_len, embedding_dim)
        attn_outputs, _ = self.mha(inputs_q, inputs_k, inputs_v, mask)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(inputs_q + attn_outputs)
        # |attn_outputs| : (batch_size, seq_len(=q_len), embedding_dim)
        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)
        # |ffn_outputs| : (batch_size, seq_len, embedding_dim)
        return ffn_outputs



class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N encoder layers.
    Args:
        hidden_dim   (int)    : dimension of the input feature
        num_layers   (int)    : number of sub-encoder-layers in the encoder
        num_heads    (int)    : number of heads in the multiheadattention models
        p_drop     (float)  : dropout value
        d_ff       (int)    : dimension of the feedforward network model
    """

    def __init__(self, hidden_dim, num_layers, num_heads, max_len, p_drop=0.1, d_ff=512, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.pos_emb = PositionalEncoding(hidden_dim,max_len+1)
        self.etype_emb = nn.Parameter(torch.randn((1, hidden_dim)))
        self.ttype_emb = nn.Parameter(torch.randn((1, hidden_dim)))
        self.drop_out = nn.Dropout(p=p_drop)
        # layers
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, num_heads, p_drop, d_ff) for _ in range(num_layers)])


    def forward(self, enroll_emb, test_feature):

        with torch.no_grad():
            pos_emb = self.pos_emb(test_feature).to(torch.cuda.current_device())
        enroll_emb = enroll_emb + self.etype_emb
        test_feature = test_feature + self.ttype_emb + pos_emb
        inputs = torch.cat((enroll_emb, test_feature), dim=1)
        outputs = self.drop_out(inputs)

        mask = self.get_attention_mask(enroll_emb.size(1), outputs.size(1), outputs.size(1)).to(
            torch.cuda.current_device())
        for i, layer in enumerate(self.layers):
            outputs = layer(outputs, outputs, outputs, mask)

        return outputs

    def get_attention_mask(self, e_len, q_len, k_len):
        attn_mask = torch.ones((q_len, k_len))
        diagonal_mask = torch.eye(e_len)
        attn_mask[:e_len, :e_len] = diagonal_mask
        attn_mask[e_len:, :e_len] = 0
        return attn_mask.unsqueeze(0).unsqueeze(0)

