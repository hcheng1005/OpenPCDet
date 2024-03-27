import torch
from torch import nn
import torch.nn.functional as F

def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y

# 可学习的position embedding模块
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


'''
names: TransformerDecoderLayer
description: Briefly describe the function of your function
return {*}
'''
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 self_posembed=None, cross_posembed=None, cross_only=False):
        super().__init__()
        self.cross_only = cross_only
        if not self.cross_only:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    '''
    names: 
    description: TransformerDecoderLayer的前向过程
    param {*} self
    param {*} query
    param {*} key
    param {*} query_pos
    param {*} key_pos
    param {*} key_padding_mask
    param {*} attn_mask
    return {*}
    '''
    def forward(self, query, key, query_pos, key_pos, key_padding_mask=None, attn_mask=None):
        # NxCxP to PxNxC

        # 首先,使用 self_posembed 和 cross_posembed 分别计算 query_pos_embed 和 key_pos_embed,并将其维度顺序调整为 (P, N, C)
        query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        
        # 将 query 和 key 的维度顺序从 (N, C, P) 调整为 (P, N, C)
        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        # 使用 with_pos_embed 函数将位置编码嵌入到 query 中,得到 q、k 和 v
        q = k = v = self.with_pos_embed(query, query_pos_embed)

        #　计算自注意力,得到更新后的 query,并通过残差连接和层归一化
        query2 = self.self_attn(q, k, value=v)[0]  # self-attention
        query = query + self.dropout1(query2)
        query = self.norm1(query) # nn.LayerNorm(d_model)
        
        # 计算跨注意力,使用 with_pos_embed 函数将位置编码嵌入到 query、key 和 value 中,并执行多头注意力计算。
        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                        key=self.with_pos_embed(key, key_pos_embed),
                                        value=self.with_pos_embed(key, key_pos_embed), 
                                        key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0] # cross-attention

        # 将跨注意力的结果通过残差连接和层归一化,得到更新后的 query
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)
        return query
