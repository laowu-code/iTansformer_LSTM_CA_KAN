import torch.nn as nn
import torch
class CrossAttention(nn.Module):
    def __init__(self, dim, lenth):
        super(CrossAttention, self).__init__()
        self.query_layer = nn.Linear(dim, dim)
        self.key_layer = nn.Linear(lenth, dim)
        self.value_layer = nn.Linear(lenth, dim)
        self.scale = dim ** 0.5

    def forward(self, input1, input2):
        # 生成查询、键、值
        query = self.query_layer(input1)  # (b, c1, d)
        key = self.key_layer(input2)  # (b, c, l)
        value = self.value_layer(input2)  # (b, c, l)
        # 计算注意力权重
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale  # (b, c1, c)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # 归一化
        # 使用注意力权重加权值
        attended_output = torch.matmul(attention_weights, value)  # (b, c1, d)
        return attended_output