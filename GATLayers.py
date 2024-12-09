import torch
from torch import nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))  # 用于注意力计算
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        # 将节点特征h与权重矩阵W相乘，得到线性转换后的特征 Wh
        # h.shape: (N, in_features), Wh.shape: (N, out_features)
        output = torch.matmul(x, self.W)  # h(bs,17,2560)*W(2560,8)==>Wh(bs,17,8)

        e = self._prepare_attentional_mechanism_input(output)  # e(bs,17,17)

        zero_vec = -9e15 * torch.ones_like(e)
        # 逐元素比较adj > 0,为TRUE时attention对应位置的值来自e,否则是zero_vec即为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # 计算注意力分数，将不需要的部分设置为负无穷，以便在softmax操作中将它们忽略
        attention = F.softmax(attention, dim=1)  # 应用 softmax 操作，将注意力分数转化为权重
        attention = F.dropout(attention, self.dropout, training=self.training)  # 应用dropout操作以减轻过拟合
        h_prime = torch.matmul(attention, output)  # attention(bs,17,17)*Wh(bs,17,8)==>h_prime(bs,17,8)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    # 定义了一个辅助函数，用于计算注意力机制的输入
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)

        # Wh(bs,17,8) *(8,1)==>Wh1/Wh2(bs,17,1)
        # a(16,1)各取一半变为(8,1)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.permute(0, 2, 1)  # e(bs,17,17)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == '__main__':
    print("测试GAT代码")
    # 定义输入特征的维度和输出特征的维度
    in_features = 2560
    out_features = 8

    # 定义其他参数
    dropout = 0.5
    alpha = 0.2
    concat = True

    batch_size = 32
    node = 17
    # 创建一个示例的输入特征和邻接矩阵
    h = torch.randn(batch_size, node, in_features)  # 17个节点，每个节点2560维特征
    adj = torch.randn(node, node)  # 示例的邻接矩阵

    # 创建一个GraphAttentionLayer实例
    gat_layer = GraphAttentionLayer(in_features, out_features, dropout, alpha, concat)

    # 打印模型信息
    print(gat_layer)

    # 调用前向传播函数
    output = gat_layer(h, adj)

    # 输出的形状
    print("Output shape:", output.shape)
