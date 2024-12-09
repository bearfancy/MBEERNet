# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

import utils
from GATLayers import GraphAttentionLayer
from HFCAT import FlashCrossAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CFFE(nn.Module):

    def __init__(self):
        super(CFFE, self).__init__()
        self.num_layers = 2
        self.batch_size = 32
        self.hidden_size1 = 15
        self.hidden_size2 = 13
        self.hidden_size3 = 5
        self.batch_first_lstm = True
        self.bidirectional = True
        self.dict = {'Fr1': np.array([0, 3, 8, 7, 6, 5]), 'Fr2': np.array([2, 4, 10, 11, 12, 13]),
                     'Tp1': np.array([14, 23, 32, 41, 50]), 'Tp2': np.array([22, 31, 40, 49, 56]),
                     'Cn1': np.array([15, 16, 17, 26, 25, 24, 33, 34, 35]),
                     'Cn2': np.array([21, 20, 19, 28, 29, 30, 39, 38, 37]),
                     'Pr1': np.array([42, 43, 44, 52, 51]), 'Pr2': np.array([48, 47, 46, 54, 55]),
                     'Oc1': np.array([58, 57]), 'Oc2': np.array([60, 61])}

        self.transformer_fL = nn.TransformerEncoderLayer(d_model=30, nhead=6, batch_first=self.batch_first_lstm)
        self.transformer_fR = nn.TransformerEncoderLayer(d_model=30, nhead=6, batch_first=self.batch_first_lstm)
        self.transformer_tL = nn.TransformerEncoderLayer(d_model=25, nhead=5, batch_first=self.batch_first_lstm)
        self.transformer_tR = nn.TransformerEncoderLayer(d_model=25, nhead=5, batch_first=self.batch_first_lstm)
        self.transformer_pL = nn.TransformerEncoderLayer(d_model=25, nhead=5, batch_first=self.batch_first_lstm)
        self.transformer_pR = nn.TransformerEncoderLayer(d_model=25, nhead=5, batch_first=self.batch_first_lstm)
        self.transformer_oL = nn.TransformerEncoderLayer(d_model=10, nhead=5, batch_first=self.batch_first_lstm)
        self.transformer_oR = nn.TransformerEncoderLayer(d_model=10, nhead=5, batch_first=self.batch_first_lstm)

        self.LSTM_f = nn.LSTM(30, self.hidden_size1, self.num_layers, batch_first=self.batch_first_lstm,
                              bidirectional=True)
        self.LSTM_t = nn.LSTM(25, self.hidden_size2, self.num_layers, batch_first=self.batch_first_lstm,
                              bidirectional=True)
        self.LSTM_p = nn.LSTM(25, self.hidden_size2, self.num_layers, batch_first=self.batch_first_lstm,
                              bidirectional=True)
        self.LSTM_o = nn.LSTM(10, self.hidden_size3, self.num_layers, batch_first=self.batch_first_lstm,
                              bidirectional=True)
        self.fc_f = nn.Sequential(
            nn.Linear(240, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 16),
            nn.Dropout(0.2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.fc_t = nn.Sequential(
            nn.Linear(208, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 16),
            nn.Dropout(0.2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.fc_p = nn.Sequential(
            nn.Linear(208, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 16),
            nn.Dropout(0.2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.fc_o = nn.Sequential(
            nn.Linear(80, 16),
            nn.Dropout(0.2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.b_n1 = nn.BatchNorm2d(5)
        self.b_n2 = nn.BatchNorm1d(64)

    def forward(self, x):
        # x的维度是（32，62，5，8）
        # 5个频段*62个电极=310个特征维度
        batch_size = x.shape[0]
        k = list(self.dict.keys())
        # k为['Fr1', 'Fr2', 'Tp1', 'Tp2', 'Cn1', 'Cn2', 'Pr1', 'Pr2', 'Oc1', 'Oc2']

        h11 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size1).cuda()  # (4,32,15)
        c11 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size1).cuda()  # (4,32,15)

        h12 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size2).cuda()  # (4,32,13)
        c12 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size2).cuda()  # (4,32,13)

        h13 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size3).cuda()  # (4,32,5)
        c13 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size3).cuda()  # (4,32,5)

        fr_l = x[:, self.dict[k[0]]]  # (32.6.5.8)
        fr_l = fr_l.reshape((fr_l.shape[0], fr_l.shape[1] * fr_l.shape[2], 8))  # (32.30.8)
        fr_l = fr_l.permute(0, 2, 1)  # (32,8,30) (batch,seq,features)

        fr_r = x[:, self.dict[k[1]]]
        fr_r = fr_r.reshape((fr_r.shape[0], fr_r.shape[1] * fr_r.shape[2], 8))
        fr_r = fr_r.permute(0, 2, 1)

        # 总的为输出O为[seq_len, batch_size, hidden_size]
        tp_l = x[:, self.dict[k[2]]]
        tp_l = tp_l.reshape((tp_l.shape[0], tp_l.shape[1] * tp_l.shape[2], 8))
        tp_l = tp_l.permute(0, 2, 1)

        tp_r = x[:, self.dict[k[3]]]
        tp_r = tp_r.reshape((tp_r.shape[0], tp_r.shape[1] * tp_r.shape[2], 8))
        tp_r = tp_r.permute(0, 2, 1)

        p_l = x[:, self.dict[k[6]]]
        p_l = p_l.reshape((p_l.shape[0], p_l.shape[1] * p_l.shape[2], 8))
        p_l = p_l.permute(0, 2, 1)

        p_r = x[:, self.dict[k[7]]]
        p_r = p_r.reshape((p_r.shape[0], p_r.shape[1] * p_r.shape[2], 8))
        p_r = p_r.permute(0, 2, 1)

        o_l = x[:, self.dict[k[8]]]
        o_l = o_l.reshape((o_l.shape[0], o_l.shape[1] * o_l.shape[2], 8))
        o_l = o_l.permute(0, 2, 1)

        o_r = x[:, self.dict[k[9]]]
        o_r = o_r.reshape((o_r.shape[0], o_r.shape[1] * o_r.shape[2], 8))
        o_r = o_r.permute(0, 2, 1)

        x_fl = self.transformer_fL(fr_l)
        x_fr = self.transformer_fR(fr_r)
        x_tl = self.transformer_tL(tp_l)
        x_tr = self.transformer_tR(tp_r)
        x_pl = self.transformer_pL(p_l)
        x_pr = self.transformer_pR(p_r)
        x_ol = self.transformer_oL(o_l)
        x_or = self.transformer_oR(o_r)

        x_f = x_fr - x_fl
        x_t = x_tr - x_tl
        x_p = x_pr - x_pl
        x_o = x_or - x_ol

        x_f, _ = self.LSTM_f(x_f, (h11, c11))
        x_t, _ = self.LSTM_t(x_t, (h12, c12))
        x_p, _ = self.LSTM_p(x_p, (h12, c12))
        x_o, _ = self.LSTM_o(x_o, (h13, c13))

        x = torch.cat((self.fc_f(x_f.reshape(batch_size, -1)), self.fc_t(x_t.reshape(batch_size, -1)),
                       self.fc_p(x_p.reshape(batch_size, -1)), self.fc_o(x_o.reshape(batch_size, -1))), dim=1)
        x = self.b_n2(x)
        x = x.reshape(batch_size, -1)

        return x


def pretrained_CFE(pretrained=False):
    model = CFFE()
    if pretrained:
        pass
    return model

class DSFE(nn.Module):
    def __init__(self):
        super(DSFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(64, 32),
            # nn.ReLU(inplace=True),
            nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x


#     上分支BG-DGAT Module代码----开始
class PowerLayer(nn.Module):
    '''
    The power layer: calculates the log-transformed power of the data
    对数据进行对数变换
    '''

    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(2)))


class Aggregator():
    '''区域聚合函数'''

    def __init__(self, idx_area):
        # chan_in_area: a list of the number of channels within each area(每个区域内的通道数列表)
        self.chan_in_area = idx_area
        # self.idx = self.get_idx(idx_area)
        self.area = len(idx_area)  # 脑部区域数量

    def forward(self, x):
        # x: batch x channel x data     (32,62,2506)
        data = []
        for i, area in enumerate(range(self.area)):
            area_data = []
            for j in range(len(self.chan_in_area[i])):
                area_data.append(x[:, self.chan_in_area[i][j], :])
            data.append(self.aggr_fun(torch.stack(area_data, dim=1), dim=1))
        return torch.stack(data, dim=1)  # 得到(32,17,2560)

    def aggr_fun(self, x, dim):
        # return torch.max(x, dim=dim).values
        return torch.mean(x, dim=dim)


class BG_DGAT(nn.Module):
    '''BG_DGAT模块'''

    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate * pool))
        )

    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph):
        super(BG_DGAT, self).__init__()
        # input_size之前的数据维度是(bs,62,5,8)==>(bs,1,62,40)  需要先对数据的维度进行处理
        # input_size:(1,62,40)
        # num_classes=64, input_size, sampling_rate=5 , num_T=64,
        #  out_graph=8, dropout_rate=0.5, pool=16, pool_step_rate=0.25, idx_graph长度为17的graph

        self.idx = idx_graph  # idx表示每个区域的电极下标的列表,共17个区域
        self.window = [2 ** 0, 2 ** 1, 2 ** 2]
        self.pool = pool
        self.channel = input_size[1]  # channel=62
        self.brain_area = len(self.idx)  # 脑部区域个数:17

        # by setting the convolutional kernel being (1,length) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[0] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception2 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[1] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception3 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[2] * sampling_rate)),
                                               self.pool, pool_step_rate)

        self.Tception4SG = nn.Conv2d(input_size[0], num_T, kernel_size=(1, 1), stride=(1, 1))

        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_t_ = nn.BatchNorm2d(num_T)
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2))  # 加入SG滤波后恢复使用池化操作
        )
        # diag(W) to assign a weight to each local areas
        size = self.get_size_temporal(input_size)  # (1,62,40)==>(1,62,2560)
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
                                                requires_grad=True)  # requires_grad=True表示参数是可学习,local_filter_weight大小为(62,2560)
        nn.init.xavier_uniform_(self.local_filter_weight)
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
                                              requires_grad=True)  # local_filter_bias(1,62,1)

        # aggregate function
        self.aggregate = Aggregator(self.idx)

        # trainable adj weight for global network
        self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)
        # to be used after local graph embedding
        self.bn = nn.BatchNorm1d(self.brain_area)
        self.bn_ = nn.BatchNorm1d(self.brain_area)
        # learn the global network of networks
        self.GAT = GraphAttentionLayer(size[-1], out_graph)# size[-1]--2560为;outgraph--8

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.brain_area * out_graph), num_classes))  # in(17*8=136),out(64)

    def forward(self, x, SG_x):
        y = self.Tception1(x)  # (bs,1,62,40)==>(bs,64,62,36==>17) Tception1包含一维卷积操作和PowerLayer两个操作,维度有两个变化
        out = y
        y = self.Tception2(x)  # 输出:(bs,64,62,31==>14)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)  # 输出:(bs,64,62,21==>9)
        out = torch.cat((out, y), dim=-1)  # 输出:(bs,64,62,88==>40)
        out = torch.cat((out, self.Tception4SG(SG_x)), dim=-1)  # 输出:(bs,64,62,80=17+14+9+40)

        out = self.BN_t(out)
        out = self.OneXOneConv(out)  # 输出:(bs,64,62,40) 加入SG滤波后恢复使用池化操作
        out = self.BN_t_(out)
        out = out.permute(0, 2, 1, 3)  # 输出:(bs,62,64,40) (batch_size,电极数量,时间长度,每个时间上的采样点数)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))  # 输出:(bs,62,64*40=2560)
        out = self.local_filter_fun(out, self.local_filter_weight)  # local_filter_weight:(62,2560) 公式(8)
        out = self.aggregate.forward(out)  # 按区域进行聚合 (32,62,2560)==>(32,17,2560)
        # 据输入的特征表示 out 计算节点之间的相似度得分，然后通过全局邻接矩阵和非线性激活函数对得分进行调整，并最终得到归一化的邻接矩阵
        adj = self.get_adj(out)  # out(bs,17,11072);adj(bs,17,17)
        out = self.bn(out)
        out = self.GAT(out, adj)
        out = self.bn_(out)
        out = out.view(out.size()[0], -1)  # out(bs,17,8)==>(bs,17*32=136)
        out = self.fc(out)  # (bs,136)==>(bs,64)
        return out

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point (频率维度,通道数,输入数据的时间维度)=>(1,62,40)
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        z = self.Tception1(data)
        out = z
        z = self.Tception2(data)
        out = torch.cat((out, z), dim=-1)
        z = self.Tception3(data)
        out = torch.cat((out, z), dim=-1)
        yy = self.Tception4SG(data)
        zz = yy
        out = torch.cat((out, zz), dim=-1)  # 输出:(bs,64,62,80=17+14+9+40)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_t_(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))  # (1,62,64*40=2560)
        size = out.size()  # (1,62,2560)

        return size  # 2560

    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)  # w(62,2560)==>(64,62,2560)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def get_adj(self, x, self_loop=True):
        # x: b, node, feature ==>(32,17,2506)
        adj = self.self_similarity(x)  # 输出adj(b, n, n)即(64,17,17)表示为节点之间的自相似性矩阵
        num_nodes = adj.shape[-1]
        adj = F.relu((adj + torch.eye(num_nodes).to(device)) * (
                self.global_adj + self.global_adj.transpose(1, 0)))  # global_adj(17,17);adj(64,17,17)
        rowsum = torch.sum(adj, dim=-1)  # 求行和, adj(64,17,17)==>rowsum(64,17)
        mask = torch.zeros_like(rowsum)  # 产生全为0的矩阵,大小等于rowsum(64,17)
        mask[rowsum == 0] = 1  # 遍历 rowsum 列表中的每个元素，如果元素的值为0，则将 mask 对应位置的元素赋值为1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)  # 计算每个元素的倒数的平方根,用于归一化邻接矩阵  d_inv_sqrt(64,17)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj  # (64,17,17)

    def self_similarity(self, x):
        # x: b, node, feature; x(32,17,2506);x_(32,2506,17)
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s  # s(32,17,17)
#     上分支BG-DGAT Module代码----结束


class MBEERNet(nn.Module):
    def __init__(self, pretrained=False, number_of_source=15, number_of_category=3):
        super(MBEERNet, self).__init__()
        local_graph_index = [[0, 3], [1, 9, 18, 27, 36, 45, 53, 59], [2, 4], [5, 6, 7, 8], [10, 11, 12, 13],
                             [14, 23, 32],
                             [15, 16, 17], [19, 20, 21], [22, 31, 40], [24, 25, 26], [28, 29, 30], [33, 34, 35],
                             [37, 38, 39],
                             [41, 42, 43, 44], [46, 47, 48, 49], [50, 51, 52, 57, 58], [54, 55, 56, 60, 61]]
        num_class = 64
        input_shape = tuple([1, 62, 40])
        sampling_rate = 5
        num_T = 64
        out_graph = 8
        drop_rate = 0.5
        pool = 4
        pool_step_rate = 0.5
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        self.BG_DGAT = BG_DGAT(num_classes=num_class, input_size=input_shape, sampling_rate=sampling_rate,
                               num_T=num_T, out_graph=out_graph, dropout_rate=drop_rate, pool=pool,
                               pool_step_rate=pool_step_rate, idx_graph=local_graph_index)

        self.kernelAttention = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.FlashCrossAttention = FlashCrossAttention(num_heads=2,
                                                       embed_dimension=64,
                                                       bias=False,
                                                       dropout=0.1)
        for i in range(number_of_source):  # 每个源域数据都需要得到对应的(DSFE+DSC)模块
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) + '=nn.Linear(32,' + str(number_of_category) + ')')

    def forward(self, data_src, number_of_source, SG_source_data, SG_target_data=0, data_tgt=0, label_src=0,
                mark=0):
        """
        Forward function for the model.

        Args:
            data_src: Source domain data.
            number_of_source: Number of source domains.
            SG_source_data: Source domain specific data for the shared graph.
            SG_target_data: Target domain specific data for the shared graph, default is 0.
            data_tgt: Target domain data, default is 0.
            label_src: Labels for the source domain data, default is 0.
            mark: Identifier for the specific source domain, default is 0.

        Returns:
            If self.training is True, returns classification loss, MMD loss, and discrepancy loss.
            Otherwise, returns the prediction results for each source domain.
        """
        mmd_loss = 0
        disc_loss = 0
        data_tgt_DSFE = []
        if self.training == True:
            # common feature extractor
            data_src_LGG_in = rearrange(data_src, 's e b a -> s e (b a)').unsqueeze(1)  # (32,62,5,8)==>(32,1,62,40)
            data_tgt_LGG_in = rearrange(data_tgt, 's e b a -> s e (b a)').unsqueeze(1)

            data_src_CFE = self.sharedNet(data_src)  # (32,62,5,8)==>(32,64)
            data_tgt_CFE = self.sharedNet(data_tgt)

            data_src_LGG = self.BG_DGAT(data_src_LGG_in, SG_source_data)  # (32,1,62,40)==>(32,64)
            data_tgt_LGG = self.BG_DGAT(data_tgt_LGG_in, SG_target_data)

            data_src_cat = self.FlashCrossAttention(data_src_LGG.unsqueeze(1), data_src_CFE.unsqueeze(1))
            data_tgt_cat = self.FlashCrossAttention(data_tgt_LGG.unsqueeze(1), data_tgt_CFE.unsqueeze(1))

            # Each domain specific feature extractor (DSFE) extracts the domain specific features of the target data_seed
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_cat)
                data_tgt_DSFE.append(data_tgt_DSFE_i)

            # Use the specific feature extractor to extract the source data_seed and calculate the mmd loss
            DSFE_name = 'self.DSFE' + str(mark)
            data_src_DSFE = eval(DSFE_name)(data_src_cat)

            # mmd_loss += utils.mmd(data_src_DSFE, data_tgt_DSFE[mark])
            mmd_loss += utils.mmd_linear(data_src_DSFE, data_tgt_DSFE[mark])
            # discrepancy loss
            for i in range(len(data_tgt_DSFE)):  # data_tgt_DSFE list contains the features of target data after being processed by different source domain specific feature extractors (DSFE)
                if i != mark:
                    disc_loss += torch.mean(torch.abs(
                        F.softmax(data_tgt_DSFE[mark], dim=1) - F.softmax(data_tgt_DSFE[i], dim=1)
                    ))
            # domain specific classifier and cls_loss (DSC)
            DSC_name = 'self.cls_fc_DSC' + str(mark)
            pred_src = eval(DSC_name)(data_src_DSFE)
            cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src.squeeze())

            return cls_loss, mmd_loss, disc_loss

        else:  # Below is the test code block
            data_src_LGG_in = rearrange(data_src, 's e b a -> s e (b a)').unsqueeze(1)  # (32,62,5,8)==>(32,1,62,40)
            data_CFE_ = self.sharedNet(data_src)
            data_src_LGG_ = self.BG_DGAT(data_src_LGG_in, SG_source_data)

            data_src_cat = self.FlashCrossAttention(data_src_LGG_.unsqueeze(1), data_CFE_.unsqueeze(1))

            pred = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_src_cat)
                pred.append(eval(DSC_name)(feature_DSFE_i))
            return pred
