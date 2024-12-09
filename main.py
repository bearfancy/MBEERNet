# -*- coding:utf-8 -*-
import argparse
import math
import copy
import os
import time
from datetime import datetime

from colorama import Fore, Style
from einops import rearrange
from scipy.signal import savgol_filter
from thop import profile
from torchinfo import summary
from tqdm import tqdm

import utils
import torch
import models
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MBEER():
    def __init__(self, model=models.MBEERNet(), source_loaders=0, target_loader=0, batch_size=64, iteration=10000,
                 lr=0.001, momentum=0.9, log_interval=10, session_id=0, subject_id=0):
        # super(MSMDAER, self).__init__()
        self.model = model
        self.model.to(device)
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        self.batch_size = batch_size
        self.iteration = iteration
        self.lr = lr
        self.momentum = momentum
        self.log_interval = log_interval
        self.session_id = session_id
        self.subject_id = subject_id

    def __getModel__(self):
        return self.model

    def model_info(self):
        total = sum([param.nelement() for param in self.model.parameters()])
        print("模型参数量: %.4f" % total)
        # return summary(self.model)

    def train(self):
        # best_model_wts = copy.deepcopy(model.state_dict())
        source_iters = []
        for i in range(len(self.source_loaders)):
            source_iters.append(iter(self.source_loaders[i]))
        target_iter = iter(self.target_loader)
        correct = 0

        for i in tqdm(range(1, self.iteration + 1),
                      bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Style.RESET_ALL)):
            self.model.train()
            LEARNING_RATE = self.lr
            optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

            for j in range(len(source_iters)):
                try:
                    source_data, source_label = next(source_iters[j])  # 调用utils.py的CustomDataset的类中的getitem函数
                except Exception as err:
                    source_iters[j] = iter(self.source_loaders[j])
                    source_data, source_label = next(source_iters[j])
                try:
                    target_data, _ = next(target_iter)
                except Exception as err:
                    target_iter = iter(self.target_loader)
                    target_data, _ = next(target_iter)
                SG_source_data = torch.FloatTensor(savgol_filter(source_data, window_length=7, polyorder=2, axis=3)).to(
                    device)
                SG_target_data = torch.FloatTensor(savgol_filter(target_data, window_length=7, polyorder=2, axis=3)).to(
                    device)

                # (32,62,5,8)==>(32,1,62,40)
                SG_source_data = rearrange(SG_source_data, 's e b a ->s e (b a)').unsqueeze(1)
                SG_target_data = rearrange(SG_target_data, 's e b a ->s e (b a)').unsqueeze(1)

                source_data, source_label = source_data.to(device), source_label.to(device)
                target_data = target_data.to(device)

                optimizer.zero_grad()

                cls_loss, mmd_loss, l1_loss = self.model(data_src=source_data, number_of_source=len(source_iters),
                                                         SG_source_data=SG_source_data, SG_target_data=SG_target_data,
                                                         data_tgt=target_data,
                                                         label_src=source_label, mark=j)
                gamma = 2 / (1 + math.exp(-10 * i / self.iteration)) - 1
                beta = gamma / 100
                loss = cls_loss + gamma * mmd_loss + beta * l1_loss

                loss.backward()
                optimizer.step()

                # if i % log_interval == 0:
                #     print('\nTrain source' + str(
                #         j) + ', iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_loss: {:.6f}\tmmd_loss {:.6f}\tl1_loss: {:.6f}'.format(
                #         i, 100. * i / self.iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()
                #     )
                #           )
                #     if j == 13:
                #         print("*" * 100)

            if i % (log_interval) == 0:
                test_correct = self.test(i)
                print("\ntest_correct精度最高", 100. * correct / len(self.target_loader.dataset))
                print("test_correct精度现在", 100. * test_correct / len(self.target_loader.dataset))
                if test_correct > correct:
                    correct = test_correct
                    TITAN_pre_path = "./saveModel/cross_subject/"
                    if subject_id_main > 9:
                        dir_temp = TITAN_pre_path + str(session_id_main) + '_' + str(subject_id_main)
                    else:
                        dir_temp = TITAN_pre_path + str(session_id_main) + '_0' + str(subject_id_main)

                    if not os.path.exists(dir_temp):
                        os.makedirs(dir_temp)
                    else:
                        for i in os.listdir(dir_temp):
                            file_data = dir_temp + "/" + i
                            if os.path.isfile(file_data) == True:
                                os.remove(file_data)
                    if subject_id_main > 9:
                        dir = dir_temp + '/' + str(session_id_main) + '_' + str(
                            subject_id_main) + '_bestmodel_acc' + str(
                            100. * correct / len(self.target_loader.dataset)) + '.pt'
                    else:
                        dir = dir_temp + '/' + str(session_id_main) + '_0' + str(
                            subject_id_main) + '_bestmodel_acc' + str(
                            100. * correct / len(self.target_loader.dataset)) + '.pt'
                    torch.save(self.model, dir)
        return 100. * correct / len(self.target_loader.dataset)  # 返回至函数cross_subject的acc = model.train()

    def test(self, i):
        self.model.eval()  # 使用model.eval()切换到测试模式，不会更新模型的k，b参数
        test_loss = 0
        correct = 0
        corrects = []
        for i in range(len(self.source_loaders)):
            corrects.append(0)
        with torch.no_grad():
            for data, target in self.target_loader:
                # data(32,62,5,8)
                SG_source_data = torch.FloatTensor(savgol_filter(data, window_length=7, polyorder=2, axis=3)).to(
                    device)
                # (32,62,5,8)==>(32,1,62,40)
                SG_source_data = rearrange(SG_source_data, 's e b a ->s e (b a)').unsqueeze(1)

                data = data.to(device)
                target = target.to(device)
                preds = self.model(data, len(self.source_loaders), SG_source_data=SG_source_data)
                for i in range(len(preds)):
                    preds[i] = F.softmax(preds[i], dim=1)
                pred = sum(preds) / len(preds)  # pred(32,3) (bs,类别数)
                test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target.squeeze()).item()
                pred = pred.data.max(1)[1]
                correct += pred.eq(target.data.squeeze()).cpu().sum()
                for j in range(len(self.source_loaders)):
                    pred = preds[j].data.max(1)[1]
                    corrects[j] += pred.eq(target.data.squeeze()).cpu().sum()

            test_loss /= len(self.target_loader.dataset)
        return correct  # 返回至类MSMDAER()中函数train()的t_correct = self.test(i)


# 跨人
def cross_subject(data, label, session_id, subject_id, category_number, batch_size, iteration, lr, momentum,
                  log_interval):
    one_session_data, one_session_label = copy.deepcopy(data[session_id]), copy.deepcopy(label[session_id])
    # one_session_data[15,225,310,8]    one_session_label[15,225,1]
    train_idxs = list(range(15))  # 0到14的15个数
    del train_idxs[subject_id]  # 删除train_idxs中索引为subject_id个数据
    test_idx = subject_id  # subject_id作为目标域
    target_data, target_label = copy.deepcopy(one_session_data[test_idx]), copy.deepcopy(one_session_label[test_idx])
    source_data, source_label = copy.deepcopy(one_session_data[train_idxs]), copy.deepcopy(
        one_session_label[train_idxs])

    del one_session_label
    del one_session_data
    source_loaders = []
    for j in range(len(source_data)):
        source_loaders.append(torch.utils.data.DataLoader(dataset=utils.CustomDataset(source_data[j], source_label[j]),
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          drop_last=True))
    target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(target_data, target_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
    model = MBEER(model=models.MBEERNet(pretrained=False, number_of_source=len(source_loaders),
                                        number_of_category=category_number),
                  source_loaders=source_loaders,
                  target_loader=target_loader,
                  batch_size=batch_size,
                  iteration=iteration,
                  lr=lr,
                  momentum=momentum,
                  log_interval=log_interval)
    # model.model_info()

    # caModel = model.__getModel__()
    # total = sum([param.nelement() for param in caModel.parameters()])
    # print("==" * 50)
    # print(model.__getModel__())
    # print("模型总参数量: %.4fM" % (total / 1e6))
    # summary(caModel)
    # print("++" * 50)
    # tensor_input = torch.randn(1, 62, 5, 8)
    # Flops, params = profile(caModel, inputs=(tensor_input,),)  # macs
    # print('Flops: % .4fG' % (Flops / 1000000000))  # 计算量
    # print('params参数量: % .4fM' % (params / 1000000))  # 参数量：等价与上面的summary输出的Total params值
    # print("==" * 50)

    acc = model.train()

    print('\nTarget_subject_id: {}, current_session_id: {}, acc: {}'.format(test_idx, session_id, acc))
    return acc.item(), model  # 返回至main函数中acc, model = cross_subject(...)


# 跨会话
def cross_session(data, label, session_id, subject_id, category_number, batch_size, iteration, lr, momentum,
                  log_interval):
    ## LOSO
    train_idxs = list(range(3))
    del train_idxs[session_id]
    test_idx = session_id

    target_data, target_label = copy.deepcopy(data[test_idx][subject_id]), copy.deepcopy(label[test_idx][subject_id])
    source_data, source_label = copy.deepcopy(data[train_idxs][:, subject_id]), copy.deepcopy(
        label[train_idxs][:, subject_id])

    source_loaders = []
    for j in range(len(source_data)):
        source_loaders.append(torch.utils.data.DataLoader(dataset=utils.CustomDataset(source_data[j], source_label[j]),
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          drop_last=True))
    target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(target_data, target_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
    model = MBEER(model=models.MBEERNet(pretrained=False, number_of_source=len(source_loaders),
                                        number_of_category=category_number),
                  source_loaders=source_loaders,
                  target_loader=target_loader,
                  batch_size=batch_size,
                  iteration=iteration,
                  lr=lr,
                  momentum=momentum,
                  log_interval=log_interval)
    # print(model.__getModel__())
    acc = model.train()
    print('Target_session_id: {}, current_subject_id: {}, acc: {}'.format(test_idx, subject_id, acc))
    return acc.item(), model  # 返回至main函数中acc, model = cross_session(...)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MS-MDAER parameters')
    parser.add_argument('--dataset', type=str, default='seed3',
                        help='the dataset used for MS-MDAER, "seed3" or "seed4"')
    parser.add_argument('--norm_type', type=str, default='ele',
                        help='the normalization type used for data_seed, "ele", "sample", "global" or "none"')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='size for one batch, integer')
    parser.add_argument('--Total_sample_length', type=int, default=120,
                        help='Total length of data for one subject in one trial N, integer')
    parser.add_argument('--epoch', type=int, default=150,
                        help='training epoch, integer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

    args = parser.parse_args()
    dataset_name = args.dataset  # seed3
    bn = args.norm_type  # ele
    N = args.Total_sample_length

    data, label = utils.load_data(dataset_name, N=N)  # ########加载获取数据、标签#########
    batch_size = args.batch_size
    epoch = args.epoch
    lr = args.lr
    print('\nBS: {}, Total data length N:{} ,epoch: {}'.format(batch_size, N, epoch))
    momentum = 0.9
    log_interval = 10
    iteration = 0
    if dataset_name == 'seed3':
        iteration = 2000
    elif dataset_name == 'seed4':
        iteration = math.ceil(epoch * 820 / batch_size)
    else:
        iteration = 5000
    print('Iteration: {}'.format(iteration))
    csub = []  # 跨受试者,存放跨受试者de模型结果
    csesn = []  # 跨会话,

    trial_total, category_number, _ = utils.get_number_of_label_n_trial(dataset_name)
    startTime = time.time()
    print("开始时间:" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(startTime)))
    # cross-validation, LOSO  （交叉验证）

    todayDate = str(datetime.today().date())  # 获取开始运行程序的日期
    projectName = os.path.basename(os.getcwd())  # 获取项目名

    # 原始45个实验一起训练部分--跨人crossSubject--开始
    for session_id_main in tqdm(range(3),
                                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTCYAN_EX, Style.RESET_ALL)):
        for subject_id_main in tqdm(range(15), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Style.RESET_ALL)):
            print("会话{}--受试者{}---开始".format(session_id_main, subject_id_main))
            acc, model = cross_subject(data, label, session_id_main, subject_id_main, category_number,
                                       batch_size, iteration, lr, momentum, log_interval)
            model.model_info()
            csub.append(acc)
            fileObject = open("./Result/csub45_" + projectName + "_RunTime" + todayDate + ".txt", 'a', encoding='utf-8')
            intervalTime = time.time()
            timerStr = utils.timer(startTime, intervalTime)
            fileObject.write(
                "crossSubject----" + timerStr + ";" + "sessinId:" + str(session_id_main) + ";" + "subjectId:" + str(
                    subject_id_main) + ";" + "实验最高精度:" + str(acc))
            fileObject.write('\n')
            fileObject.close()

            # total = sum([param.nelement() for param in model.parameters()])
            # print("模型参数量: %.4fM" % (total / 1e6))

            print("Cross-subject: ", csub)
            print("Cross-subject mean: ", np.mean(csub), "std: ", np.std(csub))
            print("会话{}--受试者{}---结束".format(session_id_main, subject_id_main))
    # 原始45个实验一起训练部分--跨人crossSubject--结束

    # # 原始45个实验一起训练部分--跨会话crossSession--开始
    # for session_id_main in tqdm(range(15),
    #                             bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTCYAN_EX, Style.RESET_ALL)):
    #     for subject_id_main in tqdm(range(3), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Style.RESET_ALL)):
    #         acc, model = cross_session(data, label, session_id_main, subject_id_main, category_number,
    #                                    batch_size, iteration, lr, momentum, log_interval)
    #         model.model_info()
    #         csesn.append(acc)
    #         fileObject = open("./Result/csesn45_" + projectName + "_RunTime" + todayDate + ".txt", 'a',
    #                           encoding='utf-8')
    #         intervalTime = time.time()
    #         timerStr = utils.timer(startTime, intervalTime)
    #         fileObject.write(
    #             "crossSession----" + timerStr + ";" + "sessinId:" + str(session_id_main) + ";" + "subjectId:" + str(
    #                 subject_id_main) + ";" + "实验最高精度:" + str(acc))
    #         fileObject.write('\n')
    #         fileObject.close()
    #
    #         # total = sum([param.nelement() for param in model.parameters()])
    #         # print("模型参数量: %.4fM" % (total / 1e6))
    #         print("Cross-session: ", csesn)
    #         print("Cross-session mean: ", np.mean(csesn), "std: ", np.std(csesn))
    # # 原始45个实验一起训练部分--跨会话crossSession--结束

    endTime = time.time()
    totalTime = endTime - startTime
    hour = totalTime / 3600
    min = totalTime % 3600 / 60
    second = totalTime % 3600 % 60
    print("开始时间:" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(startTime)))
    print("结束时间:" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(endTime)))
    print("花费: %d小时%d分%3f秒" % (hour, min, second))
