# ------------------------------------------------------------------------
# Modified from SemCKD (https://github.com/DefangChen/SemCKD)
# Copyright (c) College of Computer Science, Zhejiang University, China. All Rights Reserved
# ------------------------------------------------------------------------


from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SemCKDLoss(nn.Module):
  """Cross-Layer Distillation with Semantic Calibration, AAAI2021"""

  def __init__(self):
    super(SemCKDLoss, self).__init__()
    self.crit = nn.MSELoss(reduction='none')

  def forward(self, s_value, f_target, weight):
     bsz, num_stu, num_tea = weight.shape
     ind_loss = torch.zeros(bsz, num_stu, num_tea).cuda()

     for i in range(num_stu):
       for j in range(num_tea):
         ind_loss[:, i, j] = self.crit(s_value[i][j], f_target[i][j]).reshape(bsz, -1).mean(-1)

     loss = (weight * ind_loss).sum() / (1.0 * bsz * num_stu)  # 除法将全局损失平均到student每一层的每一个实例上
     return loss

  '''def forward(self,feat_t,feat_s):
    s_len=len(feat_s)
    t_len=len(feat_t)
    input_channel=feat_s[0].shape[0]
    s_n = [f.shape[1] for f in feat_s]
    t_n = [f.shape[1] for f in feat_t]
    selfa=SelfA(s_len, t_len, input_channel, s_n, t_n)
    s_value, f_target, weight = selfa.forward(feat_s, feat_t)#新增加部分
    bsz, num_stu, num_tea = weight.shape
    ind_loss = torch.zeros(bsz, num_stu, num_tea).cuda()

    for i in range(num_stu):
      for j in range(num_tea):
        ind_loss[:, i, j] = self.crit(s_value[i][j], f_target[i][j]).reshape(bsz, -1).mean(-1)

    loss = (weight * ind_loss).sum() / (1.0 * bsz * num_stu)  # 除法将全局损失平均到student每一层的每一个实例上
    return loss'''


class SelfA(nn.Module):
  """Cross layer Self Attention"""

  def __init__(self, s_len, t_len, input_channel, s_n, s_t, factor=4):
    super(SelfA, self).__init__()
    #s_len=len(feat_s)
    #t_len=len(feat_t)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    for i in range(t_len):
      setattr(self, 'key_weight' + str(i), MLPEmbed(input_channel, input_channel // factor))
    for i in range(s_len):
      setattr(self, 'query_weight' + str(i), MLPEmbed(input_channel, input_channel // factor))

    for i in range(s_len):
      for j in range(t_len):
        setattr(self, 'regressor' + str(i) + str(j), AAEmbed(s_n[i], s_t[j]))

  def forward(self, feat_s, feat_t):  # feat_s是所有batch中student所有层的特征，feat_t是所有batch中teacher所有层的特征

    # sim_t = list(range(len(feat_t)))
    # sim_s = list(range(len(feat_s)))
    # bsz = feat_s[0].shape[0]  # 为什么不直接用batch_size表示？？？
    # similarity matrix
    # for i in range(len(feat_t)):  # 计算batch中每一个实例与其他实例之间的相似度
    #   sim_temp = feat_t[i].reshape(bsz, -1)  # R表示的reshape操作
    #   sim_t[i] = torch.matmul(sim_temp, sim_temp.t())  # 矩阵相乘
    # for i in range(len(feat_s)):
    #   sim_temp = feat_s[i].reshape(bsz, -1)
    #   sim_s[i] = torch.matmul(sim_temp, sim_temp.t())

    # # key of target layers
    # proj_key = self.key_weight0(sim_t[0])
    # proj_key = proj_key[:, :, None]

    # for i in range(1, len(sim_t)):
    #   temp_proj_key = getattr(self, 'key_weight' + str(i))(sim_t[i])
    #   proj_key = torch.cat([proj_key, temp_proj_key[:, :, None]], 2)

    # # query of source layers
    # proj_query = self.query_weight0(sim_s[0])
    # proj_query = proj_query[:, None, :]
    # for i in range(1, len(sim_s)):
    #   temp_proj_query = getattr(self, 'query_weight' + str(i))(sim_s[i])
    #   proj_query = torch.cat([proj_query, temp_proj_query[:, None, :]], 1)

    # # attention weight
    # energy = torch.bmm(proj_query,
    #                    proj_key)  # 三维矩阵的乘法（两个tensor第一维大小相等，第一个tensor的第三维和第二个tensor的第二维大小相等）batch_size X No.stu feature X No.tea feature
    # attention = F.softmax(energy, dim=-1)  # attention的权重系数，dim=-1表示在行上进行softmax

    # feature space alignment
    # for i in range(len(sim_s)):
    #   proj_value_stu.append([])
    #   value_tea.append([])
    #   for j in range(len(sim_t)):
    #     s_H, t_H = feat_s[i].shape[2], feat_t[j].shape[2]  # 卷积特征层的高度H
    #     # 统一proj_value_stu和value_tea大小
    #     if s_H > t_H:  # 如果学生特征层高度大则池化学生层特征图大小
    #       input = F.adaptive_avg_pool2d(feat_s[i], (t_H, t_H))  # 输入，输出大小
    #       proj_value_stu[i].append(getattr(self, 'regressor' + str(i) + str(j))(input))
    #       value_tea[i].append(feat_t[j])
    #     elif s_H < t_H or s_H == t_H:  # 如果教师特征层高度大则池化教师层特征图大小
    #       target = F.adaptive_avg_pool2d(feat_t[j], (s_H, s_H))
    #       proj_value_stu[i].append(getattr(self, 'regressor' + str(i) + str(j))(feat_s[i]))
    #       value_tea[i].append(target)


    proj_value_stu = []
    value_tea = []
    proj_value_stu.append([])
    value_tea.append([])
    for i in range(3):
      proj_value_stu[0].append(feat_s[i+1])
      value_tea[0].append(feat_t[i])
    if torch.cuda.is_available():
      device = torch.device("cuda")
      attention= torch.ones(1,1,3,device=device)

    return proj_value_stu, value_tea, attention

class AAEmbed(nn.Module):
  """non-linear embed by MLP"""

  def __init__(self, num_input_channels=1024, num_target_channels=128):
    super(AAEmbed, self).__init__()
    self.num_mid_channel = 2 * num_target_channels

    def conv1x1(in_channels, out_channels, stride=1):
      return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

    def conv3x3(in_channels, out_channels, stride=1):
      return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)

    self.regressor = nn.Sequential(
      conv1x1(num_input_channels, self.num_mid_channel),
      nn.BatchNorm2d(self.num_mid_channel),
      nn.ReLU(inplace=True),
      conv3x3(self.num_mid_channel, self.num_mid_channel),
      nn.BatchNorm2d(self.num_mid_channel),
      nn.ReLU(inplace=True),
      conv1x1(self.num_mid_channel, num_target_channels),
    ).cuda()

  def forward(self, x):
    x = self.regressor(x)
    return x

class MLPEmbed(nn.Module):
  """non-linear embed by MLP"""

  def __init__(self, dim_in=1024, dim_out=128):
    super(MLPEmbed, self).__init__()
    self.linear1 = nn.Linear(dim_in, 2 * dim_out).cuda()
    self.relu = nn.ReLU(inplace=True).cuda()
    self.linear2 = nn.Linear(2 * dim_out, dim_out).cuda()
    self.l2norm = Normalize(2).cuda()

  def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = self.relu(self.linear1(x))
    x = self.l2norm(self.linear2(x))
    return x

class Normalize(nn.Module):
  """normalization layer"""

  def __init__(self, power=2):
    super(Normalize, self).__init__()
    self.power = power

  def forward(self, x):
    norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
    out = x.div(norm)
    return out