# coding:utf-8
# 五子棋程序
# 定义策略网络


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import run


# 设置学习率
def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
# 策略神经网络
class Net(nn.Module):
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        
        self.board_width = board_width
        self.board_height = board_height
        # 普通层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 行动策略层
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height, board_width*board_height)
        # 状态值层
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)
        
    # 前向传播过程
    def forward(self, state_input):
        # 普通层
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 行动策略层
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act))
        # 状态值层
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val
        
        
# 策略值网络
class PolicyValueNet:
    def __init__(self, board_width, board_height,
          model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4 # l2系数
        # 策略值网络
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)
        
        if model_file:
            net_params = torch.load(model_file, encoding = "bytes")
            self.policy_value_net.load_state_dict(net_params)

    # 输入，一批状态值
    # 输出，走法概率和状态值
    def policy_value(self, state_batch):
        if self.use_gpu:
            state_batch = torch.Tensor(state_batch).cuda()
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = torch.Tensor(state_batch)
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()
            
    # 输入棋盘状态，对每个可能的走法输出其(走法,概率)列表，以及棋盘评分
    def policy_value_fn(self, board):
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(torch.from_numpy(current_state)).cuda().float()
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(torch.from_numpy(current_state)).float()
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value
        
    # 进行一次训练步骤
    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        if self.use_gpu:
            state_batch = torch.Tensor(state_batch).cuda()
            mcts_probs = torch.Tensor(mcts_probs).cuda()
            winner_batch = torch.Tensor(winner_batch).cuda()
        else:
            state_batch = torch.Tensor(state_batch)
            mcts_probs = torch.Tensor(mcts_probs)
            winner_batch = torch.Tensor(winner_batch)
            
        # 参数梯度归零
        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)
        
        # 前向传播
        log_act_probs, value = self.policy_value_net(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # 反向传播
        loss.backward()
        self.optimizer.step()
        # 计算误差
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item()
        
    # 获得策略参数
    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params
        
    # 保存模型到文件
    @run.change_dir
    def save_model(self, model_file):
        net_params = self.get_policy_param()
        torch.save(net_params, model_file)


if __name__ == "__main__":
    policy_value_net = PolicyValueNet(8, 8)
    