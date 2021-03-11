# coding:utf-8
# 五子棋程序
# 强化学习训练过程


import random
import numpy as np
import tqdm
from collections import defaultdict, deque
from mcts import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import AlphaZeroMctsPlayer as MCTSPlayer
from policy_value_net import PolicyValueNet
from tools import *
import run
import matplotlib.pyplot as plt


class TrainPipeline:
    def __init__(self, init_model=None):
        # 棋盘数据
        self.board_width = 8
        self.board_height = 8
        # self.n_in_row = 5
        self.board = chessboard(row=self.board_width, col=self.board_height)
        # 训练参数
        self.learn_rate = 2e-4
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = 400 # 每次模拟次数
        self.c_puct = 5
        self.buffer_size = 10000000
        self.batch_size = 512 # 每批样本量
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5 # 每次更新前迭代次数
        self.kl_targ = 0.02
        self.check_freq = 2
        # 自我对弈次数
        self.game_batch_num = 10
        self.best_win_ratio = 0.0
        # 纯蒙特卡罗树搜索，用来作为基准
        self.pure_mcts_playout_num = 400
        # 有预训练模型的情况
        if init_model:
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model)
        else:
            # 从头开始训练
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)
        
    # 扩充训练数据
    def get_equi_data(self, play_data):
        # 用旋转和翻转来设置数据
        # play_data:[(state, mcts_prob, winner_z), ..., ...]
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # 顺时针旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # 垂直翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data
        
    # 进行一轮自我博弈
    def start_self_play(self, player, is_shown=0, temp=1e-3):
        self.board.reset()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        # 测试
        # t = 0
        while True:
            # t += 1
            # print(t)
            move, move_probs = player.get_action(self.board, temp = temp, return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                display(self.board)
            end, winner = self.board.game_end()
            # print(t, end, winner, self.board.count)
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
        
    # 收集自我博弈训练数据
    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            # print("测试", i)
            winner, play_data = self.start_self_play(self.mcts_player, temp=self.temp, is_shown = False)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
            
    # 更新策略值网络
    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4: # 早期停止
                break
        # 调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy
        
    # 进行一局对弈
    def start_play(self, player1, player2, start_player=1, is_shown=1):
        if start_player not in (1, 2):
            raise Exception('start_player should be either 0 (player1 first) ''or 1 (player2 first)')
        self.board.reset(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            display(self.board)
        while True:
            current_player = self.board.get_current_player()
            # print(current_player, players)
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                display(self.board)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner
        
    # 策略评估，用纯蒙特卡罗树搜索来做基准
    def policy_evaluate(self, n_games=10):
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.start_play(current_mcts_player, pure_mcts_player, start_player=i % 2 + 1, is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio
        
    # 运行训练
    @run.change_dir
    @run.timethis
    def run(self):
        try:
            losses = []
            for i in tqdm.tqdm(range(self.game_batch_num)):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))
                # 测试用的
                # self.policy_value_net.save_model('./output/best_policy.model')
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    losses.append(loss)
                    # print(i, loss)
                # 检查当前模型表现并保存模型
                if (i+1) % self.check_freq == 0:
                    print("当前自训练次数: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./output/current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("新的最佳策略!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./output/best_policy.model')
                        if (self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
            plt.figure()
            plt.plot(losses)
            plt.savefig("./output/loss.png")
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == "__main__":
    training_pipeline = TrainPipeline()
    training_pipeline.run()
    