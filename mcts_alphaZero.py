# coding:utf-8
# 五子棋程序
# 强化学习用的蒙特卡洛搜索树算法
# 使用了一个策略值网络来引导搜索和评估


import numpy as np
import copy
from mcts import *
# from tools import *


# alphazero用的mcts下棋
class AlphaZeroMctsPlayer:
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay = 0):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        
    def set_player_ind(self, p):
        self.player = p
        
    def reset_player(self):
        self.mcts.update_with_move(-1)
            
    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # 增加狄利克雷分布噪音，自己训练需要
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # 更新根节点
                self.mcts.update_with_move(move)
            else:
                # 默认参数temp = 1e-3，近似等于选择概率最高的叶子节点
                move = np.random.choice(acts, p=probs)
                # 重置根节点
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("棋盘满了")
            
    def __str__(self):
        return "MCTS {}".format(self.player)


if __name__ == "__main__":
    player1 = MCTSPlayer(c_puct = 5,
                                     n_playout = 50)
    player2 = MCTSPlayer(c_puct = 5,
                                     n_playout = 100)
    board = Board(width=8, height=8)
    board2 = chessboard(row = 8, col = 8)
    p1 = 1
    p2 = 2
    board.init_board(p1)
    player1.set_player_ind(p1)
    player2.set_player_ind(p2)
    players = {p1: player1, p2: player2}
    first = random.randint(1, 2)
    if first == 1:
        second = 2
    else:
        second = 1
    methods = [player1, player2]
    while True:
        current_player = board.get_current_player()
        player_in_turn = players[current_player]
        # move = player_in_turn.get_action(board)
        move2 = methods[first-1].get_action(board2)
        #print("测试", move)
#        h = move // 8
#        w = move % 8
#        print(h, w)
        # input()
        # board.do_move(move)
        if board2.do_move(move2) == False:
            print("和棋")
        #graphic(board2, player1.player, player2.player)
        display(board2)
        end, winner = board2.game_end()
        if end:
            if winner != -1:
                print("Game end. Winner is", players[winner])
            else:
                print("Game end. Tie")
            break
        move2 = methods[second-1].get_action(board2)
        #print("测试", move2)
#        h = move // 8
#        w = move % 8
#        print(h, w)
        # input()
        # board.do_move(move)
        if board2.do_move(move2) == False:
            print("和棋")
        # graphic(board2, player1.player, player2.player)
        display(board2)
        end, winner = board2.game_end()
        if end:
            if winner != -1:
                print("Game end. Winner is", players[winner])
            else:
                print("Game end. Tie")
            break
    