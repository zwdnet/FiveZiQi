# coding:utf-8
# 五子棋程序
# 蒙特卡洛搜索树算法


import numpy as np
import copy
from operator import itemgetter
from tools import *


# 棋盘类 暂时用一下，再改我的
class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        #print("测试")
        #print(self.availables)
        #print(move)
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


# 输入棋盘状态，输出(走法，概率)列表
def policy_value_fn(board):
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0
    
# 更快的版本，用于rollout过程
def rollout_policy_fn(board):
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


# 树的节点类，保存其值Q，先验概率P和访问计数分值u
class TreeNode:
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        
    # 通过新建子节点来扩展树，action_priors是先验概率tuple列表
    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    # 选择P值最大的节点
    def select(self, c_puct):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))
                   
    # 从叶子节点开始更新节点
    def update(self, leaf_value):
        self._n_visits += 1
        # 更新q值，即所有访问的平均值
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        
    # 对所有父节点递归更新
    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)
        
    # 返回节点值，是经过访问次数调整的
    # c_puct是一个大于0的值，调整权重
    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
           np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u
        
    # 判断是否为叶节点
    def is_leaf(self):
        return self._children == {}
        
    # 判断是否为根节点
    def is_root(self):
        return self._parent is None
        
        
# 实现蒙特卡洛树搜索
class MCTS:
    # policy_value_fn 保存一个棋盘状态和输出即一个(走法，概率)列表
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
     
    # 进行一次到叶节点的搜索，并递归更新节点   
    # state会改变，要提供复制版本
    def _playout(self, state):
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # 贪心法选择下一个节点
            action, node = node.select(self._c_puct)
            state.do_move(action)
            
        action_probs, _ = self._policy(state)
        # 检查游戏是否结束
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        # 随机评估叶子节点
        leaf_value = self._evaluate_rollout(state)
        # 递归更新
        node.update_recursive(-leaf_value)
        
    # 运行直到棋局结束，赢返回1，输返回-1， 和棋返回0
    def _evaluate_rollout(self, state, limit=1000):
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        # else:
        #    print("到达迭代限值")
            
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1
            
    # 返回访问最多的走法
    def get_move(self, state):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]
            
    # 反向传播
    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)
            
    def __str__(self):
        return "MCTS"
        
        
# 使用MCTS下棋
class MCTSPlayer:
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        
    def set_player_ind(self, p):
        self.player = p
        
    def reset_player(self):
        self.mcts.update_with_move(-1)
        
    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("棋盘满了")
            
    def __str__(self):
        return "MCTS {}".format(self.player)
        
        
# 输出棋盘
def graphic(board, player1, player2):
    """Draw the board and show game info"""
    os.system("clear")
    width = board.width
    height = board.height

    print("Player", player1, "with X".rjust(3))
    print("Player", player2, "with O".rjust(3))
    print()
    for x in range(width):
        print("{0:8}".format(x), end='')
    print('\r\n')
    for i in range(height - 1, -1, -1):
        print("{0:4d}".format(i), end='')
        for j in range(width):
            loc = i * width + j
            p = board.states.get(loc, -1)
            if p == player1:
                print('X'.center(8), end='')
            elif p == player2:
                print('O'.center(8), end='')
            else:
                print('_'.center(8), end='')
        print('\r\n\r\n')


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
        
    