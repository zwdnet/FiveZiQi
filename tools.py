# coding:utf-8
# 五子棋程序
# 工具程序


import sys, time, os
import random
import tqdm
import run
from searchTree import *
from mcts import *


# 棋盘类
# 从字符串加载棋局或者导出字符串，判断输赢
class chessboard():
    def __init__(self, forbidden = 0, row = 8, col = 8):
        self.__board = [[0 for n in range(row)] for m in range(col)]
        self.__forbidden = forbidden
        self.__dirs = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
        self.DIRS = self.__dirs
        self.won = {}
        self.count = 0    # 落子总数
        self.row = row
        self.col = col
        self.last = [-1, -1]   # 最后一次落子的位置
        # 为适应蒙特卡罗树搜索加的内容
        self.states = {}
        self.availables = list(range(self.row * self.col))
        self.current_player = 0
        self.players = [1, 2]
        self.width = row
        self.height = col
        
    # 清空棋盘
    def reset(self):
        for j in range(self.col):
            for i in range(self.row):
                self.__board[i][j] = 0
        self.states = {}
        self.availables = list(range(self.row * self.col))
        self.current_player = 0
        return 0
        
    # 索引器
    def __getitem__(self, row):
        return self.__board[row]
        
    # 将棋盘转换成字符串
    def __str__(self):
        enter = "\n"*10
        player = "棋手1:O\n"
        player += "棋手2:X\n"
        text = "  A B C D E F G H\n"
        text = enter+player+text
        mark = (". ", "O ", "X ")
        nrow = 0
        for row in self.__board:
            line = ''.join([mark[n] for n in row])
            text += chr(ord('A') + nrow) + ' ' + line
            nrow += 1
            if nrow < self.row:
                text += "\n"
        last_row = chr(ord('A') + self.last[0])
        last_col = chr(ord('A') + self.last[1])
        text += "\n最后落子坐标:" + last_row + last_col
        return text
        
    # 转成字符串
    def __repr__(self):
        return self.__str__()
        
    def get(self, row, col):
        if row < 0 or row >= self.row or col < 0 or col >= self.col:
            return 0
        return self.__board[row][col]
        
    def put(self, row, col, x):
        if (row >= 0 and row < self.row and col >= 0 and col < self.col) and self.__board[row][col] == 0:
            self.__board[row][col] = x
            #if x == 1:
#                self.current_player = 2
#            else:
#                self.current_player = 1
            self.count += 1
            self.last[0] = row
            self.last[1] = col
            return 0
        else:
            return 1
            
    # 获得最后一次落子的位置
    def getLast(self):
        return self.last
            
    # 判断是否棋盘已满
    def full(self):
        if self.count < self.row*self.col-1:
            return False
        return True
        
    # 判断棋盘是否为空的
    def empty(self):
        if self.count == 0:
            return True
        return False
        
    # 判断输赢 0-无输赢 1-白赢 2-黑赢
    def check(self):
        board = self.__board
        dirs = ((1, -1), (1, 0), (1, 1), (0, 1))
        for i in range(self.row):
            for j in range(self.col):
                if board[i][j] == 0:
                    continue
                id = board[i][j]
                for d in dirs:
                    x, y = j, i
                    count = 0
                    for k in range(self.row):
                        if self.get(y, x) != id:
                            break
                        y += d[0]
                        x += d[1]
                        count += 1
                    if count == 5:
                        self.won = {}
                        r, c = i, j
                        for z in range(5):
                            self.won[(r, c)] = 1
                            r += d[0]
                            c += d[1]
                        return id
        return 0
        
    # 返回数组对象
    def board(self):
        return self.__board
        
    # 导出棋局到字符串
    def dump(self):
        import StringIO
        sio = StringIO.StringIO
        board = self.__board
        for i in range(self.row):
            for j in range(self.col):
                stone = board[i][j]
                if stone != 0:
                    ti = chr(ord("A") + i)
                    tj = chr(ord("A") + j)
                    sio.write("%d:%s%s "%(stone, ti, tj))
        return sio.getvalue()
        
    # 从字符串加载棋局
    def load(self, text):
        self.reset()
        board = self.__board
        for item in text.strip('\r\n\t ').replace(',', ' ').split(' '):
            n = item.strip("\n\r\t")
            if not n:
                continue
            n = n.split(":")
            stone = int(n[0])
            i = ord(n[1][0].upper()) - ord('A')
            j = ord(n[1][1].upper()) - ord('A')
            board[i][j] = store
        return 0
        
    # 输出
    def show(self):
        print("棋手1:O")
        print("棋手2:X")
        print("   A   B   C   D   E   F   G   H  ")
        mark = (" . ", " O ", " X ")
        nrow = 0
        self.check()
        for row in range(self.row):
            print(chr(ord("A") + row))
            for col in range(self.col):
                ch = self.__board[row][col]
                if ch == 0:
                    print(" . ")
                elif ch == 1:
                    print(" O ")
                elif ch == 2:
                    print(" X ")
            print(" ")
        return 0
        
    # 为蒙特卡罗树搜索增加的
    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        row = move // self.row
        col = move % self.row
        return [row, col]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        row = location[0]
        col = location[1]
        move = row * self.row + col
        if move not in range(self.row * self.col):
            return -1
        return move
    
    def game_end(self):
        id = self.check()
        if id == 0:
            return False, -1
        return True, id
        
    def get_current_player(self):
        return self.current_player
        
    def do_move(self, move):
        if self.full():
            return False
        self.states[move] = self.current_player
        row, col = self.move_to_location(move)
        self.put(row, col, self.current_player)
        self.availables.remove(move)
        # print("测试a", self.availables)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        return True
        # print("测试b", self.current_player, move, row, col)
        # self.last_move = move
        
    def set_current_player(self, who):
        if who not in (1, 2):
            return
        self.current_player = who
        
        
# 人下棋过程
def putchess(board, who):
    down = input("请第%d位游戏者下棋:(输入两位字母坐标，输x退出)"%(who))
    if len(down) == 0:
        print("没有任何输入")
        return False
    if down[0].upper() == 'X':
        print("程序结束")
        exit(1)
    if len(down) != 2 or down[0].upper() < 'A' or down[0].upper() > 'H' or down[1].upper() < 'A' or down[1].upper() > 'H':
        print("输入错误，请输入两位字母。")
        return False
    i = ord(down[0].upper()) - ord('A')
    j = ord(down[1].upper()) - ord('A')
    if board.put(i, j, who) == 1:
        print("不能在有棋子的地方再落棋，请重新输入。")
        return False
    return True
    
    
 # 输出棋盘
def display(board):
    os.system("clear")
    print(board)
    
    
# 随机算法确定下棋位置
def randomPut(board, who):
    random.seed(time.time())
    if board.full():
        return False
    while True:
        i = random.randint(0, 7)
        j = random.randint(0, 7)
        if board[i][j] == 0:
            board.put(i, j, who)
            return True
    return False
    
    
# 改进随机算法，在对方上次落子位置附近5×5的范围内随机落子
def nearRandomPut(board, who):
    random.seed(time.time())
    if board.full():
        # print("a")
        return False
    last = board.getLast()
    if last == [-1, -1]:
        x = 4
        y = 4
    else:
        x = last[0]
        y = last[1]
    count = 0
    while True:
        count += 1
        #  防止所有附近区域都下满的情况
        if count >= 100: 
            i = random.randint(0, 7)
            j = random.randint(0, 7)
        else:
            i = random.randint(max(x-2, 0), min(x+2, 7))
            j = random.randint(max(y-2, 0), min(y+2, 7))
        if board[i][j] == 0:
            board.put(i, j, who)
            # print("b")
            # print(i, j, who, board.full())
            return True
    # print("c")
    return False
    
    
# 算法与算法(包括人)之间对弈的一般过程
def contest(method1, method2, show = True, bTree = False, depth1 = 0, depth2 = 0):
    random.seed(time.time())
    methods = [method1, method2]
    b = chessboard()
    b.reset()
    if show:
        display(b)
    first = random.randint(1, 2)
    if first == 1:
        second = 2
    else:
        second = 1
    # print(first, second)
    # input("按任意键继续")
    # 游戏循环
    while True:
        while True:
            b.set_current_player(first)
            if bTree == False:
                if methods[first-1](b, first) == True:
                    break
            else:
                if first == 1:
                    param = depth1
                else:
                    param = depth2
                if methods[first-1](b, first, param) == True:
                    break
        if show:
            display(b)
        if b.check() == first:
            return first
        b.set_current_player(second)
        if bTree == False:
            if methods[second -1](b, second) == False:
                if b.full() == True:
                    print("无法落子，游戏结束!")
                return -1
        else:
            if second == 1:
                param = depth1
            else:
                param = depth2
            if methods[second -1](b, second, param) == False:
                if b.full() == True:
                    print("无法落子，游戏结束!")
                return -1
        if b.check() == second:
            if show:
                display(b)
            return second
        if show:
            display(b)
        
    return -1
    
    
# 博弈树算法下棋过程
def treePut(board, who, depth = 2):
    # 如果是先手，随机下一个地方
    last = board.getLast()
    if last == [-1, -1]:
        row = random.randint(2, 5)
        col = random.randint(2, 5)
        if board[row][col] == 0:
            board.put(row, col, who)
            return True
        return False
    s = searcher()
    s.board = board
    if s.board.full():
        return False
    
    # 设置难度
    DEPTH = depth
    score, row, col = s.search(who, DEPTH)
    if board[row][col] == 0:
        board.put(row, col, who)
        return True
    return False
    
    
# 蒙特卡罗树搜索落子
def MCTSput(board, who, n_playout = 400):
    # print("n_playout=", n_playout)
    # input("按任意键继续")
    player = MCTSPlayer(c_puct = 5,
                                     n_playout = n_playout)
    # 设置当前下棋者，使用do_move的才要
    # board.set_current_player(who)
    # 如果是先手，随机下一个地方
    last = board.getLast()
    if last == [-1, -1]:
        row = random.randint(2, 5)
        col = random.randint(2, 5)
        if board[row][col] == 0:
            move = board.location_to_move((row, col))
            if board.do_move(move):
                return True
        return False
    # 不是先手
    move = player.get_action(board)
    #print(board.current_player, who)
#    input("按任意键继续")
    return board.do_move(move)
    