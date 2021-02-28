# coding:utf-8
# 五子棋程序


import sys, time, os
import random


# 棋盘类
# 从字符串加载棋局或者导出字符串，判断输赢
class chessboard():
    def __init__(self, forbidden = 0, row = 15, col = 15):
        self.__board = [[0 for n in range(row)] for m in range(col)]
        self.__forbidden = forbidden
        self.__dirs = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
        self.DIRS = self.__dirs
        self.won = {}
        self.count = 0    # 落子总数
        self.row = row
        self.col = col
        self.last = [-1, -1]   # 最后一次落子的位置
        
    # 清空棋盘
    def reset(self):
        for j in range(15):
            for i in range(15):
                self.__board[i][j] = 0
        return 0
        
    # 索引器
    def __getitem__(self, row):
        return self.__board[row]
        
    # 将棋盘转换成字符串
    def __str__(self):
        enter = "\n"*10
        text = "  A B C D E F G H I J K L M N O\n"
        text = enter+text
        mark = (". ", "O ", "X ")
        nrow = 0
        for row in self.__board:
            line = ''.join([mark[n] for n in row])
            text += chr(ord('A') + nrow) + ' ' + line
            nrow += 1
            if nrow < 15:
                text += "\n"
        return text
        
    # 转成字符串
    def __repr__(self):
        return self.__str__()
        
    def get(self, row, col):
        if row < 0 or row >= 15 or col < 0 or col >= 15:
            return 0
        return self.__board[row][col]
        
    def put(self, row, col, x):
        if (row >= 0 or row < 15 or col >= 0 or col < 15) and self.__board[row][col] == 0:
            self.__board[row][col] = x
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
        
    # 判断输赢 0-无输赢 1-白赢 2-黑赢
    def check(self):
        board = self.__board
        dirs = ((1, -1), (1, 0), (1, 1), (0, 1))
        for i in range(15):
            for j in range(15):
                if board[i][j] == 0:
                    continue
                id = board[i][j]
                for d in dirs:
                    x, y = j, i
                    count = 0
                    for k in range(15):
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
        for i in range(15):
            for j in range(15):
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
        print("  A B C D E F G H I J K L M N O")
        mark = (". ", "O ", "X ")
        nrow = 0
        self.check()
        for row in range(15):
            print(chr(ord("A") + row))
            for col in range(15):
                ch = self.__board[row][col]
                if ch == 0:
                    print(".")
                elif ch == 1:
                    print("O")
                elif ch == 2:
                    print("X")
            print(" ")
        return 0
        
        
# 双人对战游戏主循环
def gamemain():
    while True:
        os.system("clear")
        print("欢迎进入五子棋游戏!")
        print("1.双人对战.")
        print("2.人机对战(随机算法).")
        print("3.退出.")
        choice = input("请输入您的选择:")
        # print(choice, type(choice), choice.isdigit(), int(choice))
        if choice.isdigit() == False and int(choice) > 3:
            print("输入错误请重新输入")
            input("按任意键继续")
            continue
        choice = int(choice)
        if choice == 1:
            P2P()
        elif choice == 2:
            P2Random()
        elif choice == 3:
            print("再见！")
            exit(0)
    
            
# 双人对战
def P2P():
    who = 1
    b = chessboard()
    b.reset()
    display(b)
    # 游戏循环
    while True:
        # b.show()
        # print(b)
        while True:
            if putchess(who, b) == True:
                break
        # b.show()
        display(b)
        if b.check() == 1:
            print("第%d位游戏者赢了"%(who))
            input("按任意键继续")
            return
        who = who+1
        while True:
            if putchess(who, b) == True:
                break
        if b.check() == 2:
            display(b)
            print("第%d位游戏者赢了"%(who))
            input("按任意键继续")
            return
        display(b)
        who = who+1
        if who > 2:
            who = who - 2
            
            
# 下棋过程
def putchess(who, board):
    down = input("请第%d位游戏者下棋:(输入两位字母坐标，输x退出)"%(who))
    if len(down) == 0:
        print("没有任何输入")
        return False
    if down[0].upper() == 'X':
        print("程序结束")
        exit(1)
    if len(down) != 2 or down[0].upper() < 'A' or down[0].upper() > 'O' or down[1].upper() < 'A' or down[1].upper() > 'O':
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
    
    
# 人机对战的随机算法
def P2Random():
    who = 1
    b = chessboard()
    b.reset()
    display(b)
    # 游戏循环
    while True:
        # b.show()
        # print(b)
        while True:
            if putchess(who, b) == True:
                break
        # b.show()
        display(b)
        if b.check() == 1:
            print("您赢了")
            input("按任意键继续")
            return
        who = who+1
        if nearRandomPut(b, who) == False:
            print("计算机无法落子，游戏结束!")
            input("按任意键继续")
            return
        if b.check() == 2:
            display(b)
            print("电脑赢了!")
            input("按任意键继续")
            return
        display(b)
        who = who+1
        if who > 2:
            who = who - 2
            
            
# 随机算法确定下棋位置
def randomPut(board, who):
    random.seed(time.time())
    if board.full():
        return False
    while True:
        i = random.randint(0, 14)
        j = random.randint(0, 14)
        if board[i][j] == 0:
            board.put(i, j, who)
            return True
    return False
    
    
# 改进随机算法，在对方上次落子位置附近5×5的范围内随机落子
def nearRandomPut(board, who):
    random.seed(time.time())
    if board.full():
        return False
    last = board.getLast()
    x = max(last[0] - 2, 0)
    y = min(last[1] + 2, 14)
    while True:
        i = random.randint(x, y)
        j = random.randint(x, y)
        if board[i][j] == 0:
            board.put(i, j, who)
            return True
    return False


if __name__ == "__main__":
    def test1():
        print("测试1")
        b = chessboard()
        b[10][10] = 1
        b[11][11] = 2
        for i in range(4):
            b[5+i][2+i] = 2
        for i in range(4):
            b[7-0][3+i] = 2
        print(b)
        print("check", b.check())
        return 0
        
    def test2():
        print("测试2")
        b = chessboard()
        b[7][7] = 1
        b[8][8] = 2
        b[7][9] = 1
        
    # test1()
    gamemain()
    # x = input("请输入:")
    # print(x, len(x), x[0], x[1])
    