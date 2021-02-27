# coding:utf-8
# 五子棋程序


import sys, time, os


# 棋盘类
# 从字符串加载棋局或者导出字符串，判断输赢
class chessboard():
    def __init__(self, forbidden = 0):
        self.__board = [[0 for n in range(15)] for m in range(15)]
        self.__forbidden = forbidden
        self.__dirs = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
        self.DIRS = self.__dirs
        self.won = {}
        
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
        text = "  A B C D E F G H I J K L M N O\n"
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
            return 0
        else:
            return 1
        
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
            return
        who = who+1
        while True:
            if putchess(who, b) == True:
                break
        if b.check() == 2:
            display(b)
            print("第%d位游戏者赢了"%(who))
            return
        display(b)
        who = who+1
        if who > 2:
            who = who - 2
            
            
# 下棋过程
def putchess(who, board):
    down = input("请第%d位游戏者下棋:"%(who))
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
    