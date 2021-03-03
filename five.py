# coding:utf-8
# 五子棋程序


import sys, time, os
import random
import tqdm
import run


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
        
    # 判断棋盘是否为空的
    def empty(self):
        if self.count == 0:
            return True
        return False
        
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
        
        
# 游戏主循环
def gamemain():
    while True:
        os.system("clear")
        print("欢迎进入五子棋游戏!")
        print("1.双人对战.")
        print("2.人机对战(随机算法).")
        print("3.比较随机算法.")
        print("4.人机对战(博弈树算法).")
        print("5.比较随机算法和博弈树算法.")
        print("6.比较不同深度的博弈树算法.")
        print("7.退出.")
        choice = input("请输入您的选择:")
        # print(choice, type(choice), choice.isdigit(), int(choice))
        if choice.isdigit() == False or int(choice) > 7:
            print("输入错误请重新输入")
            input("按任意键继续")
            continue
        choice = int(choice)
        if choice == 1:
            P2P()
        elif choice == 2:
            P2Random()
        elif choice == 3:
            compareRandom()
        elif choice == 4:
            P2Tree()
        elif choice == 5:
            compareRandomTree()
        elif choice == 6:
            compareTrees()
        elif choice == 7:
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
            if putchess(b, who) == True:
                break
        # b.show()
        display(b)
        if b.check() == 1:
            print("第%d位游戏者赢了"%(who))
            input("按任意键继续")
            return
        who = who+1
        while True:
            if putchess(b, who) == True:
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
            
            
# 人下棋过程
def putchess(board, who):
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
            if putchess(b, who) == True:
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
    if last == [-1, -1]:
        x = 7
        y = 7
    else:
        x = last[0]
        y = last[1]
    count = 0
    while True:
        count += 1
        #  防止所有附近区域都下满的情况
        if count >= 100: 
            i = random.randint(0, 14)
            j = random.randint(0, 14)
        else:
            i = random.randint(max(x-2, 0), min(x+2, 14))
            j = random.randint(max(y-2, 0), min(y+2, 14))
        if board[i][j] == 0:
            board.put(i, j, who)
            return True
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
    # 游戏循环
    while True:
        while True:
            if bTree == False:
                if methods[first-1](b, first) == True:
                    break
            else:
                if methods[first-1](b, first, depth1) == True:
                    break
        if show:
            display(b)
        if b.check() == first:
            return first
        if bTree == False:
            if methods[second -1](b, second) == False:
                print("无法落子，游戏结束!")
                input("按任意键继续")
                return -1
        else:
            if methods[second -1](b, second, depth2) == False:
                print("无法落子，游戏结束!")
                input("按任意键继续")
                return -1
        if b.check() == second:
            if show:
                display(b)
            return second
        if show:
            display(b)
        
    return -1
    
    
# 比较两种随机算法
def compareRandom():
    win = [0, 0]
    b = chessboard()
    while True:
        epochs = input("请输入对弈次数:")
        if epochs.isdigit() and int(epochs) > 0:
            epochs = int(epochs)
            break
    for i in tqdm.tqdm(range(epochs)):
        b.reset()
        result = contest(randomPut, nearRandomPut, show = False)
        win[result-1] += 1
        # print("第%d次对弈，%d取胜" % (i+1, result))
        
    winrate = [win[0]/epochs, win[1]/epochs]
    print("获胜概率:", winrate)
    input("按任意键继续")
    
    
# ————搜索树算法————
# 参考 https://github.com/skywind3000/gobang/blob/master/gobang.py

#----------------------------------------------------------------------
# evaluation: 棋盘评估类，给当前棋盘打分用
#----------------------------------------------------------------------
class evaluation (object):

	def __init__ (self):
		self.POS = []
		for i in range(15):
			row = [ (7 - max(abs(i - 7), abs(j - 7))) for j in range(15) ]
			self.POS.append(tuple(row))
		self.POS = tuple(self.POS)
		self.STWO = 1		# 冲二
		self.STHREE = 2		# 冲三
		self.SFOUR = 3		# 冲四
		self.TWO = 4		# 活二
		self.THREE = 5		# 活三
		self.FOUR = 6		# 活四
		self.FIVE = 7		# 活五
		self.DFOUR = 8		# 双四
		self.FOURT = 9		# 四三
		self.DTHREE = 10	# 双三
		self.NOTYPE = 11	
		self.ANALYSED = 255		# 已经分析过
		self.TODO = 0			# 没有分析过
		self.result = [ 0 for i in range(30) ]		# 保存当前直线分析值
		self.line = [ 0 for i in range(30) ]		# 当前直线数据
		self.record = []			# 全盘分析结果 [row][col][方向]
		for i in range(15):
			self.record.append([])
			self.record[i] = []
			for j in range(15):
				self.record[i].append([ 0, 0, 0, 0])
		self.count = []				# 每种棋局的个数：count[黑棋/白棋][模式]
		for i in range(3):
			data = [ 0 for i in range(20) ]
			self.count.append(data)
		self.reset()

	# 复位数据
	def reset (self):
		TODO = self.TODO
		count = self.count
		for i in range(15):
			line = self.record[i]
			for j in range(15):
				line[j][0] = TODO
				line[j][1] = TODO
				line[j][2] = TODO
				line[j][3] = TODO
		for i in range(20):
			count[0][i] = 0
			count[1][i] = 0
			count[2][i] = 0
		return 0

	# 四个方向（水平，垂直，左斜，右斜）分析评估棋盘，然后根据分析结果打分
	def evaluate (self, board, turn):
		score = self.__evaluate(board, turn)
		count = self.count
		if score < -9000:
			stone = turn == 1 and 2 or 1
			for i in range(20):
				if count[stone][i] > 0:
					score -= i
		elif score > 9000:
			stone = turn == 1 and 2 or 1
			for i in range(20):
				if count[turn][i] > 0:
					score += i
		return score
	
	# 四个方向（水平，垂直，左斜，右斜）分析评估棋盘，然后根据分析结果打分
	def __evaluate (self, board, turn):
		record, count = self.record, self.count
		TODO, ANALYSED = self.TODO, self.ANALYSED
		self.reset()
		# 四个方向分析
		for i in range(15):
			boardrow = board[i]
			recordrow = record[i]
			for j in range(15):
				if boardrow[j] != 0:
					if recordrow[j][0] == TODO:		# 水平没有分析过？
						self.__analysis_horizon(board, i, j)
					if recordrow[j][1] == TODO:		# 垂直没有分析过？
						self.__analysis_vertical(board, i, j)
					if recordrow[j][2] == TODO:		# 左斜没有分析过？
						self.__analysis_left(board, i, j)
					if recordrow[j][3] == TODO:		# 右斜没有分析过
						self.__analysis_right(board, i, j)

		FIVE, FOUR, THREE, TWO = self.FIVE, self.FOUR, self.THREE, self.TWO
		SFOUR, STHREE, STWO = self.SFOUR, self.STHREE, self.STWO
		check = {}

		# 分别对白棋黑棋计算：FIVE, FOUR, THREE, TWO等出现的次数
		for c in (FIVE, FOUR, SFOUR, THREE, STHREE, TWO, STWO):
			check[c] = 1
		for i in range(15):
			for j in range(15):
				stone = board[i][j]
				if stone != 0:
					for k in range(4):
						ch = record[i][j][k]
						if ch in check:
							count[stone][ch] += 1
		
		# 如果有五连则马上返回分数
		BLACK, WHITE = 1, 2
		if turn == WHITE:			# 当前是白棋
			if count[BLACK][FIVE]:
				return -9999
			if count[WHITE][FIVE]:
				return 9999
		else:						# 当前是黑棋
			if count[WHITE][FIVE]:
				return -9999
			if count[BLACK][FIVE]:
				return 9999
		
		# 如果存在两个冲四，则相当于有一个活四
		if count[WHITE][SFOUR] >= 2:
			count[WHITE][FOUR] += 1
		if count[BLACK][SFOUR] >= 2:
			count[BLACK][FOUR] += 1

		# 具体打分
		wvalue, bvalue, win = 0, 0, 0
		if turn == WHITE:
			if count[WHITE][FOUR] > 0: return 9990
			if count[WHITE][SFOUR] > 0: return 9980
			if count[BLACK][FOUR] > 0: return -9970
			if count[BLACK][SFOUR] and count[BLACK][THREE]: 
				return -9960
			if count[WHITE][THREE] and count[BLACK][SFOUR] == 0:
				return 9950
			if	count[BLACK][THREE] > 1 and \
				count[WHITE][SFOUR] == 0 and \
				count[WHITE][THREE] == 0 and \
				count[WHITE][STHREE] == 0:
					return -9940
			if count[WHITE][THREE] > 1:
				wvalue += 2000
			elif count[WHITE][THREE]:
				wvalue += 200
			if count[BLACK][THREE] > 1:
				bvalue += 500
			elif count[BLACK][THREE]:
				bvalue += 100
			if count[WHITE][STHREE]:
				wvalue += count[WHITE][STHREE] * 10
			if count[BLACK][STHREE]:
				bvalue += count[BLACK][STHREE] * 10
			if count[WHITE][TWO]:
				wvalue += count[WHITE][TWO] * 4
			if count[BLACK][TWO]:
				bvalue += count[BLACK][TWO] * 4
			if count[WHITE][STWO]:
				wvalue += count[WHITE][STWO]
			if count[BLACK][STWO]:
				bvalue += count[BLACK][STWO]
		else:
			if count[BLACK][FOUR] > 0: return 9990
			if count[BLACK][SFOUR] > 0: return 9980
			if count[WHITE][FOUR] > 0: return -9970
			if count[WHITE][SFOUR] and count[WHITE][THREE]:
				return -9960
			if count[BLACK][THREE] and count[WHITE][SFOUR] == 0:
				return 9950
			if	count[WHITE][THREE] > 1 and \
				count[BLACK][SFOUR] == 0 and \
				count[BLACK][THREE] == 0 and \
				count[BLACK][STHREE] == 0:
					return -9940
			if count[BLACK][THREE] > 1:
				bvalue += 2000
			elif count[BLACK][THREE]:
				bvalue += 200
			if count[WHITE][THREE] > 1:
				wvalue += 500
			elif count[WHITE][THREE]:
				wvalue += 100
			if count[BLACK][STHREE]:
				bvalue += count[BLACK][STHREE] * 10
			if count[WHITE][STHREE]:
				wvalue += count[WHITE][STHREE] * 10
			if count[BLACK][TWO]:
				bvalue += count[BLACK][TWO] * 4
			if count[WHITE][TWO]:
				wvalue += count[WHITE][TWO] * 4
			if count[BLACK][STWO]:
				bvalue += count[BLACK][STWO]
			if count[WHITE][STWO]:
				wvalue += count[WHITE][STWO]
		
		# 加上位置权值，棋盘最中心点权值是7，往外一格-1，最外圈是0
		wc, bc = 0, 0
		for i in range(15):
			for j in range(15):
				stone = board[i][j]
				if stone != 0:
					if stone == WHITE:
						wc += self.POS[i][j]
					else:
						bc += self.POS[i][j]
		wvalue += wc
		bvalue += bc
		
		if turn == WHITE:
			return wvalue - bvalue

		return bvalue - wvalue
	
	# 分析横向
	def __analysis_horizon (self, board, i, j):
		line, result, record = self.line, self.result, self.record
		TODO = self.TODO
		for x in range(15):
			line[x] = board[i][x]
		self.analysis_line(line, result, 15, j)
		for x in range(15):
			if result[x] != TODO:
				record[i][x][0] = result[x]
		return record[i][j][0]
	
	# 分析横向
	def __analysis_vertical (self, board, i, j):
		line, result, record = self.line, self.result, self.record
		TODO = self.TODO
		for x in range(15):
			line[x] = board[x][j]
		self.analysis_line(line, result, 15, i)
		for x in range(15):
			if result[x] != TODO:
				record[x][j][1] = result[x]
		return record[i][j][1]
	
	# 分析左斜
	def __analysis_left (self, board, i, j):
		line, result, record = self.line, self.result, self.record
		TODO = self.TODO
		if i < j: x, y = j - i, 0
		else: x, y = 0, i - j
		k = 0
		while k < 15:
			if x + k > 14 or y + k > 14:
				break
			line[k] = board[y + k][x + k]
			k += 1
		self.analysis_line(line, result, k, j - x)
		for s in range(k):
			if result[s] != TODO:
				record[y + s][x + s][2] = result[s]
		return record[i][j][2]

	# 分析右斜
	def __analysis_right (self, board, i, j):
		line, result, record = self.line, self.result, self.record
		TODO = self.TODO
		if 14 - i < j: x, y, realnum = j - 14 + i, 14, 14 - i
		else: x, y, realnum = 0, i + j, j
		k = 0
		while k < 15:
			if x + k > 14 or y - k < 0:
				break
			line[k] = board[y - k][x + k]
			k += 1
		self.analysis_line(line, result, k, j - x)
		for s in range(k):
			if result[s] != TODO:
				record[y - s][x + s][3] = result[s]
		return record[i][j][3]
	
	def test (self, board):
		self.reset()
		record = self.record
		TODO = self.TODO
		for i in range(15):
			for j in range(15):
				if board[i][j] != 0 and 1:
					if self.record[i][j][0] == TODO:
						self.__analysis_horizon(board, i, j)
						pass
					if self.record[i][j][1] == TODO:
						self.__analysis_vertical(board, i, j)
						pass
					if self.record[i][j][2] == TODO:
						self.__analysis_left(board, i, j)
						pass
					if self.record[i][j][3] == TODO:
						self.__analysis_right(board, i, j)
						pass
		return 0
	
	# 分析一条线：五四三二等棋型
	def analysis_line (self, line, record, num, pos):
		TODO, ANALYSED = self.TODO, self.ANALYSED
		THREE, STHREE = self.THREE, self.STHREE
		FOUR, SFOUR = self.FOUR, self.SFOUR
		while len(line) < 30: line.append(0xf)
		while len(record) < 30: record.append(TODO)
		for i in range(num, 30):
			line[i] = 0xf
		for i in range(num):
			record[i] = TODO
		if num < 5:
			for i in range(num): 
				record[i] = ANALYSED
			return 0
		stone = line[pos]
		inverse = (0, 2, 1)[stone]
		num -= 1
		xl = pos
		xr = pos
		while xl > 0:		# 探索左边界
			if line[xl - 1] != stone: break
			xl -= 1
		while xr < num:		# 探索右边界
			if line[xr + 1] != stone: break
			xr += 1
		left_range = xl
		right_range = xr
		while left_range > 0:		# 探索左边范围（非对方棋子的格子坐标）
			if line[left_range - 1] == inverse: break
			left_range -= 1
		while right_range < num:	# 探索右边范围（非对方棋子的格子坐标）
			if line[right_range + 1] == inverse: break
			right_range += 1
		
		# 如果该直线范围小于 5，则直接返回
		if right_range - left_range < 4:
			for k in range(left_range, right_range + 1):
				record[k] = ANALYSED
			return 0
		
		# 设置已经分析过
		for k in range(xl, xr + 1):
			record[k] = ANALYSED
		
		srange = xr - xl

		# 如果是 5连
		if srange >= 4:	
			record[pos] = self.FIVE
			return self.FIVE
		
		# 如果是 4连
		if srange == 3:	
			leftfour = False	# 是否左边是空格
			if xl > 0:
				if line[xl - 1] == 0:		# 活四
					leftfour = True
			if xr < num:
				if line[xr + 1] == 0:
					if leftfour:
						record[pos] = self.FOUR		# 活四
					else:
						record[pos] = self.SFOUR	# 冲四
				else:
					if leftfour:
						record[pos] = self.SFOUR	# 冲四
			else:
				if leftfour:
					record[pos] = self.SFOUR		# 冲四
			return record[pos]
		
		# 如果是 3连
		if srange == 2:		# 三连
			left3 = False	# 是否左边是空格
			if xl > 0:
				if line[xl - 1] == 0:	# 左边有气
					if xl > 1 and line[xl - 2] == stone:
						record[xl] = SFOUR
						record[xl - 2] = ANALYSED
					else:
						left3 = True
				elif xr == num or line[xr + 1] != 0:
					return 0
			if xr < num:
				if line[xr + 1] == 0:	# 右边有气
					if xr < num - 1 and line[xr + 2] == stone:
						record[xr] = SFOUR	# XXX-X 相当于冲四
						record[xr + 2] = ANALYSED
					elif left3:
						record[xr] = THREE
					else:
						record[xr] = STHREE
				elif record[xl] == SFOUR:
					return record[xl]
				elif left3:
					record[pos] = STHREE
			else:
				if record[xl] == SFOUR:
					return record[xl]
				if left3:
					record[pos] = STHREE
			return record[pos]
		
		# 如果是 2连
		if srange == 1:		# 两连
			left2 = False
			if xl > 2:
				if line[xl - 1] == 0:		# 左边有气
					if line[xl - 2] == stone:
						if line[xl - 3] == stone:
							record[xl - 3] = ANALYSED
							record[xl - 2] = ANALYSED
							record[xl] = SFOUR
						elif line[xl - 3] == 0:
							record[xl - 2] = ANALYSED
							record[xl] = STHREE
					else:
						left2 = True
			if xr < num:
				if line[xr + 1] == 0:	# 左边有气
					if xr < num - 2 and line[xr + 2] == stone:
						if line[xr + 3] == stone:
							record[xr + 3] = ANALYSED
							record[xr + 2] = ANALYSED
							record[xr] = SFOUR
						elif line[xr + 3] == 0:
							record[xr + 2] = ANALYSED
							record[xr] = left2 and THREE or STHREE
					else:
						if record[xl] == SFOUR:
							return record[xl]
						if record[xl] == STHREE:
							record[xl] = THREE
							return record[xl]
						if left2:
							record[pos] = self.TWO
						else:
							record[pos] = self.STWO
				else:
					if record[xl] == SFOUR:
						return record[xl]
					if left2:
						record[pos] = self.STWO
			return record[pos]
		return 0
	def textrec (self, direction = 0):
		text = []
		for i in range(15):
			line = ''
			for j in range(15):
				line += '%x '%(self.record[i][j][direction] & 0xf)
			text.append(line)
		return '\n'.join(text)

        
    
# 深度优先搜索树
class searcher:
    # 初始化
    def __init__(self, row = 15, col = 15):
        self.evaluator = evaluation()
        self.row = row
        self.col = col
        self.board = [[0 for n in range(self.row)] for i in range(self.col)]
        self.gameover = 0
        self.overvalue = 0
        self.maxdepth = 3
        
    # 产生当前棋局的走法
    def genmove(self, turn):
        moves = []
        board = self.board
        POSES = self.evaluator.POS
        for i in range(self.row):
            for j in range(self.col):
                if board[i][j] == 0:
                    score = POSES[i][j]
                    moves.append((score, i, j))
        moves.sort()
        moves.reverse()
        return moves
        
    # 递归搜索，返回最佳分数
    def __search(self, turn, depth, alpha = -0x7fffffff, beta = 0x7fffffff):
        # 深度为零，评估棋盘并返回
        if depth <= 0:
            score = self.evaluator.evaluate(self.board, turn)
            return score
            
        # 游戏结束，立马返回
        score = self.evaluator.evaluate(self.board, turn)
        if abs(score) >= 9999 and depth < self.maxdepth:
            return score
            
        # 产生新的走法
        moves = self.genmove(turn)
        bestmove = None
        
        # 枚举当前所有走法
        for score, row, col in moves:
            # 标记当前走法到棋盘
            self.board[row][col] = turn
            # 计算下一回合该谁走
            nturn = turn == 1 and 2 or 1
            # 深度优先搜索，返回评分，走的行和列
            score = -self.__search(nturn, depth-1, -beta, -alpha)
            # 棋盘上清除当前走法
            self.board[row][col] = 0
            
            # 计算最好分值的走法
            # alpha/beta剪枝
            if score > alpha:
                alpha = score
                bestmove = (row, col)
                if alpha >= beta:
                    break
                
        # 如果是第一层，记录最好的走法
        if depth == self.maxdepth and bestmove:
            self.bestmove = bestmove
            
        # 返回当前最好分数，及对应走法
        return alpha
        
    # 具体搜索
    def search(self, turn, depth = 3):
        self.maxdepth = depth
        self.bestmove = None
        score = self.__search(turn, depth)
        if abs(score) > 8000:
            self.maxdepth = depth
            score = self.__search(turn, 1)
        row, col = self.bestmove
        return score, row, col
        
        
# 测试搜索算法
def testsearch():
    b = chessboard()
    s = searcher()
    s.board = b.board()
    
    
# 人机对战，博弈树算法
def P2Tree():
    b = chessboard()
    b.reset()
    result = contest(putchess, treePut, show = True)
    if result == 1:
        print("您赢了！")
    elif result == 2:
        print("计算机赢了!")
    else:
        print("和棋")
    input("按任意键继续")
    
    
# 博弈树算法下棋过程
def treePut(board, who, depth = 1):
    # 如果是先手，随机下一个地方
    last = board.getLast()
    if last == [-1, -1]:
        row = random.randint(0, 14)
        col = random.randint(0, 14)
        if board[row][col] == 0:
            board.put(row, col, who)
            return True
        return False
    s = searcher()
    s.board = board
    
    # 设置难度
    DEPTH = depth
    score, row, col = s.search(who, DEPTH)
    if board[row][col] == 0:
        board.put(row, col, who)
        return True
    return False
    
    
# 比较随机算法和博弈树算法
def compareRandomTree():
    win = [0, 0]
    b = chessboard()
    while True:
        epochs = input("请输入对弈次数:")
        if epochs.isdigit() and int(epochs) > 0:
            epochs = int(epochs)
            break
    for i in tqdm.tqdm(range(epochs)):
        b.reset()
        result = contest(nearRandomPut, treePut, show = False)
        win[result-1] += 1
        # print("第%d次对弈，%d取胜" % (i+1, result))
        
    winrate = [win[0]/epochs, win[1]/epochs]
    print("获胜概率:", winrate)
    input("按任意键继续")
    
    
# 比较不同搜索深度的博弈树算法
def compareTrees():
    win = [0, 0]
    b = chessboard()
    while True:
        epochs = input("请输入对弈次数:")
        if epochs.isdigit() and int(epochs) > 0:
            epochs = int(epochs)
            break
    while True:
        depth1 = input("请输入算法1的搜索深度:")
        if depth1.isdigit() and int(depth1) > 0:
            depth1 = int(depth1)
            break
    while True:
        depth2 = input("请输入算法2的搜索深度:")
        if depth2.isdigit() and int(depth2) > 0:
            depth2 = int(depth2)
            break
    while True:
        bShow = input("是否输出棋盘?(Y/N)")
        if bShow.upper() == "Y":
            bShow = True
            break
        elif bShow.upper() == "N":
            bShow = False
            break
    for i in tqdm.tqdm(range(epochs)):
        b.reset()
        result = contest(treePut, treePut, show = bShow, bTree = True, depth1 = depth1, depth2 = depth2)
        win[result-1] += 1
        # print("第%d次对弈，%d取胜" % (i+1, result))
        
    winrate = [win[0]/epochs, win[1]/epochs]
    print("获胜概率:", winrate)
    input("按任意键继续")


if __name__ == "__main__":
    gamemain()
    # testsearch()
    