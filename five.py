# coding:utf-8
# 五子棋程序

from tools import *
from searchTree import *

        
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
        print("7.人机对战(蒙特卡罗树搜索)")
        print("8.比较博弈树算法与蒙特卡罗树搜索算法.")
        print("9.退出.")
        choice = input("请输入您的选择:")
        # print(choice, type(choice), choice.isdigit(), int(choice))
        if choice.isdigit() == False or int(choice) > 9:
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
            P2MCTS()
        elif choice == 8:
            compareTreeMCTS()
        elif choice == 9:
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
        if b.full():
            print("和棋")
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
        if b.full():
            print("和棋")
            input("按任意键继续")
            return
        who = who+1
        if who > 2:
            who = who - 2
            

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
        if b.full():
            print("和棋")
            input("按任意键继续")
            return
        who = who+1
        if nearRandomPut(b, who) == False:
            print("计算机无法落子，游戏结束!")
            input("按任意键继续")
            return
        if b.full():
            print("和棋")
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
    
    
# 比较随机算法和博弈树算法
def compareRandomTree():
    win = [0, 0]
    b = chessboard()
    N = 0
    while True:
        epochs = input("请输入对弈次数:")
        if epochs.isdigit() and int(epochs) > 0:
            epochs = int(epochs)
            break
    for i in tqdm.tqdm(range(epochs)):
        b.reset()
        result = contest(nearRandomPut, treePut, show = False)
        if result != -1:
            win[result-1] += 1
            N += 1
        # print("第%d次对弈，%d取胜" % (i+1, result))
        
    winrate = [win[0]/N, win[1]/N]
    print("获胜概率:", winrate)
    input("按任意键继续")
    
    
# 比较不同搜索深度的博弈树算法
def compareTrees():
    win = [0, 0]
    b = chessboard()
    N = 0
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
        if result != -1:
            win[result-1] += 1
            N += 1
        # print("第%d次对弈，%d取胜" % (i+1, result))
    if N == 0:
        winrate = [0.0, 0.0]
    else:
        winrate = [win[0]/N, win[1]/N]
    print("获胜概率:", winrate)
    input("按任意键继续")
    
    
# 人机对战，蒙特卡罗树搜索算法
def P2MCTS():
    b = chessboard()
    b.reset()
    result = contest(putchess, MCTSput, show = True)
    if result == 1:
        print("您赢了！")
    elif result == 2:
        print("计算机赢了!")
    else:
        print("和棋")
    input("按任意键继续")
    

# 比较博弈树和蒙特卡罗树搜索算法
def compareTreeMCTS():
    win = [0, 0]
    b = chessboard()
    N = 0
    while True:
        epochs = input("请输入对弈次数:")
        if epochs.isdigit() and int(epochs) > 0:
            epochs = int(epochs)
            break
    while True:
        depth1 = input("请输入搜索树的搜索深度:")
        if depth1.isdigit() and int(depth1) > 0:
            depth1 = int(depth1)
            break
    while True:
        n_playout = input("请输入MCTS的搜索次数:")
        if n_playout.isdigit() and int(n_playout) > 0:
            n_playout = int(n_playout)
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
        result = contest(treePut, MCTSput, show = bShow, bTree = True, depth1 = depth1, depth2 = n_playout)
        if result != -1:
            win[result-1] += 1
            N += 1
        # print("第%d次对弈，%d取胜" % (i+1, result))
    if N == 0:
        winrate = [0.0, 0.0]
    else:
        winrate = [win[0]/N, win[1]/N]
    print("获胜概率:", winrate)
    input("按任意键继续")



if __name__ == "__main__":
    gamemain()
    # testsearch()
    