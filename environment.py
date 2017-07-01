'''
该文件主要用于计算下了一步棋的回报值，以及是否结束棋局
'''

import numpy as np

UNIT = 40   # pixels
Gobang_H = 19 # grid height
Gobang_W = 19  # grid width
value1=50000         # *****
value2=4320             # +****+ 会算成两次下面
value3=720             # -****+   or  +****-
value4=720              # +***+
value5=720              # -***++   or  ++***-  +**+* *+**+
value6=120              # ++**+   0r  +**++  不计算**+++这种
value7=720               #  **+**     or *+***  or ***+*
value8=720                #  *++**  or **++*
value9=0                 # *+*+*
value10=1000
class env():
    def __init__(self):
        self.action_space = np.arange(361) #定义动作的空间，[0,0....0]
        self.n_actions = 361    #定义动作的可能个数
        self.n_features = 361                    #????????
        self.qipan = np.zeros([19, 19],dtype=np.int)      #定义棋盘矩阵19*19，初始值为0
        self.num1 = 0  # 白棋White
        self.num2 = 0  # 黑棋Black
        self.done=False


    #初始化，重置棋盘
    def reset(self):
        self.qipan=np.zeros([19,19],dtype=np.int)
        self.done=False
        return np.copy(self.qipan)

    def pwb(self,flag):       #The possibility of winning before 之前赢的可能性  status为棋局，flag为当前方
        p1=self.pww(flag)
        return p1

    def pwn(self,action,flag):                       #the possibility of winning next step 下一步赢的可能性
        # print("计算当前方走完下一步赢的可能性")
        # '''这里需修改qipan的状态'''
        ##1为计算方的棋子white
        ##2为对方的棋子black
        x=action//19
        y=action%19
        if flag=='White':
            self.qipan[x,y]=1
        else:
            self.qipan[x,y]=2
        p2=self.pww(flag)
        return p2
    def pww(self,flag):       # The probability of winning the white
        # print("计算传入方flag赢的可能性")
        self.num1=0    #白棋#白棋对应目标为’x‘ 2
        self.num2=0    #黑棋#黑棋为’o‘ 1

        '''计算左斜线方向'''
        for i in range(4, 19):      #计算上半部分，长度为5
            for j in range(i - 4 + 1):
                listcase=[self.qipan[j, i - j],self.qipan[j + 1, i - j - 1],self.qipan[j + 2, i - j - 2],
                          self.qipan[j + 3, i - j - 3],self.qipan[j + 4, i - j - 4]]
                self.match(listcase)
        for i in range(5, 19):     #计算上半部分，长度为6
            for j in range(i - 4):
                listcase=[self.qipan[j, i - j],self.qipan[j + 1, i - j - 1],self.qipan[j + 2, i - j - 2],
                          self.qipan[j + 3, i - j - 3],self.qipan[j + 4, i - j - 4],self.qipan[j + 5, i - j - 5]]
                self.match(listcase)

        for i in range(1, 15):     #计算下半部分，长度为5
            for j in range(18, i + 4 - 1, -1):
                listcase=[self.qipan[i - j + 18, j],self.qipan[i - j + 19, j - 1],self.qipan[i - j + 20, j - 2],
                          self.qipan[i - j + 21, j - 3],self.qipan[i - j + 22, j - 4]]
                self.match(listcase)
        for i in range(1, 15):     #计算下半部分，长度为6
            for j in range(18, i + 4, -1):
                listcase =[self.qipan[i - j + 18, j],self.qipan[i - j + 19, j - 1],self.qipan[i - j + 20, j - 2],
                           self.qipan[i - j + 21, j - 3],self.qipan[i - j + 22, j - 4],self.qipan[i - j + 23, j - 5]]

        '''计算右斜线方向'''
        for i in range(14, -1, -1):  #计算下半部分，长度为5
            for j in range(i + 1):
                listcase=[self.qipan[i, j], self.qipan[i + 1, j + 1], self.qipan[i + 2, j + 2],
                          self.qipan[i + 3, j + 3],self.qipan[i + 4, j + 4]]
                self.match(listcase)
        for j in range(1, 15):       #计算上半部分，长度为5
            for i in range(15 - j):
                listcase =[self.qipan[i, j + i],self.qipan[i + 1, j + i + 1],self.qipan[i + 2, j + i + 2],
                           self.qipan[i + 3, j + i + 3],self.qipan[i + 4, j + i + 4]]
                self.match(listcase)
        for j in range(1, 14):  #计算上半部分，长度为6
            for i in range(14 - j):
                listcase =[self.qipan[i, j + i],self.qipan[i + 1, j + i + 1],self.qipan[i + 2, j + i + 2],
                           self.qipan[i + 3, j + i + 3],self.qipan[i + 4, j + i + 4],self.qipan[i + 5, j + i + 5]]
                self.match(listcase)
        for i in range(13, -1, -1):  # 计算下半部分，长度为6
            for j in range(i + 1):
                listcase = [self.qipan[i, j],self.qipan[i + 1, j + 1],self.qipan[i + 2, j + 2],
                            self.qipan[i + 3, j + 3],self.qipan[i + 4, j + 4],self.qipan[i + 5, j + 5]]
                self.match(listcase)

        '''计算横方向'''
        for i in range(19):       #长度为5
            for j in range(15):
                listcase=[self.qipan[i, j],self.qipan[i, j + 1],self.qipan[i, j + 2],self.qipan[i, j + 3],
                          self.qipan[i, j + 4]]
                self.match(listcase)
        for i in range(19):  # 长度为6
            for j in range(14):
                listcase = [self.qipan[i, j],self.qipan[i, j + 1],self.qipan[i, j + 2],self.qipan[i, j + 3],
                            self.qipan[i, j + 4],self.qipan[i, j + 5]]
                self.match(listcase)

        '''计算纵方向'''
        for j in range(19):    #长度为5
            for i in range(15):
                listcase = [self.qipan[i, j],self.qipan[i + 1, j],self.qipan[i + 2, j],self.qipan[i + 3, j],
                            self.qipan[i + 4, j]]
                self.match(listcase)
        for j in range(19):  # 长度为5
            for i in range(14):
                listcase = [self.qipan[i, j],self.qipan[i + 1, j],self.qipan[i + 2, j],self.qipan[i + 3, j],
                            self.qipan[i + 4, j],
                            self.qipan[i + 5, j]]
                self.match(listcase)
        if flag=='White':     #判断是哪一方
            return self.num1-self.num2
        else:
            return self.num2-self.num1
    def match(self,listcase):
        if len(listcase)==5:
            if listcase in [[1, 1, 1, 1, 1]]:  # ***** 5
                self.done=True
                self.num1 += value1
            elif listcase in [[2, 2, 2, 2, 2]]:  # ***** 5
                self.done = True
                self.num2 += value1
            elif listcase in [[0, 1, 1, 1, 0],[0,1,1,0,1],[1,0,1,1,0],[1,1,1,0,0],[0,0,1,1,1]]:  # +***+ 5  +**+* *+**+
                self.num1 += value4
            elif listcase in [[0, 2, 2, 2, 0],[0,2,2,0,2],[2,0,2,2,0],[2,2,2,0,0],[0,0,2,2,2]]:
                self.num2 += value4
            elif listcase in [[0, 0, 1, 1, 0],[0, 1, 1, 0, 0],[1,1,0,0,0]]:  # +**++  **+++5
                self.num1 += value6
            elif listcase in [[0, 0, 2, 2, 0],[0, 2, 2, 0, 0],[2,2,0,0,0]]:  # +**++  **+++5
                self.num2 += value6
            elif listcase in [[1, 1, 0, 1, 1],[1, 0, 1, 1, 1],
                              [1, 1, 1, 0, 1],[1,1,1,1,0],[0,1,1,1,1]]:  # 5 **+**     or *+***  or ***+*
                self.num1 += value10
            elif listcase in [[2, 2, 0, 2,2],[2, 0, 2, 2,2],[2,2, 2, 0, 2],[2,2,2,2,0],[0,2,2,2,2]]:
                self.num2 += value10
            elif listcase in [[2, 2, 0, 0, 2],[2, 0, 0, 2, 2],[2,2,0,2,0]]:  # 5 *++**
                self.num2 += value8
            elif listcase in [[1, 0, 0, 1, 1],[1, 1, 0, 0, 1],[1,1,0,1,0]]:
                self.num1 += value8
            elif listcase in [[1, 0, 1, 0, 1]]:
                self.num1 += value9
            elif listcase in [[2, 0,2, 0, 2]]:
                self.num2 += value9
            elif listcase in [[2, 2, 2, 0, 0],[0,0,2,2,2],[2,2,0,0,2],[2,2,0,2,0]]:
                self.num2 += value5
            elif listcase in [[1, 1, 1, 0, 0],[0,0,1,1,1],[1,1,0,0,1],[1,1,0,1,0]]:
                self.num1 += value5
        elif len(listcase)==6:
            if listcase in[[0,1,1,1,1,0]]:     #+****+ 6
                self.num1+=value2
            elif listcase in [[0,2,2,2,2,0]]:     #+****+  6
                self.num2+= value2
            elif listcase in[[2,1, 1, 1,1,0],[0,1, 1,1, 1,2]]:  #-****+  6
                self.num1 += value3
            elif listcase in[[1,2,2,2,2,0],[0,2,2,2,2 ,1]]:   #-****+ 6
                self.num2 += value3
            elif listcase in[[1,2, 2,2, 0,0],[0,0,2,2,2 ,1]]: #-***++ 6
                self.num2+=value5
            elif listcase in [[2,1, 1,1, 0, 0],[0, 0,1,1,1,2]]:  # -***++ 6
                self.num1 += value5
