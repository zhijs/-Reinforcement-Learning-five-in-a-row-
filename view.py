from tkinter import *
import tkinter.messagebox
from environment import env
import numpy as np
import time
from DQN import DeepQNetwork
'''
tkinter界面类
'''
space=361
space_col=19

class view(tkinter.Tk):
    def __init__(self):
        self.gameStart=False
        self.status=False
        self.reward=0
        super(view, self).__init__()
        self.n_actions = 361    #定义动作的可能个数
        self.n_features = 361
        self.doneList=[]
        self.allphoto=[]
        self.initView()
        self.env=env()
        self.wobservation=None
        self.wobservation_=None
        self.action1=None
        self.RL = DeepQNetwork(self.n_actions, self.n_features )

    def callback(self,event):
        if self.gameStart:
            mouse_x = event.x
            mouse_y = event.y
            if 590 > mouse_x > 20 and 590 > mouse_y > 20:
                # 横向为a,纵向为b
                a = round((mouse_x - 40) / 30)
                b = round((mouse_y - 40) / 30)
                action = b * 19 + a
                # self.env.qipan[b, a] = 2，非计算机方
                observation =self.getdouble(np.reshape(np.copy(self.env.qipan), [1, space]))
                bobservation=self.transfore(observation)
                qipan,observation_, reward, done=self.step(action, 'Black')
                bobservation_=self.transfore(observation_)
                print('人工下棋的reward:%d'%reward)
                self.RL.store_transition(bobservation, action, reward*1.5, bobservation_) #此处默认人的掷棋是最优的
                if done:
                    tkinter.messagebox.showinfo(title='提示', message='you win!!!1')
                    self.RL.learn(flag=2)
                    self.RL.saveavarriable()
                    self.RL.plot_cost()
                    self.gameStart=False
                # self.status = True
                #计算机选择动作
                self.bqipan=np.copy(self.env.qipan)
                wobservation = self.getdouble(np.reshape(self.bqipan,[1,space]))
                action1= self.RL.choose_action(self.bqipan,wobservation)     #这里让电脑选择下一步下
                bqipan_,wobservation_,reward,done=self.step(action1,'White')
                print('计算机下棋的reward:%d'%reward)
                self.RL.store_transition(observation, action, reward, observation_)
                if done:
                    tkinter.messagebox.showinfo(title='提示', message='you failure')
                    self.RL.saveavarriable()
                    self.RL.plot_cost()
                    self.gameStart = False

    def initView(self):
        def buttonCallBack():
            self.RL.getvarriable()
            self.gameStart = True
            if len(self.allphoto) > 0:

                for i in self.allphoto:
                    self.w.delete(i)

            self.allphoto.clear()
            self.doneList.clear()
            observation = self.env.reset()

        self.master = Tk()
        self.master.title("五子棋")
        self.master.resizable(width=False, height=False)
        self.w = Canvas(self.master, bg="#FFFFF0", width=700, height=630)
        for c in range(40, 610, 30):  # 竖向
            x0, y0, x1, y1 = c, 40, c, 580
            self.w.create_line(x0, y0, x1, y1)
        for r in range(40, 610, 30):
            x0, y0, x1, y1 = 40, r, 580, r
            self.w.create_line(x0, y0, x1, y1)
        Label(self.w, text=1, bg="#FFFFF0").place(x=5, y=5)
        x1 = 60
        y1 = 5
        for i in range(2, 20):
            Label(self.w, text=i, bg="#FFFFF0").place(x=x1, y=y1)
            x1 += 30
        x1 = 5
        y1 = 60
        for i in range(2, 20):
            Label(self.w, text=i, bg="#FFFFF0").place(x=x1, y=y1)
            y1 += 30
        Button(self.w, text="开始游戏", bg="yellow", activebackground="Black", command=buttonCallBack).place(x=610, y=500)
        self.w.bind("<Double-Button-1>", self.callback)
        self.w.pack()
        #self.master.mainloop()


    def show(self,action,flag):
        y=(action//19)*30+40
        x=(action%19)*30+40
        if flag=='Black':
            a=self.w.create_oval(x-14,y-14,x+14,y+14,fill="Black")
        elif flag=='White':
            a = self.w.create_oval(x-14, y-14, x+14, y+14, fill="White")
        self.allphoto.append(a)
        self.update()

    def setPosition(self,action,flag):
        if action in self.doneList:
            tkinter.messagebox.showinfo(title='提示', message='当前位置不可下')

        else:
            self.doneList.append(action)
            self.show(action,flag)

    def reset(self):
        if len(self.allphoto)>0:

            for i in self.allphoto:
                self.w.delete(i)
        self.allphoto.clear()
        self.doneList.clear()
        self.gameStart=False
        observation=self.env.reset()
        ob=self.getdouble(np.reshape(observation,[1,space]))
        return np.copy(self.env.qipan),ob


    #############################################
    def step(self,action,flag):
        # 根据不同的掷棋方，返回reward
        # print(flag)
        # print('ation:%d'%action)
        p1 = self.env.pwb(flag)
        p2 = self.env.pwn(action, flag)  # 走完后赢的可能性

        # print('落子前所得分数%d'%p1)
        # print('落子后所得分数%d'%p2)
        s=p2-p1
        # if s<=0:
        #     self.reward=0
        # elif 0<s<150:
        #     self.reward=300
        # elif 150<=s<800:
        #     self.reward=500
        # elif 800<=s<3500:
        #     self.reward=2000
        # elif 3500<=s<4800:
        #     self.reward=4000
        # elif s>4800:
        #     self.reward=6000

        print("该步的回报值：%d"%s)

        self.setPosition(action,flag)
        if(s==-120):
            time.sleep(10000)
        qipan=self.getdouble(np.reshape(np.copy(self.env.qipan),[1,space]))
        return np.copy(self.env.qipan),qipan,s,self.env.done


    def tryPosition(self,Ob,ation,flag):
         qipan=np.copy(Ob)
         if flag=='White':
             qipan[0,ation]=1
         else:
             qipan[0,ation]=2
         return qipan


    def render(self):
        self.update()

    def transfore(self,observation):
        # print(np.shape(shape)[1])
        s1=observation[0,:space]
        s2=observation[0,space:]
        s=np.hstack((s1,s2))
        return s

    #将棋盘1*361转化为1*722形式
    def getdouble(self,qipan):
        w_qipan=np.zeros([1,space])
        b_qipan=np.zeros([1,space])
        w_array=np.where(qipan==1)[1]
        b_array=np.where(qipan==2)[1]
        w_qipan[0,w_array]=1
        b_qipan[0,b_array]=1
        s=np.hstack((w_qipan,b_qipan))  #转化为1*722矩阵，前361是白字的状态，后361是黑子的状态
        return s








