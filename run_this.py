from view import *
import numpy as np
'''
运行该文件进行计算机的自我对弈训练
'''
space=361
space_col=19
#定义一个函数，用于转换状态
def transfore(observation):
   # print(np.shape(shape)[1])
        s1=observation[0,:space]
        s2=observation[0,space:]
        s=np.hstack((s1,s2))
        return s
def run_maze(myview):

    try:
        myview.RL.getvarriable()
        step=0
        for episode in range(25000):
            print('第%d轮训练'%episode)
            qipan,observation = myview.reset() #初始换环境，并得到状态的观测值
            flag=True
            while True:
                if flag:  #whitte/1掷棋，训练
                    waction = myview.RL.choose_action(qipan,observation) #根据当前的状态，得到动作，observation相当于state
                    qipan,observation_, reward, done = myview.step(waction,'White')  #执行动作，得到环境的反馈回报和执行动作后的下一个状态，和是否结束
                    myview.RL.store_transition(observation, waction, reward, observation_)  #将上一个观测值，动作，回报，以及下一个观测值存起来，便于下次学习
                else: #black掷棋，转换状态训练
                    #print('balck掷棋,转换棋盘')
                    newobservation=transfore(np.copy(observation))
                    # #print(newobservation)
                    baction = myview.RL.choose_action(newobservation) #根据当前的状态，得到动作，observation相当于state
                    observation_, reward, done = myview.step(baction,'Black')  #执行动作，得到环境的反馈回报和执行动作后的下一个状态，和是否结束
                    newobservation_=transfore(np.copy(observation_))
                    myview.RL.store_transition(observation_, baction, reward, newobservation_)  #将上一个观测值，动作，回报，以及下一个观测值存起来，便于下次学习

                flag=not flag
                if (step >200) and (step % 100 == 0):
                    myview.RL.learn()  #进行学习
                    myview.RL.saveavarriable()
                observation = np.copy(observation_) #改变当前的观测值
                if done:
                    break
                step += 1
            print('共走了%d步'%step)
        myflag=True
    except:
        step = 1
        for episode in range(25000):
            print('第%d轮训练'%episode)
            qipan,observation = myview.reset() #初始换环境，并得到状态的观测值
            flag=True
            while True:
                if flag:  #whitte/1掷棋，训练
                    waction = myview.RL.choose_action(qipan,observation) #根据当前的状态，得到动作，observation相当于state
                    qipan,observation_, reward, done = myview.step(waction,'White')  #执行动作，得到环境的反馈回报和执行动作后的下一个状态，和是否结束
                    myview.RL.store_transition(observation, waction, reward, observation_)  #将上一个观测值，动作，回报，以及下一个观测值存起来，便于下次学习

                    #print(observation==observation_)
                else: #black掷棋，转换状态训练
                    #print('balck掷棋,转换棋盘')
                    newobservation=transfore(np.copy(observation))
                    baction = myview.RL.choose_action(qipan,newobservation) #根据当前的状态，得到动作，observation相当于state
                    qipan,observation_, reward, done = myview.step(baction,'Black')  #执行动作，得到环境的反馈回报和执行动作后的下一个状态，和是否结束
                    newobservation_=transfore(np.copy(observation_))
                    myview.RL.store_transition(observation_, baction, reward, newobservation_)  #将上一个观测值，动作，回报，以及下一个观测值存起来，便于下次学习

                flag=not flag
                if (step >200) and (step % 100 == 0):
                    myview.RL.learn()  #进行学习
                    myview.RL.saveavarriable()

                observation = np.copy(observation_) #改变当前的观测值
                if done:
                    break
                step += 1
                print('共走了%d步\n'%step)
        myflag=True
    return myflag


if __name__ == "__main__":
    # maze game

    myview =view()
    myflag=run_maze(myview)
    print(myflag)
    if myflag:
         myview.RL.saveavarriable()
         myview.RL.plot_cost()

    myview.mainloop()

