import tensorflow as tf
import numpy as np
import os
"""
神经网络结构类
"""
l_1num=722
l_2num=722
l_3num=540
l_4num=361
col=19

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.18, #学习率
            reward_decay=0.01,#延迟回报率
            e_greedy=0.9,
            replace_target_iter=50, #一定的时候替换神经网络的参数
            memory_size=500, #训练数据的存储量
            batch_size=100,  #每次学习用的数据条数
            e_greedy_increment=0.0001, #随机性降低的频率
            output_graph=False,
            savefile='varriable.ckpt'
    ):
        self.n_actions = n_actions
        self.n_features = n_features  #255
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon =1 #
        self.n_r=1

       #学习步数
        self.learn_step_counter = 0

        self.action_space = np.arange(361)

        #设置数据存储的维度[s,a,r,s_]
        self.memory = np.zeros((self.memory_size, l_1num*2+2))

       #初始化时创建神经网络
        self._build_net()

        self.sess = tf.Session()
        self.saveflie=savefile
        self.saver = tf.train.Saver()
        self.path=os.path.abspath('.')+'/meta/'+self.saveflie
        self.qipan_col=19

        if output_graph:
            #输出神经网络结构图
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []


    def _build_net(self):
        # ------------------ 创建估计网络 ------------------
        #2列的数据，s用于网络输入
        self.s = tf.placeholder(tf.float32, [None, l_1num], name='s')  # input  定义‘符号’变量，也称为占位符
        #用来接收Q_target
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):   #
            #  c_names(collections_names) 是在更新 target_net 参数时会用到
            #因为输入为1行361列的矩阵，那么w必须为361行，其列数对应神经元的个数，此处定义20个经元
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES],l_1num , \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # eval_net 的第一层. collections 是在更新 target_net 参数时会用到
            #棋盘19*19=361
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [l_1num, l_1num], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, l_1num], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            #eval_net 的第二层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [l_1num, l_2num], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, l_2num], initializer=b_initializer, collections=c_names)
                #l2= tf.matmul(l1, w2) + b2
                l2=tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [l_2num, l_3num], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, l_3num], initializer=b_initializer, collections=c_names)
                l3=tf.nn.relu(tf.matmul(l2, w3) + b3)

            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [l_3num, l_4num], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [1, l_4num], initializer=b_initializer, collections=c_names)
                l4=tf.nn.relu(tf.matmul(l3, w4) + b4)

            with tf.variable_scope('l5'):
                w5= tf.get_variable('w5', [l_4num, self.n_actions], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('b5', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval=tf.matmul(l4, w5) + b5

        # 求误差
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))  #计算误差
         # 梯度下降
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)  #反向传递修改误差

        # ------------------ 创建 target 神经网络, 提供 target Q ------------------
         #接收下个 observation
        self.s_ = tf.placeholder(tf.float32, [None, l_1num], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) 是在更新 target_net 参数时会用到
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

             # target_net 的第一层. collections 是在更新 target_net 参数时会用到
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [l_1num, l_1num], initializer=w_initializer, collections=c_names)
            b1 = tf.get_variable('b1', [1, l_1num], initializer=b_initializer, collections=c_names)
            #此处输入数据，进行正向传播
            l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

         # target_net 的第二层. collections 是在更新 target_net 参数时会用到
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [l_1num, l_2num], initializer=w_initializer, collections=c_names)
            b2 = tf.get_variable('b2', [1, l_2num], initializer=b_initializer, collections=c_names)
            l2 =tf.nn.relu(tf.matmul(l1, w2) + b2)
        with tf.variable_scope('l3'):
            w3 = tf.get_variable('w3', [l_2num, l_3num], initializer=w_initializer, collections=c_names)
            b3 = tf.get_variable('b3', [1, l_3num], initializer=b_initializer, collections=c_names)
            l3=tf.nn.relu(tf.matmul(l2, w3) + b3)

        with tf.variable_scope('l4'):
            w4 = tf.get_variable('w4', [l_3num, l_4num], initializer=w_initializer, collections=c_names)
            b4 = tf.get_variable('b4', [1, l_4num], initializer=b_initializer, collections=c_names)
            l4=tf.nn.relu(tf.matmul(l3, w4) + b4)

        with tf.variable_scope('l5'):
            w5= tf.get_variable('w5', [l_4num, self.n_actions], initializer=w_initializer, collections=c_names)
            b5 = tf.get_variable('b5', [1, self.n_actions], initializer=b_initializer, collections=c_names)
            self.q_next=tf.matmul(l4, w5) + b5


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        #构造相同的维度
        a=np.reshape(a,[1,self.n_r])
        r=np.reshape(r,[1,self.n_r])
        s=np.reshape(s,[1,l_1num])
        s_=np.reshape(s_,[1,l_1num])
        #记录一条 [s, a, r, s_] 记录
        transition = np.hstack((s, a, r, s_))

       # 总 memory 大小是固定的, 如果超出总大小, 旧 memory 就被新 memory 替换
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self,qipan, observation):  #根据当前的状态的到动作
        if np.random.uniform()<=self.epsilon:  #在该可能性下选择最优
             # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
            ob=np.reshape(observation,[1,l_1num])
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s:ob})[0]
            action_list=np.reshape(qipan,[1, self.n_features])
            #得到可以下子的位置索引
            a=set([])
            #对掷棋位置进行限制
            for i in range(self.n_actions):
                if action_list[0,i]==1 or action_list[0,i]==2:
                    #棋子以上的方向
                    if 0<=(i-col) and action_list[0,i-col]==0:
                        a.add(i-col)
                    #棋子往左方向
                    if 0<=(i-1)and action_list[0,i-1]==0:
                        a.add(i-1)
                    #棋子往右方向
                    if (i+1)<=(self.n_actions-1)and action_list[0,i+1]==0:
                        a.add(i+1)
                    #棋子往下
                    if (i+col)<=(self.n_actions-1) and action_list[0,i+col]==0:
                        a.add(i+col)
                    #棋子往上左斜方向
                    if 0<=(i-col-1) and action_list[0,i-col-1]==0:
                        a.add(i-col-1)
                    #棋子往上右斜方向
                    if 0<=(i-col+1) and action_list[0,i-col+1]==0:
                        a.add(i-col+1)
                    #棋子下左方向
                    if (i+col-1)<=(self.n_actions-1) and action_list[0,i+col-1]==0:
                        a.add(i+col-1)
                    #棋子往下右方向
                    if (i+col+1)<=(self.n_actions-1) and action_list[0,i+col+1]==0:
                        a.add(i+col+1)
            b1=list(a)
            if len(b1)==0:
                action=np.random.randint(0, self.n_actions)
                return action
            b=list([actions_value[j] for j in b1])
            #得到可下的点
            #得到最大的索引,对应的值
            action =b1[np.argmax(b)]
            print('选择最大的q值走')
        else:   #该可能性下随机
            action_list=np.reshape(qipan,[1, self.n_features])
            #得到可以下子的位置索引
            a=[i for i in range(np.shape(action_list)[1]) if action_list[0,i]==0]
            action =a[np.random.randint(0, len(a))]
            print('随机走')
        return action
    #替换旧的神经网络的参数
    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
    #神经网络学习函数
    def learn(self,flag=1):
        # 检查是否替换 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget_params_replaced\n')

          # 从 memory 中随机抽取 batch_size 这么多记忆
        if flag==2:
            self.batch_size=1
        if self.memory_counter > self.memory_size:
            #从0-memory_size随机选取 batch_size个数
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]


        ## 获取 q_next (target_net 产生了 q) 和 q_eval(eval_net 产生的 q)
        #分别计算q_next=tf.matmul(l1, w2) + b2 和q_eval = tf.matmul(l1, w2) + b2
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -l_1num:],  #现实网络， 传入下一个 state
                self.s: batch_memory[:, :l_1num],  #目标网络，传入当前state
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32) #生成一个 0-batch_size-1的array对象
        eval_act_index = batch_memory[:,l_1num].astype(int)  #选择特定的列
        reward = batch_memory[:, l_1num + self.n_r]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        """
        print("反向传播修改参数")
        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :l_1num],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

       # 逐步降低随机走的概率
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def saveavarriable(self):
        #使用saver提供的简便方法去调用 save
        print('参数保存')
        self.saver.save(self.sess, self.path) #file_name.ckpt如果不存在的话，会自动创建


    #读取文件变量
    def getvarriable(self):
        #恢复权重
        data=self.saver.restore(self.sess, self.path)


    #输出参数
    def showVarriable(self):
        print('eval-net参数')
        print(self.sess.run(tf.get_collection('eval_net_params')[0]))
        print('\ntarget-net参数')
        print(self.sess.run(tf.get_collection('target_net_params')[0]))

    #绘制误差曲线
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()





