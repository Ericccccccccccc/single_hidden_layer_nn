
import numpy as np
import keras
import h5py
import test

class mynn(object):

    def __init__(self):
        (self.x_train,self.y_train),(self.x_test,self.y_test) = keras.datasets.mnist.load_data()
        #输入部分

        self.x_train = self.x_train.flatten()
        self.x_train = self.x_train / 255
        self.x_train = self.x_train.reshape(60000, 784)  #输入层的输入
        self.aerfa = np.zeros(30)  #隐层的输入
        self.beita = np.zeros(10)   #输出层的输入

        #权值部分
        self.v = np.random.randn(784,30) #输入层到隐层
        self.w = np.random.randn(30,10)  #隐层到输出层

        #阀值部分
        self.gama = np.random.randn(30)  #隐层
        self.thita = np.random.randn(10)  #输出层

        #输出部分（即经过激活函数）
        self.b = np.zeros(30)   #隐藏层的输出
        self.y = np.zeros(10)    #最终输出

        #学习速率和正确率
        self.correct_rate = 0.0
        self.correct = 0.0
        self.learning_rate = 1

        #标签
        self.true_result = np.zeros([60000,10])
        for i in range(60000):
            self.true_result[i][self.y_train[i]] = 1

        #测试集的初始化
        self.x_test = self.x_test.flatten()
        self.x_test = self.x_test / 255
        self.x_test = self.x_test.reshape(10000, 784)

        #训练数据的存储与读取
        self.file = h5py.File("training_data.h5",'w')
        self.acc = []
        self.testacc = []
        self.loss = []
        self.losssum = 0

    def debug(self):
        print(self.y_train)
        print(self.true_result)


    #定义激活函数
    def stimulation(self,x):
        return 1/(1+np.exp(-x))

    def deviation(self,x,y):
        pass



    def proceeding(self,index):
        self.aerfa = self.x_train[index].dot(self.v)
        self.b = self.stimulation(self.aerfa-self.gama)
        self.beita = self.b.dot(self.w)
        self.y = self.stimulation(self.beita-self.thita)
        #当前正确率
        for max_index in range(10):
            if max(self.y) == self.y[max_index]:
                break;
        if max_index == self.y_train[index]:
            self.correct+=1
        if (index+1) % 1000 == 0:
            print("当前正确率为")
            print("correct:%d index: %d"%(self.correct,index+1))
            print(self.correct/10)
            self.acc.append(self.correct/10)
            self.loss.append(self.losssum)
            self.losssum = 0
            self.correct = 0

        #以下为BP算法
        self.losssum += np.sum(self.true_result[index] - self.y)
        g = self.y*(1-self.y)*(self.true_result[index] - self.y)
        w_tmp = np.zeros(30)
        for i in range(30):
            w_tmp[i] = self.w[i].dot(g.transpose())
        e = self.b*(1-self.b)*w_tmp
        #以下开始更新参数
        for i in range(30):
            self.w[i]    +=  self.learning_rate*g*self.b[i]
            self.gama[i] += -self.learning_rate * e[i]

        for i in range(784):
            self.v[i]    += self.learning_rate * e * self.x_train[index][i]

        self.thita += -self.learning_rate*g

    def testing(self,index):
        self.aerfa = self.x_test[index].dot(self.v)
        self.b = self.stimulation(self.aerfa-self.gama)
        self.beita = self.b.dot(self.w)
        self.y = self.stimulation(self.beita-self.thita)
        #当前正确率
        for max_index in range(10):
            if max(self.y) == self.y[max_index]:
                break;
        if max_index == self.y_test[index]:
            self.correct+=1
        if (index+1) % 1000 == 0:
            print("当前测试正确率为")
            print("correct:%d index: %d"%(self.correct,index+1))
            print(self.correct/10)
            self.correct = 0


    def save(self):
        self.file.create_dataset('v', (784, 30), 'd',data=self.v)
        self.file.create_dataset('w', (30, 10), 'd', data=self.w)
        self.file.create_dataset('thita',(1,10), 'd', data=self.thita)
        self.file.create_dataset('gama', (1,30), 'd', data=self.gama)
        facc = open('acc.txt','w',encoding = 'utf-8')
        floss = open('loss.txt','w',encoding = 'utf-8')
        for i in self.acc:
            facc.write(str(i)+'\n')
        for i in self.loss:
            floss.write(str(i)+'\n')
        facc.close()
        floss.close()

    def load_from_data(self):
        self.v = self.file['v'][:]
        self.w = self.file['w'][:]
        self.thita = self.file['thita'][:]
        self.gama = self.file['game'][:]

    def start(self):
            for j in range(10):
                print("第%d轮训练"%(j+1))
                for i in range(60000):
                    self.proceeding(i)
               # self.save()
                for i in range(10000):
                    self.testing(i)

            print("开始测试，每1000为间隔：")
            for i in range(10000):
                self.testing(i)











test = mynn()
test.start()






