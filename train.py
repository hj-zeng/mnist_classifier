# -*- coding: utf-8 -*-

#训练vgg网络
import os 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from neural_network import  le_net, alexnet, vgg_16, fc_net
from datetime import datetime
import matplotlib.pyplot as plt


#中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#参数设置
BATCH_SIZE = 20 #batch大小 
LEARNINNG_RATE_BASE = 0.01 #基础学习率
LEARNING_BATE_DECAY = 0.98 #学习率衰减率
REGULARATION_RATE = 0.00001#正则化权重
TRANING_STEPS = 10001 #训练次数
IMAGE_SIZE = 28 #输入图片大小
NUM_CHANNELS = 1 #输入图片维度 
INPUT_NODE = 784
OUTPUT_NODE = 10 #神经网络输出维度=标签维度


#画图函数
def draw_train_process(steps,para1, para2,name,net_name):
    """训练过程中的损失值/正确率变化"""
    title= net_name + "-训练过程中参数变化"
    plt.title(title, fontsize=20)
    plt.xlabel("训练次数", fontsize=14)
    # plt.ylabel("损失值", fontsize=14)
    plt.plot(steps, para1,color='red',label='损失值') 
    plt.plot(steps, para2,color='blue',label='正确率') 
    plt.legend(['损失值', '正确率'],loc='upper right')
    plt.grid()
    plt.savefig(name)
    plt.show()     
    

def train(mnist, network, model_path, model_name, png_name, 
          txt_name1, txt_name2, net_name, is_fc):
    """训练模型"""
    #输入参数:数据集，神经网络、模型保存地址、模型名称、图片保存名称、txt文件名、是否是全连接网络
    
    with tf.name_scope('input'):
        if is_fc:
            x = tf.compat.v1.placeholder(tf.float32,[
                None,
                INPUT_NODE], 
                name='x-input') 
        else:
            x = tf.compat.v1.placeholder(tf.float32,[
                None,
                IMAGE_SIZE, 
                IMAGE_SIZE,
                NUM_CHANNELS], 
                name='x-input') 

        
        y_ = tf.compat.v1.placeholder(tf.float32,
                                      [None, OUTPUT_NODE],
                                      name='y-input')
        
    regularizer = tf.contrib.layers.l2_regularizer(REGULARATION_RATE)

    y = network(x, 10, regularizer)  
    
    global_step = tf.Variable(0, trainable=False) 
       
    #生成损失函数
    #利用函数生成交叉熵 
    with tf.name_scope('loss_function'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_,1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.compat.v1.get_collection('losses')) 
        
    
    #指数衰减法设置学习率
    learning_rate = tf.compat.v1.train.exponential_decay(
        LEARNINNG_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_BATE_DECAY
        )

    
    #优化损失函数
    with tf.name_scope('train_step'):
        train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)\
            .minimize(loss, global_step=global_step)
            
        #反向传播更新参数 
        with tf.control_dependencies([train_step]):
            train_op = tf.no_op(name='train')
       
    #计算正确率 比较输出结果和标签
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1)) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    #初保存器初始化
    saver = tf.compat.v1.train.Saver()
    steps = []
    accracys = []
    loss_values = []
    
    with tf.compat.v1.Session() as sess:
        
        tf.compat.v1.global_variables_initializer().run() #参数初始化
            
        for i in range(TRANING_STEPS): #开始训练

            #训练
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            if is_fc:
                reshape_xs= xs
            else:
                reshape_xs=np.reshape(xs,(BATCH_SIZE,
                              IMAGE_SIZE,
                              IMAGE_SIZE,
                              NUM_CHANNELS))
                                
            _ , losses_value, step = sess.run(
                [train_op, loss, global_step],feed_dict ={x: reshape_xs, y_: ys})
            
            #验证
            valxs,valys = mnist.validation.images, mnist.validation.labels
            
            if is_fc:
                reshape_val_xs= valxs
            else:

                reshape_val_xs=np.reshape(valxs,(mnist.validation.num_examples,
                  IMAGE_SIZE,
                  IMAGE_SIZE,
                  NUM_CHANNELS))
                
            validate_acc = sess.run(accuracy,feed_dict ={x: reshape_val_xs, y_: valys})

            #参数保存
            steps.append(i)
            accracys.append(validate_acc)
            loss_values.append(losses_value)
            
            #打印训练过程的参数变化
            if i % 1000 ==0:
                print("训练 %d 轮后的损失值为 %g" %(step, losses_value))
                #验证
                print("训练 %d 轮后的正确率为 %g" %(i,validate_acc))
                #保存模型
                saver.save(sess, os.path.join(model_path, model_name), 
                            global_step=global_step)
            
    #把数据写入文件
    
    for i in range(len(accracys)):
        with open(txt_name1,'a') as obj:
            obj.write(str(accracys[i])+'\n')
        with open(txt_name2,'a') as obj:
            obj.write(str(loss_values[i])+'\n')
    
    draw_train_process(steps,loss_values, accracys, png_name, net_name)        

def train_fc(mnist):
    
    tf.compat.v1.reset_default_graph() #清空计算图
    strat_time = datetime.now()
    #参数设置 
    model_path = "model/fc_net" #模型保存路径
    model_name = 'model.ckpt'  #模型名字
    png_name = r'image/fc_net-损失值.jpg'
    txt_name1 = r'txt/fc_net_accracys.txt'
    txt_name2 = r'txt/fc_net_losses.txt'
    net_name = 'fc_net'
    network = fc_net
    #训练
    train(mnist, network, model_path, model_name, png_name, 
          txt_name1, txt_name2,net_name,True)
    end_time = datetime.now() 
    use_time = end_time - strat_time
    print('训练所用时间' + str(use_time))
    
def train_letnet(mnist):
    
    tf.compat.v1.reset_default_graph() #清空计算图
    strat_time = datetime.now()
    #参数设置 
    model_path = "model/le_net" #模型保存路径
    model_name = 'model.ckpt'  #模型名字
    png_name = r'image/Le_Net-损失值.jpg'
    txt_name1 = r'txt/le_net_accracys.txt'
    txt_name2 = r'txt/le_net_losses.txt'
    net_name = 'Le_Net'
    network = le_net
    #训练
    train(mnist, network, model_path, model_name, png_name,
          txt_name1, txt_name2,net_name, None)
    end_time = datetime.now() 
    use_time = end_time - strat_time
    print('训练所用时间' + str(use_time))
    
def train_alexnet(mnist):
    
    tf.compat.v1.reset_default_graph() #清空计算图
    strat_time = datetime.now()
    #参数设置 
    model_path = "model/alexnet" #模型保存路径
    model_name = 'model.ckpt'  #模型名字
    png_name = r'image/AlexNet-损失值.jpg'
    txt_name1 = r'txt/Alexnet_accracys.txt'
    txt_name2 = r'txt/Alexnet_losses.txt'
    net_name = 'AlexNet'
    network = alexnet
    #训练
    train(mnist, network, model_path, model_name, png_name,
          txt_name1, txt_name2,net_name,None)
    end_time = datetime.now() 
    use_time = end_time - strat_time
    print('训练所用时间' + str(use_time))
    
    
def train_vgg(mnist):
    
    tf.compat.v1.reset_default_graph() #清空计算图
    strat_time = datetime.now()
    #参数设置 
    model_path = "model/vgg_16" #模型保存路径
    model_name = 'model.ckpt'  #模型名字
    png_name = r'image/vgg_16-损失值.jpg'
    txt_name1 = r'txt/vgg_16_accracys.txt'
    txt_name2 = r'txt/vgg_16_losses.txt'
    net_name = 'vgg_16'
    network = vgg_16
    #训练
    train(mnist, network, model_path, model_name, png_name,
          txt_name1, txt_name2,net_name,None)
    end_time = datetime.now() 
    use_time = end_time - strat_time
    print('训练所用时间' + str(use_time))
    
    
#主程序                                                     
def main(argv=None):
    mnist = input_data.read_data_sets("mnist",one_hot=True)
    train_fc(mnist)
    # train_letnet(mnist)
    # train_alexnet(mnist)
    # train_vgg(mnist)
    #数据维度
    # print(mnist.train.label.shape) #(55000, 10)
    # print(mnist.test.labels.shape) #(10000, 10)
    # print(mnist.validation.labels.shape) #(5000, 10)
    # print(mnist.train.next_batch(10))
    
if __name__=='__main__':
    tf.compat.v1.app.run()  

