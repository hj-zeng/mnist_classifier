# -*- coding: utf-8 -*-

#神经网络测试

import tensorflow as tf
from neural_network import  le_net, alexnet, vgg_16,fc_net
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
INPUT_NODE = 784
IMAGE_SIZE = 28
NUM_CHANNELS = 1
OUTPUT_NODE = 10

def evalute(network, mnist, model_save_path,is_fc):
    
    with tf.Graph().as_default() as g:

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
        y = network(x, 10, None) 
        saver = tf.compat.v1.train.Saver()
        #测试正确率
        #计算正确率 比较输出结果和标签
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1)) 
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        with tf.compat.v1.Session() as sess:
             #函数直接找到目录中的最新模型
             ckpt = tf.train.get_checkpoint_state(
                 model_save_path)

             if ckpt and ckpt.model_checkpoint_path:
                 #加载数据
                 testx,testy = mnist.test.images, mnist.test.labels
                 if is_fc:
                     reshape_test_xs= testx
                 else:
                     reshape_test_xs = np.reshape(testx,( mnist.test.num_examples,
                      IMAGE_SIZE,
                      IMAGE_SIZE,
                      NUM_CHANNELS))
                 
                 #加载模型
                 saver.restore(sess, ckpt.model_checkpoint_path)
                 
                 test_accuracy = sess.run(accuracy,feed_dict={x:reshape_test_xs,y_:testy})
                 
                 return test_accuracy 
 
def evaluate_fc(mnist):
    MODEL_SAVE_PATH = r"model/fc_net"
    network = fc_net
    print(evalute(network, mnist, MODEL_SAVE_PATH,True))
    #0.9462
            
def evaluate_le_net(mnist):
    MODEL_SAVE_PATH = r"model/le_net"
    network = le_net
    print(evalute(network, mnist, MODEL_SAVE_PATH,None))
    #0.9759
    
    
def evaluate_alex_net(mnist):
    MODEL_SAVE_PATH = r"model/alexnet"
    network = alexnet
    print(evalute(network, mnist, MODEL_SAVE_PATH,None))
    #alexnet 0.9782
    
def evaluate_vgg(mnist):
    MODEL_SAVE_PATH = r"model/vgg"
    network = vgg_16
    print(evalute(network, mnist, MODEL_SAVE_PATH,None))
    

    
#主程序                                                     
def main(argv=None):
    mnist = input_data.read_data_sets("mnist",one_hot=True)
    evaluate_fc(mnist)
    # evaluate_le_net(mnist)
    # evaluate_alex_net(mnist)
    
    
if __name__=='__main__':
    tf.compat.v1.app.run()  
