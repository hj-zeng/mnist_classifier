# -*- coding: utf-8 -*-
#神经网络模型
import tensorflow as tf

def conv_relu(input_, input_deep, output_deep,ksize, stride,name):
    """卷积层函数"""
    #输入参数->输入，输入维度，输出维度，卷积核大小，移动步长，变量名
    #输出参数->卷积层输出
    with tf.compat.v1.variable_scope(name):
        conv_weights = tf.compat.v1.get_variable(
            'weight', [ksize, ksize, input_deep, output_deep],
            initializer = tf.truncated_normal_initializer(stddev=0.1))
        
        conv_biases = tf.compat.v1.get_variable(
            'biases', [output_deep], initializer = tf.constant_initializer(0.0))
        
        conv = tf.nn.conv2d(
            input_, conv_weights, strides=[1,stride,stride,1], padding='SAME')
        
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
        
    return relu

def conv(input_, input_deep, output_deep,ksize, stride,name):
    """卷积层函数"""
    #输入参数->输入，输入维度，输出维度，卷积核大小，移动步长，变量名
    #输出参数->卷积层输出
    with tf.compat.v1.variable_scope(name):
        conv_weights = tf.compat.v1.get_variable(
            'weight', [ksize, ksize, input_deep, output_deep],
            initializer = tf.truncated_normal_initializer(stddev=0.1))
        
        conv_biases = tf.compat.v1.get_variable(
            'biases', [output_deep], initializer = tf.constant_initializer(0.0))
        
        conv = tf.nn.conv2d(
            input_, conv_weights, strides=[1,stride,stride,1], padding='SAME')
        
        tanh = tf.nn.tanh(tf.nn.bias_add(conv, conv_biases))
        
    return tanh

def maxpool(input_, ksize, stride,name):
    """最大池化层函数"""
    #输入参数->输入，池化核大小，移动步长，变量名
    #输出参数->池化层输出
    with tf.name_scope(name):
        pool = tf.nn.max_pool2d(input_, ksize=[1,ksize,ksize,1], strides=[1,stride,stride,1],
                                padding='SAME')
    return pool

def fc(input_, input_deep, out_deep, name,regularizer):
    """全连接函数"""
    #输入参数->输入，输入维度，输出维度，变量名，正则化
    #输出参数->全连接层输出
    with tf.compat.v1.variable_scope(name):
        weight = tf.Variable(
            tf.truncated_normal([input_deep, out_deep], stddev=0.05),
            name="weights")
        
        bias = tf.Variable(
            tf.constant(0.1, dtype=tf.float32, shape=[out_deep]), 
            name="bias")

        if regularizer != None:
            tf.compat.v1.add_to_collection('losses', regularizer(weight))
            
        net = tf.add(tf.matmul(input_, weight), bias)     
        
    return net
  
def fc_net(input_,class_number,regularizer):
    
    INPUT_NODE=784
    LAYER1_NODE = 500
    weight1 = tf.Variable(tf.random.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1)) 
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    weight2 = tf.Variable(tf.random.truncated_normal([LAYER1_NODE,class_number],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[class_number]))
    
    if regularizer != None:
        tf.compat.v1.add_to_collection('losses', regularizer(weight1))
        tf.compat.v1.add_to_collection('losses', regularizer(weight2))
            
    layer1 = tf.nn.relu(tf.matmul(input_,weight1) + biases1)
 
    return tf.matmul(layer1,weight2) + biases2
    
def le_net(input_, class_number, regularizer):
    """Le_Net网络"""
    #输入函数->网络输入，输出维度，正则化
    #输出函数->网络输出
    conv1 = conv_relu(input_, 1, 6, 5, 1,'layer1_conv1')
    
    pool1 = maxpool(conv1, 2, 2,'layer2_pool1')
    
    conv2 = conv_relu(pool1, 6, 16, 5, 1,'layer3_conv2')
    
    pool2 = maxpool(conv2, 2, 2,'layer4_pool2')
    
    pool_shape = pool2.get_shape().as_list()

    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    
    reshaped=tf.layers.flatten(pool2)
    
    # reshaped = tf.reshape(pool1, [pool_shape[0], nodes])
    
    fc1 = fc(reshaped, nodes, 120, 'layer5_fc1', regularizer)
    
    fc2 = fc(fc1, 120,  84, 'layer6_fc2', regularizer)
    
    fc3 = fc(fc2, 84,  class_number, 'layer7_fc3', regularizer)
    
    return fc3
   
def alexnet(input_, class_number, regularizer):
    """Alexnet网络"""
    #输入函数->网络输入，输出维度，训练标志，正则化，dropout概率
    #输出函数->网络输出
    conv1 = conv(input_, 1, 48, 11, 4,'layer1_conv1')
    # print(conv1)
    pool1 = maxpool(conv1, 3, 2,'layer2_pol1')
    # print(pool1)
    conv2 = conv(pool1, 48, 128, 5, 1,'layer3_conv2')
    # print(conv2)
    pool2 = maxpool(conv2, 3, 2,'layer4_pool2')
    # print(pool2)
    conv3 = conv(pool2, 128, 192, 3, 1,'layer5_conv3')
    # print(conv3)
    conv4 = conv(conv3, 192, 192, 3, 1,'layer6_conv4')
    # print(conv4)
    conv5 = conv(conv4, 192, 128, 3, 1,'layer7_conv5')
    # print(conv5)
    pool3 = maxpool(conv5, 3, 2,'layer8_pool3')
    # print(pool3)
    pool_shape = pool3.get_shape().as_list()

    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    reshaped=tf.layers.flatten(pool3)
    
    fc1 = fc(reshaped, nodes, 2048, 'layer9_fc1', regularizer)

    # if is_train:fc1 = tf.nn.dropout(fc1, p)
    
    fc2 = fc(fc1, 2048, 2048, 'layer10_fc2', regularizer)

    # if is_train:fc2 = tf.nn.dropout(fc2, p)
    
    fc3 = fc(fc2, 2048, 1000, 'layer11_fc3' ,regularizer )

    # if is_train:fc3 = tf.nn.dropout(fc3, p)
    
    fc4 = fc(fc3, 1000, class_number, 'layer12' ,regularizer )
    
    return fc4
    

def vgg_16(input_, class_number, regularizer):
    """vgg16网络结构函数"""
    #输入函数->网络输入，输出维度，正则化
    #输出函数->网络输出
    conv1 = conv(input_, 1, 64, 3, 1,'layer1_conv1')
    # print(conv1)
    conv2 = conv(conv1, 64, 64, 3, 1,'layer2_conv2')
    # print(conv2)
    pool1 = maxpool(conv2, 3, 2,'layer3_pool1')
    # print(pool1)
    conv3 = conv(pool1, 64, 128, 3, 1,'layer4_conv3')
    # print(conv3)
    conv4 = conv(conv3, 128, 128, 3, 1,'layer5_conv4')
    # print(conv4)
    pool2 = maxpool(conv4, 3, 2,'layer6_pool2')
    # print(pool2)
    conv5 = conv(pool2, 128, 256, 3, 1,'layer7_conv5')
    # print(conv5)
    conv6 = conv(conv5, 256, 256, 3, 1,'layer8_conv6')

    conv7 = conv(conv6, 256, 256, 1, 1,'layer9_conv7')

    pool3 = maxpool(conv7, 3, 2,'layer10_pool3')
    
    conv8 = conv(pool3, 256, 512, 3, 1,'layer11_conv8')

    conv9 = conv(conv8, 512, 512, 3, 1,'layer12_conv9')
    # print(conv9)
    conv10 = conv(conv9, 512, 512, 1, 1,'layer13_conv10')
    # print(conv10)

    pool4 = maxpool(conv10, 3, 2,'layer14_pool4')
    
    conv11 = conv(pool4, 512, 512, 3, 1,'layer15_conv11')

    conv12 = conv(conv11, 512, 512, 3, 1,'layer16_conv12')

    conv13 = conv(conv12, 512, 512, 1, 1,'layer17_conv13')

    pool5 = maxpool(conv13, 3, 2,'layer18_pool5')
    

    pool_shape = pool5.get_shape().as_list()

    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    reshaped=tf.layers.flatten(pool5)
    
    fc1 = fc(reshaped, nodes, 4096, 'layer19_fc1', regularizer)
    
    fc2 = fc(fc1, 4096, 4096, 'layer20_fc2', regularizer)
    
    fc3 = fc(fc2, 4096, class_number, 'layer21_fc3' ,regularizer)
    
    return fc3

##测试程序
# import cv2
# import numpy as np
# path = r"C:\Users\user\Desktop\python_work\study_tensorflow\flower_test_image\123.jpg"
# tf.compat.v1.reset_default_graph() #先清空计算图
# image = cv2.imread(path, 0)
# image_data = tf.image.convert_image_dtype(image,dtype=tf.float32)
# with tf.compat.v1.Session() as sess:
#     image = sess.run(image_data)
#     image = cv2.resize(image,(28,28))
#     reshape_xs=np.reshape(image,(1,28,28,1))
#     y = alexnet(reshape_xs, 10, None,)
#     print(y) #Tensor("layer21_fc3/Add:0", shape=(1, 5), dtype=float32)
    
# with tf.compat.v1.Session() as sess:
#     image = sess.run(image_data)
#     image = cv2.resize(image,(1,784))
#     reshape_xs=np.reshape(image,(1,784))
#     y = fc_net(reshape_xs, 10, None,)
#     print(y) #Tensor("layer21_fc3/Add:0", shape=(1, 5), dtype=float32)
    