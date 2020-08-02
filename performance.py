# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from neural_network import  le_net, alexnet, vgg_16,fc_net
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import scipy.io

#中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
        soft_y = tf.nn.softmax(y)
        y_max = tf.argmax(soft_y,1)
        y_max_ = tf.argmax(y_,1)
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
                 
                 predicts, reals = sess.run([soft_y,y_max_],feed_dict={x:reshape_test_xs,y_:testy})

                 return predicts,reals
                 
 
def indictor(predicts, reals,class_num,p,key):
    #输入参数-> 神经网络softmax输出、真实值、分类数
    #根据阈值进行预测结果转换 
    tps,fps,fns,tns=[],[],[],[]
    for i in range(class_num):
        tp ,fp,fn,tn = 0,0,0,0
        for j in range(len(predicts)):
            max_real = reals[j]
            if predicts[j][i]>=p:
                pre_result = i
            else:
                pre_result = -1
            if pre_result==i :
                if max_real==i:
                    tp += 1
                elif max_real!=i:
                    fp += 1
            elif pre_result!=i :
                if max_real==i:
                    fn += 1
                elif max_real!=i:
                    tn += 1
                    
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)
        
    
    result_tp = np.mean(tps)
    result_fp = np.mean(fps)
    result_fn = np.mean(fns)
    result_tn = np.mean(tns)
    
    recall = result_tp/(result_tp + result_fn)
    precision = result_tp/(result_tp + result_fp)
    fpr = result_fp/(result_tn+result_fp)

    
    if key=='pr':
        return recall,precision
    elif key=='roc':
        return recall,fpr



def get_pr_data(mnist,model,model_path,save_path,is_fc):
    """获取pr值"""
    predicts,reals = evalute(model, mnist, model_path,is_fc) 
    recalls = []
    precisions = []
    thresholds = np.linspace(0,0.99,100)
    for threshold in thresholds:
        recall,precision = indictor(predicts, reals, 10, threshold, 'pr')
        recalls.append(recall)
        precisions.append(precision)
        
    data_dict = {'recalls':recalls,'precisions':precisions}
    with open(save_path,'w') as obj:
        obj.write(str(data_dict))
        
        
def get_roc_data(mnist,model,model_path,save_path,is_fc):
    """获取roc值"""
    predicts,reals = evalute(model, mnist, model_path,is_fc) 
    recalls = []
    fprs = []
    thresholds = np.linspace(0,0.99,100)
    for threshold in thresholds:
        recall,fpr = indictor(predicts, reals, 10, threshold, 'roc')
        recalls.append(recall)
        fprs.append(fpr)
        
    data_dict = {'recalls':recalls,'fprs':fprs}
    with open(save_path,'w') as obj:
        obj.write(str(data_dict))
        

def get_all_pr_data():
    """获取所有的pr值"""
    mnist = input_data.read_data_sets("mnist",one_hot=True)
    save_fc_path = r'txt/pr_curve/FC.txt'
    save_le_path = r'txt/pr_curve/Le_Net.txt'
    save_alex_path = r'txt/pr_curve/Alex_Net.txt'
    
    fc_model_path = r'model/fc_net'
    le_model_path = r'model/le_net'
    alex_model_path = r'model/alexnet'
    
    get_pr_data(mnist,fc_net, fc_model_path, save_fc_path,True)
    get_pr_data(mnist,le_net, le_model_path, save_le_path,None)
    get_pr_data(mnist,alexnet, alex_model_path, save_alex_path,None)

    
def get_all_roc_data():
    """获取所有的roc值"""
    mnist = input_data.read_data_sets("mnist",one_hot=True)
    save_fc_path = r'txt/roc_curve/FC.txt'
    save_le_path = r'txt/roc_curve/Le_Net.txt'
    save_alex_path = r'txt/roc_curve/Alex_Net.txt'
    
    fc_model_path = r'model/fc_net'
    le_model_path = r'model/le_net'
    alex_model_path = r'model/alexnet'
    
    get_roc_data(mnist,fc_net, fc_model_path, save_fc_path,True)
    get_roc_data(mnist,le_net, le_model_path, save_le_path,None)
    get_roc_data(mnist,alexnet, alex_model_path, save_alex_path,None)
    
def draw_train_process(recalls,precisions,title,xlabel,ylabel,save_path):
    """画图函数"""
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.plot(recalls[0], precisions[0],linewidth=3) 
    plt.plot(recalls[1], precisions[1],linewidth=3) 
    plt.plot(recalls[2], precisions[2],linewidth=3) 
    plt.legend(['FC','Le-Net','Alex-Net'])
    plt.savefig(save_path)
    plt.grid()
    plt.show()
    
def draw_pr():
    """画pr曲线"""
    txt1 = r'txt/pr_curve/FC.txt'
    txt2 = r'txt/pr_curve/Le_Net.txt'
    txt3 = r'txt/pr_curve/Alex_Net.txt'
    txts = [txt1,txt2,txt3]
    precisions,recalls = [], []
    for txt in txts:
        with open(txt) as obj:
            data = obj.read()
            data_dict = eval(data)
            precisions.append(data_dict['precisions'])
            recalls.append(data_dict['recalls'])
            
    # return precisions,recalls
    draw_train_process(recalls,precisions,'PR-CURVE','召回率','精确率',r'image/pr-curve')
     
def get_f():
    """获取所有f值"""
    fs = []
    txt1 = r'txt/pr_curve/FC.txt'
    txt2 = r'txt/pr_curve/Le_Net.txt'
    txt3 = r'txt/pr_curve/Alex_Net.txt'
    txts = [txt1,txt2,txt3]
    precisions,recalls = [], []
    for txt in txts:
        with open(txt) as obj:
            data = obj.read()
            data_dict = eval(data)
            precisions.append(data_dict['precisions'])
            recalls.append(data_dict['recalls'])
            
        recall = np.mean(recalls)
        precision = np.mean(precisions)
        f = (2*recall*precision)/(recall + precision)
        fs.append(f)
    return fs
    
def draw_roc():
    txt1 = r'txt/roc_curve/FC.txt'
    txt2 = r'txt/roc_curve/Le_Net.txt'
    txt3 = r'txt/roc_curve/Alex_Net.txt'
    txts = [txt1,txt2,txt3]
    fprs,recalls = [], []
    for txt in txts:
        with open(txt) as obj:
            data = obj.read()
            data_dict = eval(data)
            fprs.append(data_dict['fprs'])
            recalls.append(data_dict['recalls'])
            
    # return precisions,recalls
    draw_train_process(fprs,recalls,'ROC-CURVE','假正例率','真正例率',r'image/roc-curve')
    
def compute_auc():
    """计算auc"""
    all_auc = []
    txt1 = r'txt/roc_curve/FC.txt'
    txt2 = r'txt/roc_curve/Le_Net.txt'
    txt3 = r'txt/roc_curve/Alex_Net.txt'
    txts = [txt1,txt2,txt3]
    fprs,recalls = [], []
    for txt in txts:
        with open(txt) as obj:
            data = obj.read()
            data_dict = eval(data)
            fprs.append(data_dict['fprs'])
            recalls.append(data_dict['recalls'])
            
    for j in range(len(fprs)):
        fpr = fprs[j]
        recall = recalls[j]

        areas = []
        for i in range(len(fprs)):
            if i==len(fprs)-1:
                break
            w = fpr[i+1] -  fpr[i]
            h = 0.5*(recall[i+1] + recall[i])
            s = w*h
            areas.append(s)
        
        area = sum(areas)
        all_auc.append(abs(area))
    
    return all_auc

#主程序                                                     
def main(argv=None):
    #存储
    # get_all_pr_data()
    # get_all_roc_data()
    
    #计算F
    fs = get_f()
    print(fs)
    #读取-画图
    # draw_pr()
    # draw_roc()
    
    #计算auc
    # all_auc = compute_auc()
    # print(all_auc) #[0.9214100911111112, 0.9765490977777778, 0.9835803150000001]

if __name__=='__main__':
    tf.compat.v1.app.run()  



