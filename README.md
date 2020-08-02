# 介绍

本次采用mnist数据集，利用全连接神经网络以及卷积神经网络实现手写数字识别。并比较两种类型的网络在性能上的表现，包括P-R曲线图、F值、ROC曲线和AUC值。
# mnist数据集

mnist数据集是手写数字数据集，由Yann LeCun等人收集的人类手写数字。数据集包含60000张训练图片和10000张测试图片；每张图片仅仅含有一个数字，且每张图片的大小固定为28×28。

mnist数据集的组成部分:

| 文件名                  | 文件       |
| ----------------------- | ---------- |
| t10k-images-idx3-ubyte  | 测试集图片 |
| t10k-labels-idx1-ubyte  | 测试集标签 |
| train-images-idx3-ubyte | 训练集图片 |
| train-labels-idx1-ubyte | 训练集标签 |

# 网络结构

1. **全连接神经网络**

   全连接神经网络结构由输入层、隐藏层和输出层组成，其中隐藏层的神经元个数取500。

2. **卷积神经网络**

   * Le-Net网络，网络结构如下。

     |                | 卷积核大小 | 卷积核移动步长 | 卷积核个数（输出深度） |
     | -------------- | ---------- | -------------- | ---------------------- |
     | 第一层卷积层   | 5×5        | 1              | 6                      |
     | 第二层池化层   | 2×2        | 2              | 6                      |
     | 第三层卷积层   | 5×5        | 1              | 16                     |
     | 第四层卷积层   | 2×2        | 2              | 16                     |
     | 第五层全连接层 | ——         | ——             | 120                    |
     | 第六层全连接层 | ——         | ——             | 84                     |
     | 第七层输出层   | ——         | ——             | 10                     |

   * Alex-Net网络，网络结构如下。

     |                  | 卷积核大小 | 卷积核移动步长 | 卷积核个数（输出深度） |
     | ---------------- | ---------- | -------------- | ---------------------- |
     | 第一层卷积层     | 11×11      | 4              | 48                     |
     | 第二层池化层     | 3×3        | 2              | 48                     |
     | 第三层卷积层     | 5×5        | 1              | 128                    |
     | 第四层池化层     | 3×3        | 2              | 128                    |
     | 第五层卷积层     | 3×3        | 1              | 192                    |
     | 第六层卷积层     | 3×3        | 1              | 192                    |
     | 第七层卷积层     | 3×3        | 1              | 128                    |
     | 第八层池化层     | 3×3        | 2              | 128                    |
     | 第九层全连接层   | ——         | ——             | 2048                   |
     | 第十层全连接层   | ——         | ——             | 2048                   |
     | 第十一层全连接层 | ——         | ——             | 1000                   |
     | 第十二层输出层   | ——         | ——             | 10                     |

     

# 训练网络配置

* 损失函数：交叉熵
* 优化算法：GDO，梯度下降算法

* 学习率（指数衰减）：基础值0.01，衰减值0.98
* L2正则化权重：le-5
* 训练轮数：10000

# 模型测试

利用mnist数据集对训练好的神经网络进行正确率测试。结果如下。

| 网络       | 正确率 |
| ---------- | ------ |
| 全连接网络 | 94.62% |
| Le-Net     | 97.59% |
| Alex-Net   | 97.82% |

# 性能评测

二分类混淆矩阵

<table>
  <tr>
    <td rowspan="2">真实情况</td>
      <td colspan="2"><center>预测结果</center></td>
  </tr>
  <tr>
    <td>正例</td>
    <td>反例</td>
  </tr>
  <tr>
    <td>正例</td>
    <td>TP</td>
    <td>FN</td>
  </tr>
  <tr>
    <td>反例</td>
    <td>FP</td>
    <td>TN</td>
  </tr>
</table>

通过下面的方式建立多分类混淆矩阵

| 正例 | 反例  | TP   | FN   | FP   | TN   |
| ---- | ----- | ---- | ---- | ---- | ---- |
| 0    | 除0外 | TP0  | FN0  | FP0  | TN0  |
| 1    | 除1外 | TP1  | FN1  | FP1  | TN1  |
| 2    | 除2外 | TP2  | FN2  | FP2  | TN2  |
| ...  |       |      |      |      |      |
| 9    | 除9外 | TP9  | FN9  | FP9  | TN9  |

<table>
  <tr>
    <td rowspan="2">真实情况</td>
      <td colspan="2"><center>预测结果</center></td>
  </tr>
  <tr>
    <td>正例</td>
    <td>反例</td>
  </tr>
  <tr>
    <td>正例</td>
    <td>TP=(TP0+TP1+...+TP9)/10</td>
    <td>FN=(FN0+FN1+...+FN9)/10</td>
  </tr>
  <tr>
    <td>反例</td>
    <td>FP=(FP0+FP1+...+FP9)/10</td>
    <td>TN=(TN0+TN1+...+TN9)/10</td>
  </tr>
</table>

* **PR曲线**

  > 以召回率( TP/(TP+FN) )为横坐标，以精确率( TP/(TP+FP) )为纵坐标的曲线图。若分类器的PR曲线在分类器B的曲线上方，则可以分类器A性能优于B。

  ![PR曲线](https://github.com/hj-zeng/mnist_classifier/blob/master/image/pr-curve.png)

* **F值**

  > F = (2 * recall  * precision) / (recall + precision) 综合召回率和精确率两个指标，F值越大越好。

  | 网络     | F                  |
  | -------- | ------------------ |
  | 全连接   | 0.9121159172410195 |
  | Le-Net   | 0.9370994372322008 |
  | Alex-Net | 0.9472779479411854 |

* **ROC曲线**

  >以假正例率( FP/(TN+FP) )为横坐标，以真正例·率( TP/(TP+FN) )为纵坐标的曲线图。性能比较方式与PR曲线一致。

  ![ROC曲线](https://github.com/hj-zeng/mnist_classifier/blob/master/image/roc-curve.png)

* **AUC值**

  > AUC是ROC曲线下的面积，AUC值越大则性能越好。

  | 网络     | AUC                |
  | -------- | ------------------ |
  | 全连接   | 0.9214100911111112 |
  | Le-Net   | 0.9765490977777778 |
  | Alex-Net | 0.9835803150000001 |



# 文件说明

* **[mnist](https://github.com/hj-zeng/mnist_classifier/tree/master/mnist)**:  mnist数据集文件
* **[image](https://github.com/hj-zeng/mnist_classifier/tree/master/image)**: 神经网络训练过程的损失值和正确率的变化曲线图，以及PR、ROC曲线图
* **[txt](https://github.com/hj-zeng/mnist_classifier/tree/master/txt): 数据文件**。包括神经网络损失值和正确率数据，以及PR、ROC对应的数据。PR、ROC对应的数据采用字典的形式进行存储。PR曲线数据的键值为**’recalls'**和**‘precisions’**；ROC曲线的键值为**‘fprs’**和**’recalls'**
* **py文件**：
  * [evalute.py](https://github.com/hj-zeng/mnist_classifier/blob/master/evalute.py) : 测试神经网络程序
  * [neural_network.py](https://github.com/hj-zeng/mnist_classifier/blob/master/neural_network.py) : 搭建神经网络模型程序
  * [performance.py](https://github.com/hj-zeng/mnist_classifier/blob/master/performance.py) : 性能评测程序（PR、ROC、F、AUC)
  * [train](https://github.com/hj-zeng/mnist_classifier/blob/master/train.py) : 神经网络训练程序

