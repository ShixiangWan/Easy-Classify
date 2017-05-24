# Easy-Classify
version 0.60

## Easy-Classify是什么?

Easy-Classify是一个基于python的sklearn包，自动生成二分类Excel实验报告和ROC值的小脚本，是二分类集成分类器的良好解决方案。分类器目前集成：

* Nearest Neighbors
* Bagging
* GradientBoosting
* SGD
* LibSVM
* Linear SVM
* SMO
* LinearSVC
* Decision Tree
* Random Forest
* AdaBoost
* Naive Bayes
* Neural Network
* ......

## 运行环境

* python 2.7及其基础科学计算包numpy、scipy、pandas；
* python的scikit-learn包用于跑分类器：
```python
  pip install scikit-learn
```
* python的scikit-neuralnetwork包用于跑神经网络：
```python
  pip install scikit-neuralnetwork
```
* python的xlwt用于写入excel结果报告：
```python
  pip intall xlwt
```

## 输入输出

* 输入：包含全部正反例的libsvm或arff格式文件，支持多文件混合输入。文件正反例标签为{0,1}，arff格式为weka软件默认格式，libsvm格式如：
```ssh
  1 1:7.964601769911504 2:0.8849557522123894 3:1.1799410029498525
  0 1:9.583333333333334 2:0.8333333333333334 3:4.1666666666666660
  1 1:6.427423674343867 2:0.8569898232458489 3:5.9989287627209430
  0 1:12.50000000000000 2:2.2727272727272730 3:5.1136363636363640
```
* 输出：
  * easy_classify.py: 输出Excel实验表格，如results.xls文件所示
  * easy_roc.py: 输出pdf格式的roc曲线图数据表，如ROC.xls文件所示
 
## 使用命令

#### 1. easy_classify.py专用于生成excel实验报告：

* 必选参数：
 * `-i`：输入的arff或libsvm格式文件，支持混输。如：`-i train.libsvm,train2.arff`，注意文件之间用英文`,`连接；
 * `-c`或`-t`：`-c`为交叉验证模式，值为交叉验证折数，如`-c 5`，默认为5；`-t`为训练测试模式，值为训练集测试集分割比例，如`-t 0.33`，默认为0.33.
 
 ```ssh
  # 交叉验证如：python easy_classify.py -i train.libsvm -c 10
  # 训练测如：python easy_classify.py -i train.libsvm -t 0.25
 ```

* 可选参数：
 * `-o`：指定输出excel文件名。默认为results.xls。
 * `-s`：是否寻找最佳分类器参数。`0`为不寻找，`1`为寻找。默认为`0`。
 * `-m`：是否并行运算，1GB以上大数据集不推荐使用。`0`为单线程运算，`1`为多线程并行运算，线程数是同时运行的分类器数，适合CPU和内存资源强大的用户。默认为`1`。

* 帮助：
```ssh
  python easy_classify.py -h
```
####2. easy_roc.py专用于生成绘制ROC曲线图需要的数据（只支持交叉验证）：
* 交叉验证：
```ssh
  python easy_roc.py -i {input_file.libsvm} -c {int: cross validate folds}
  # 单文件命令如：python easy_roc.py -i train.libsvm -c 5
  # 多文件命令如：python easy_roc.py -i train.libsvm,train2.libsvm -c 5
```

* 帮助：
```ssh
  python easy_roc.py -h
```

##升级日志
 * 2016-08-08，version 0.20:
   * 完成基本功能框架，集成主要分类器
   * 自动生成测试报告
   * 支持并行
 * 2016-08-12，version 0.40：
   * 增加神经网络等更多分类器
   * 增加分类器参数自动调优
   * 支持多种文件同时输入
 * 2016-09-15，version 0.50：
   * 增加ROC曲线图输出数据
 * 2016-10-26，version 0.60：
   * 更好地支持并行
   * 增加参数可选
   * 修复一些bug
 
 
