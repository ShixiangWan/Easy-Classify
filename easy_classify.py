#!/usr/bin/env python
# encoding:utf-8
import os
import sys
import getopt
import threading
import math
import numpy as np
from time import clock

from sklearn.externals.joblib import Memory
from sklearn import cross_validation, metrics

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import load_svmlight_file
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier

import easy_excel


mem = Memory("./mycache")
@mem.cache
def get_data(file_name):
    data = load_svmlight_file(file_name)
    return data[0], data[1]


def arff2svm(arff_files):
    svm_files = []
    for arff_file in arff_files:
        name = arff_file[0: arff_file.rindex('.')]
        tpe = arff_file[arff_file.rindex('.')+1:]
        svm_file = name+".libsvm"
        svm_files.append(svm_file)
        if tpe == "arff":
            if os.path.exists(svm_file):
                pass
            else:
                f = open(arff_file)
                w = open(svm_file, 'w')
                flag = False
                for line in f.readlines():
                    if flag:
                        if line.strip() == '':
                            continue
                        temp = line.strip('\n').split(',')
                        w.write(temp[len(temp)-1])
                        for i in range(len(temp)-1):
                            w.write(' '+str(i+1)+':'+str(temp[i]))
                        w.write('\n')
                    else:
                        line = line.upper()
                        if line.startswith('@DATA') or flag:
                            flag = True
                f.close()
                w.close()
        elif tpe == "libsvm":
            continue
        else:
            print "File format error! Arff and libsvm are passed."
            sys.exit()
    return svm_files


def get_classifier(dimen, isSearch):
    if dimen > 10:
        if dimen > 50:
            dimen = 50
        pca = map(int, np.linspace(10, dimen, dimen / 10))
    else:
        pca = [dimen]
    if isSearch:
        all_classifiers = [
            ('Nearest Neighbors', KNeighborsClassifier(),
             dict(pca__n_components=pca, param__n_neighbors=map(int, np.linspace(5, 20, 4)))),
            ('LogisticRegression', LogisticRegression(),
             dict(pca__n_components=pca, param__C=np.logspace(-4, 4, 3))),
            ('Bagging', BaggingClassifier(),
            dict(pca__n_components=pca, param__n_estimators=map(int, np.linspace(5, 50, 10)))),
            ('GradientBoosting', GradientBoostingClassifier(),
             dict(pca__n_components=pca, param__learning_rate=[0.001, 0.01, 0.1, 0.2, 0.5, 1])),
            ('SGD', SGDClassifier(),
             dict(pca__n_components=pca, param__loss=['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'])),
            ('LibSVM', SVC(kernel="linear", C=0.025),dict()),
            ('LinearSVC', LinearSVC(), dict(pca__n_components=pca, param__C=np.logspace(-4, 4, 3))),
            ('Decision Tree', DecisionTreeClassifier(),
             dict(pca__n_components=pca, param__max_depth=map(int, np.logspace(2, 6, 6, base=2)))),
            ('Random Forest', RandomForestClassifier(),
             dict(pca__n_components=pca, param__max_depth=map(int, np.logspace(2, 6, 6, base=2)))),
            ('ExtraTrees', ExtraTreesClassifier(),
             dict(pca__n_components=pca, param__max_depth=map(int, np.logspace(2, 6, 6, base=2)))),
            ('AdaBoost', AdaBoostClassifier(),
             dict(pca__n_components=pca, param__learning_rate=[0.001, 0.01, 0.1, 0.2, 0.5, 1])),
            ('Naive Bayes', BernoulliNB(),
             dict(pca__n_components=pca, param__alpha=[0.001, 0.01, 0.1, 0.2, 0.5, 1]))
        ]
    else:
        all_classifiers = [
            ('Nearest Neighbors', KNeighborsClassifier(), dict()),
            ('LogisticRegression', LogisticRegression(), dict()),
            ('Bagging', BaggingClassifier(), dict()),
            ('GradientBoosting', GradientBoostingClassifier(), dict()),
            ('SGD', SGDClassifier(), dict()),
            ('LibSVM', SVC(), dict()),
            ('LinearSVC', LinearSVC(), dict()),
            ('Decision Tree', DecisionTreeClassifier(), dict()),
            ('Random Forest', RandomForestClassifier(), dict()),
            ('ExtraTrees', ExtraTreesClassifier(), dict()),
            ('AdaBoost', AdaBoostClassifier(), dict()),
            ('Naive Bayes', BernoulliNB(), dict())
        ]
    return all_classifiers


class ClassifyThread (threading.Thread):
    def __init__(self, lab, clf, train_x, train_y, test_x=None, test_y=None, cv=None):
        threading.Thread.__init__(self)
        self.lab = lab
        self.clf = clf
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.cv = cv
    def run(self):
        loop_classifier(self.lab, self.clf, self.train_x, self.train_y, self.test_x, self.test_y, self.cv)


def loop_classifier(lab, clf, train_x, train_y, test_x=None, test_y=None, cv=None):
    global results
    try:
        clf.fit(train_x, train_y)
        print lab, "Thread: ", 'Best Param: ', clf.best_params_
        if cv is not None:
            forecast = cross_validation.cross_val_predict(clf, train_x, train_y, cv=cv)
            test_y = train_y
        else:
            forecast = clf.predict(test_x)
        mat = metrics.confusion_matrix(test_y, forecast)
        tp = float(mat[0][0])
        fp = float(mat[1][0])
        fn = float(mat[0][1])
        tn = float(mat[1][1])
        ac = '%0.4f' % metrics.accuracy_score(test_y, forecast)
        fc = '%0.4f' % metrics.f1_score(test_y, forecast)
        if cv is not None:
            roc_auc_score = '%0.4f' % cross_validation.cross_val_score(clf, train_x, train_y, scoring='roc_auc',cv=cv)\
                            .mean()
        else:
            roc_auc_score = '%0.4f' % metrics.roc_auc_score(test_y, forecast)
        pos = int(tp + fn)
        neg = int(fp + tn)
        if (tp + fp) == 0:
            precision = 1
        else:
            precision = tp / (tp + fp)
        if (tp + fn) == 0:
            recall = 1
        else:
            recall = se = tp / (tp + fn)
        if (tn + fp) == 0:
            sp = 1
        else:
            sp = tn / (tn + fp)
        if se == 1 or sp == 1:
            gm = 1
        else:
            gm = math.sqrt(se * sp)
        f_measure = f_score = fc
        if (tp + fp) * (tn + fn) * (tp + fn) * (tn + fp) == 0:
            mcc = 1
        else:
            mcc = (tp * tn - fn * fp) / (math.sqrt((tp + fp) * (tn + fn) * (tp + fn) * (tn + fp)))
        print lab, "Thread: ", 'Accuracy: ', ac
        # Label,Accuracy,Precision,Recall,SE,SP,GM,F_measure,F-Score,MCC,Matrix,TP,FN,FP,TN
        results.append([lab, ac, '%0.4f' % precision, '%0.4f' % recall, '%0.4f' % se, '%0.4f' % sp, '%0.4f' % gm,
                        f_measure, f_score, '%0.4f' % mcc, roc_auc_score, tp, fn, fp, tn, pos, neg]
        )
    except Exception:
        results.append([lab, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''])


# 接收命令行参数，-i接收输入libsvm格式文件，-c接收交叉验证折数，-t接收训练集分割率
cv = 5
split_rate = 0.33
input_files = []
excel_name = ""
isSearch = False
isMultipleThread = True
opts, args = getopt.getopt(sys.argv[1:], "hi:c:t:o:s:m:", )
for op, value in opts:
    if op == "-i":
        input_files = str(value)
        input_files = input_files.replace(" ", "").split(',')
        for input_file in input_files:
            if input_file == "":
                print "Warning: please insure no blank in your input files !"
                sys.exit()
    elif op == "-c":
        cv = int(value)
    elif op == "-t":
        split_rate = float(value)
    elif op == "-o":
        excel_name = str(value)
    elif op == "-s":
        if str(value) != "0":
            isSearch = True
    elif op == "-m":
        if str(value) == "0":
            isMultipleThread = False
    elif op == "-h":
        print 'Cross-Validate: python easy_classify.py -i {input_file.libsvm} -c {int: cross validate folds}'
        print 'Train-Test: python easy_classify.py -i {input_file.libsvm} -t {float: test size rate of file}'
        print 'More information: https://github.com/ShixiangWan/Easy-Classify'
        sys.exit()

print '*** Validating file format ...'
input_files = arff2svm(input_files)

experiment = ''
results = []
dimensions = []
big_results = []
sec = clock()
for input_file in input_files:
    # 导入原始数据
    X, y = get_data(input_file)
    X = X.todense()
    results = []
    print '*** Time cost on loading ', input_file, ': ', clock() - sec

    # 对数据切分或交叉验证，得出结果
    dimension = int(X.shape[1])
    print "Dimension:", dimension
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=split_rate, random_state=0)
    classifiers = get_classifier(dimension, isSearch)
    threads = []
    for name, classifier, grid in classifiers:
        classifier2 = Pipeline([('pca', PCA()), ('param', classifier)])
        grid_search = GridSearchCV(classifier2, param_grid=grid)
        if cv == 0:
            experiment = '训练测试结果'
            print u'>>>', name, 'is training...searching best parms...'
            if isMultipleThread:
                new_thread = ClassifyThread(name, grid_search, X_train, y_train, test_x=X_test, test_y=y_test)
                new_thread.start()
                threads.append(new_thread)
            else:
                loop_classifier(name, grid_search, X_train, y_train, test_x=X_test, test_y=y_test)
        else:
            experiment = '交叉验证结果'
            print u'>>>', name, 'is cross validating...searching best parms...'
            if isMultipleThread:
                new_thread = ClassifyThread(name, grid_search, X, y, cv=cv)
                new_thread.start()
                threads.append(new_thread)
            else:
                loop_classifier(name, grid_search, X, y, cv=cv)
        print 'Time cost: ', clock() - sec
    # 等待所有线程完成
    for t in threads:
        t.join()
    dimensions.append(str(dimension))
    big_results.append(results)
print 'Time cost: ', clock() - sec

# 保存结果至Excel
print '====================='
if easy_excel.save(experiment, dimensions, big_results, excel_name):
    print 'Save excel result file successfully.'
else:
    print 'Failed. Please close excel result file first.'
