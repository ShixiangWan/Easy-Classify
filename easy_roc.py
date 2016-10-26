#!/usr/bin/env python
# encoding:utf-8
import sys
import xlwt
import getopt
import numpy as np
from scipy import interp

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.externals.joblib import Memory
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
        BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression

###############################################################################

mem = Memory("./mycache")
@mem.cache
def get_data(name):
    data = load_svmlight_file(name)
    return data[0], data[1]


def get_classifier():
    all_classifiers = [
        ('Nearest Neighbors', KNeighborsClassifier()),
        ('LogisticRegression', LogisticRegression()),
        ('Bagging', BaggingClassifier()),
        ('GradientBoosting', GradientBoostingClassifier()),
        ('SGD', SGDClassifier(loss='modified_huber')),
        ('LibSVM', SVC(kernel="linear", probability=True, C=0.025)),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier()),
        ('ExtraTrees', ExtraTreesClassifier()),
        ('AdaBoost', AdaBoostClassifier()),
        ('Naive Bayes', BernoulliNB())
    ]
    return all_classifiers
###############################################################################

# 接收命令行参数，-i接收输入libsvm格式文件，-c接收交叉验证折数，-t接收训练集分割率
cross = 5
input_files = []
opts, args = getopt.getopt(sys.argv[1:], "hi:c:", )
for op, value in opts:
    if op == "-i":
        input_files = str(value)
        input_files = input_files.split(',')
    elif op == "-c":
        cross = int(value)
    elif op == "-h":
        print 'command: python easy_roc.py -i {input_file.libsvm} -c {int: cross validate folds}'
        sys.exit()

# input_files = ['ttt.libsvm', 'ttt2.libsvm']

for input_file in input_files:

    X, y = get_data(input_file)
    X = X.todense()

    wb = xlwt.Workbook(encoding='utf-8')
    for name, classifier in get_classifier():
        print u'>>>', name, '...',
        cv = StratifiedKFold(y, n_folds=cross)
        # classifier = svm.SVC(kernel='linear', probability=True, random_state=0)

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        all_fpr = []
        all_roc_auc = []
        for i, (train, test) in enumerate(cv):
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            all_tpr.append(tpr.tolist())
            all_fpr.append(fpr.tolist())
            all_roc_auc.append(roc_auc)

        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        mean_tpr = mean_tpr.tolist()
        mean_fpr = mean_fpr.tolist()

        ws = wb.add_sheet(name)
        ws.write(0, 0, 'Mean FPR')
        ws.write(1, 0, 'Mean TPR')
        for i in range(0, len(mean_fpr)):
            ws.write(0, i+1, mean_fpr[i])
            ws.write(1, i+1, mean_tpr[i])
        ws.write(2, 0, 'Mean ROC Area: %0.4f' % mean_auc)

        count = 3
        for num in range(0, len(all_tpr)):
            fold = num + 1
            ws.write(count, 0, 'FPR fold '+str(fold))
            ws.write(count+1, 0, 'TPR fold '+str(fold))
            # break
            for i in range(0, len(all_tpr[num])):
                ws.write(count, i + 1, all_fpr[num][i])
                ws.write(count+1, i + 1, all_tpr[num][i])
            ws.write(count + 2, 0, 'ROC Area: %0.4f' % all_roc_auc[num])
            count += 3
        print 'OK!'

    # 保存结果至Excel
    print '====================='
    try:
        wb.save('ROC+'+input_file+'.xls')
        print 'Save "ROC+'+input_file+'.xls" successfully.'
    except:
        print 'Fail to save "ROC.xls". Please close "ROC.xls" first.'
