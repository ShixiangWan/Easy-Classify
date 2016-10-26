#!/usr/bin/env python
# encoding:utf-8
import sys
import getopt

def split_unbalance(file_name, pos_lab, neg_lab):
    data = open(file_name)
    pos_list = []
    neg_list = []
    for line in data.readlines():
        if line == "\n":
            continue
        type = line[:line.index(" ")]
        if type == pos_lab:
            pos_list.append(line)
        elif type == neg_lab:
            neg_list.append(line)
    data.close()

    pos_num = len(pos_list)
    neg_num = len(neg_list)
    if pos_num >= neg_num*2:
        temp_list = pos_list
        pos_list = neg_list
        neg_list = temp_list
        pos_num = len(pos_list)
        neg_num = len(neg_list)
    elif neg_num >= pos_num*2:
        pass
    else:
        print "Warning: Data set is not unbalance data set."
        return file_name

    short = file_name[:file_name.rindex(".")]
    for i in range(0, neg_num/pos_num):
        data = open(short+str(i+1)+".libsvm", "w")
        for j in range(i*pos_num, (i+1)*pos_num):
            data.write(pos_list[j - i*pos_num])
            data.write(neg_list[j])
        data.close()
        print ">> writing "+short+str(i+1)+".libsvm ..."
    print "Finished."

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "hi:p:n:", )
    for op, value in opts:
        if op == "-i":
            input_file = str(value)
        elif op == "-p":
            pos_lab = str(value)
        elif op == "-t":
            neg_lab = str(value)

    split_unbalance(input_file, pos_lab, neg_lab)

