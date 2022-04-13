import random
from random import shuffle
import os

cwd=os.getcwd()

def load5foldData(obj):
    if 'Twitter' in obj:
        labelPath = os.path.join(cwd,"data/" +obj+"/"+ obj + "_label_all.txt")
        labelset_nonR, labelset_f = ['news', 'non-rumor', '0'], ['false', '1']
        print("loading tree label" )
        NR,F = [],[]
        l1=l2=0
        labelDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            label, eid = line.split('\t')[1], line.split('\t')[0]
            labelDic[eid] = label.lower()
            if label in labelset_nonR:
                NR.append(eid)
                l1 += 1
            if labelDic[eid] in labelset_f:
                F.append(eid)
                l2 += 1
        
        print(len(labelDic))
        print(l1,l2)
        random.shuffle(NR)
        random.shuffle(F)
        

        fold0_x_test,fold1_x_test,fold2_x_test,fold3_x_test,fold4_x_test=[],[],[],[],[]
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)
        #leng3 = int(l3 * 0.2)
        #leng4 = int(l4 * 0.2)

        fold0_x_test.extend(NR[0:leng1])
        fold0_x_test.extend(F[0:leng2])
        #fold0_x_test.extend(T[0:leng3])
        #fold0_x_test.extend(U[0:leng4])
        fold0_x_train.extend(NR[leng1:])
        fold0_x_train.extend(F[leng2:])
        #fold0_x_train.extend(T[leng3:])
        #fold0_x_train.extend(U[leng4:])
        fold1_x_train.extend(NR[0:leng1])
        fold1_x_train.extend(NR[leng1 * 2:])
        fold1_x_train.extend(F[0:leng2])
        fold1_x_train.extend(F[leng2 * 2:])
        #fold1_x_train.extend(T[0:leng3])
        #fold1_x_train.extend(T[leng3 * 2:])
        #fold1_x_train.extend(U[0:leng4])
        #fold1_x_train.extend(U[leng4 * 2:])
        fold1_x_test.extend(NR[leng1:leng1*2])
        fold1_x_test.extend(F[leng2:leng2*2])
        #fold1_x_test.extend(T[leng3:leng3*2])
        #fold1_x_test.extend(U[leng4:leng4*2])
        fold2_x_train.extend(NR[0:leng1*2])
        fold2_x_train.extend(NR[leng1*3:])
        fold2_x_train.extend(F[0:leng2*2])
        fold2_x_train.extend(F[leng2*3:])
        #fold2_x_train.extend(T[0:leng3*2])
        #fold2_x_train.extend(T[leng3*3:])
        #fold2_x_train.extend(U[0:leng4*2])
        #fold2_x_train.extend(U[leng4*3:])
        fold2_x_test.extend(NR[leng1*2:leng1*3])
        fold2_x_test.extend(F[leng2*2:leng2*3])
        #fold2_x_test.extend(T[leng3*2:leng3*3])
        #fold2_x_test.extend(U[leng4*2:leng4*3])
        fold3_x_train.extend(NR[0:leng1*3])
        fold3_x_train.extend(NR[leng1*4:])
        fold3_x_train.extend(F[0:leng2*3])
        fold3_x_train.extend(F[leng2*4:])
        #fold3_x_train.extend(T[0:leng3*3])
        #fold3_x_train.extend(T[leng3*4:])
        #fold3_x_train.extend(U[0:leng4*3])
        #fold3_x_train.extend(U[leng4*4:])
        fold3_x_test.extend(NR[leng1*3:leng1*4])
        fold3_x_test.extend(F[leng2*3:leng2*4])
        #fold3_x_test.extend(T[leng3*3:leng3*4])
        #fold3_x_test.extend(U[leng4*3:leng4*4])
        fold4_x_train.extend(NR[0:leng1*4])
        fold4_x_train.extend(NR[leng1*5:])
        fold4_x_train.extend(F[0:leng2*4])
        fold4_x_train.extend(F[leng2*5:])
        #fold4_x_train.extend(T[0:leng3*4])
        #fold4_x_train.extend(T[leng3*5:])
        #fold4_x_train.extend(U[0:leng4*4])
        #fold4_x_train.extend(U[leng4*5:])
        fold4_x_test.extend(NR[leng1*4:leng1*5])
        fold4_x_test.extend(F[leng2*4:leng2*5])
        #fold4_x_test.extend(T[leng3*4:leng3*5])
        #fold4_x_test.extend(U[leng4*4:leng4*5])

    if obj == "Weibo":
        labelPath = os.path.join(cwd,"data/Weibo/weibo_covid19_label.txt")
        print("loading weibo label:")
        F, T = [], []
        l1 = l2 = 0
        labelDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            eid,label = line.split('\t')[0], line.split('\t')[1]
            labelDic[eid] = int(label)
            if labelDic[eid]==0:
                T.append(eid)
                l1 += 1
            if labelDic[eid]==1:
                F.append(eid)
                l2 += 1
        print(len(labelDic))
        print(l1, l2)
        random.shuffle(T)
        random.shuffle(F)

        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)
         
        #print(leng1)
        #print(leng2)
        #print(len(T))
        fold0_x_test.extend(F[0:leng2])
        fold0_x_test.extend(T[0:leng1])
        fold0_x_train.extend(F[leng2:])
        fold0_x_train.extend(T[leng1:])
        fold1_x_train.extend(F[0:leng2])
        fold1_x_train.extend(F[leng2 * 2:])
        fold1_x_train.extend(T[0:leng1])
        fold1_x_train.extend(T[leng1 * 2:])
        fold1_x_test.extend(F[leng2:leng2 * 2])
        fold1_x_test.extend(T[leng1:leng1 * 2])
        fold2_x_train.extend(F[0:leng2 * 2])
        fold2_x_train.extend(F[leng2 * 3:])
        fold2_x_train.extend(T[0:leng1 * 2])
        fold2_x_train.extend(T[leng1 * 3:])
        fold2_x_test.extend(F[leng2 * 2:leng2 * 3])
        fold2_x_test.extend(T[leng1 * 2:leng1 * 3])
        fold3_x_train.extend(F[0:leng2 * 3])
        fold3_x_train.extend(F[leng2 * 4:])
        fold3_x_train.extend(T[0:leng1 * 3])
        fold3_x_train.extend(T[leng1 * 4:])
        fold3_x_test.extend(F[leng2 * 3:leng2 * 4])
        fold3_x_test.extend(T[leng1 * 3:leng1 * 4])
        fold4_x_train.extend(F[0:leng2 * 4])
        fold4_x_train.extend(F[leng2 * 5:])
        fold4_x_train.extend(T[0:leng1 * 4])
        fold4_x_train.extend(T[leng1 * 5:])
        fold4_x_test.extend(F[leng2 * 4:leng2 * 5])
        fold4_x_test.extend(T[leng1 * 4:leng1 * 5])

        label_path_twitter = os.path.join(cwd,"data/Twitter/Twitter_label_all.txt")
        print("loading Twitter label")
        twitter_label=[]
        labelDic = {}
        train_num=0
        for line in open(label_path_twitter):
            line = line.rstrip()
            label, eid = line.split('\t')[1], line.split('\t')[0]
            labelDic[eid] = label.lower()
            twitter_label.append(eid)
            train_num += 1


    fold0_test = list(fold0_x_test)
    shuffle(fold0_test)
    fold0_train = list(fold0_x_train)
    shuffle(fold0_train)
    fold1_test = list(fold1_x_test)
    shuffle(fold1_test)
    fold1_train = list(fold1_x_train)
    shuffle(fold1_train)
    fold2_test = list(fold2_x_test)
    shuffle(fold2_test)
    fold2_train = list(fold2_x_train)
    shuffle(fold2_train)
    fold3_test = list(fold3_x_test)
    shuffle(fold3_test)
    fold3_train = list(fold3_x_train)
    shuffle(fold3_train)
    fold4_test = list(fold4_x_test)
    shuffle(fold4_test)
    fold4_train = list(fold4_x_train)
    shuffle(fold4_train)
    twitter_train = list(twitter_label)
    shuffle(twitter_train)
    #print(len(fold3_test))
    #print(len(fold4_test))

    return list(fold0_train),list(fold0_test),\
           list(fold1_train),list(fold1_test),\
           list(fold2_train),list(fold2_test),\
           list(fold3_train),list(fold3_test),\
           list(fold4_train), list(fold4_test),\
           list(twitter_train)
