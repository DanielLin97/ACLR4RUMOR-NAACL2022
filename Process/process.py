import os
from Process.dataset import GraphDataset,BiGraphDataset,UdGraphDataset
cwd=os.getcwd()


################################### load tree#####################################
def loadTree(dataname):
    if 'Twitter' in dataname:
        treePath = os.path.join(cwd,'data/'+dataname+'/Twitter_data_all.txt')
        print("reading twitter tree")
        treeDic = {}
        for line in open(treePath):
            line = line.strip('\n')
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            time_delay, text = float(line.split('\t')[3]), str(line.split('\t')[4])
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'time_delay': time_delay, 'text': text}
        print('tree no:', len(treeDic))

    if dataname == "Weibo":
        treePath = os.path.join(cwd,'data/Weibo/weibo_covid19_data.txt')
        print("reading Weibo tree")
        treeDic = {}
        for line in open(treePath):
            line = line.strip('\n')
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            time_delay, text = float(line.split('\t')[3]), str(line.split('\t')[4])
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'time_delay': time_delay, 'text': text}
        print('weibo tree no:', len(treeDic))

        tree_path_twitter = os.path.join(cwd,'data/Twitter/Twitter_data_all.txt')
        print("reading twitter tree")
        for line in open(tree_path_twitter):
            line = line.strip('\n')
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            time_delay, text = float(line.split('\t')[3]), str(line.split('\t')[4])
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'time_delay': time_delay, 'text': text}
        print('total tree no:', len(treeDic))

    return treeDic

################################# load data ###################################
def loadData(dataname, treeDic,fold_x_train,fold_x_test,droprate):
    data_path=os.path.join(cwd, 'data', dataname+'graph')
    print("loading train set", )
    traindata_list = GraphDataset(fold_x_train, treeDic, droprate=droprate,data_path= data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = GraphDataset(fold_x_test, treeDic,data_path= data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadUdData(dataname, treeDic,fold_x_train,fold_x_test,droprate):
    data_path=os.path.join(cwd, 'data',dataname+'graph')
    print("loading train set", )
    traindata_list = UdGraphDataset(fold_x_train, treeDic, droprate=droprate,data_path= data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = UdGraphDataset(fold_x_test, treeDic,data_path= data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadBiData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate,BUdroprate):
    data_path = os.path.join(cwd,'data', dataname + 'graph')
    print("loading train set", )
    traindata_list = BiGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)
    print("train no:", len(traindata_list))
    if len(fold_x_test) >0:
        print("loading test set", )
        testdata_list = BiGraphDataset(fold_x_test, treeDic, data_path=data_path)
        print("test no:", len(testdata_list))
        return traindata_list, testdata_list
    # twitter_path = os.path.join(cwd, 'data/Twittergraph')
    # print('loading twitter set')
    # twitterdata_list = BiGraphDataset(twitter_train)
    else:
        return traindata_list



