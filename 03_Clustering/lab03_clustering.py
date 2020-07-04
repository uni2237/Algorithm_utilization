#1. Scikit-learn에서제공하는DBSCAN, Agglomerative Clustering 활용하여 그래프그리기
# 2. K-Means 클러스터링구현및활용하여그래프그리기
import codecs
import copy
import random
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import MinMaxScaler

def euclid(data,centers):
    distance=0
    for i in range(len(data)):
        distance +=(data[i]-centers[i])**2
    return distance**0.5

def draw_graph(data,labels):
    plt.figure()
    plt.scatter(data[:,0],data[:,1],c=labels,cmap='rainbow')
    plt.show()

def norm(data):
    scaler=MinMaxScaler()
    norm_data=scaler.fit_transform(data)
    return norm_data

def read_data():
    data=[]
    with codecs.open("covid-19.txt",'r','utf-8') as file:
        lines=file.readlines()
        del lines[0]
        for index,line in enumerate(lines):
            line=line.rstrip().split('\t')
            data.append(line[5:7])
        return data


def dbscan(data):
    clustering=DBSCAN(eps=0.1,min_samples=2).fit(data)
    draw_graph(data,clustering.labels_)

def hierarchical(data):
    clustering=AgglomerativeClustering(n_clusters=8,affinity='Euclidean',linkage='complete').fit(data)
    draw_graph(data, clustering.labels_)


class KMeans:
    def __init__(self,data,n):
        self.data=data
        self.n=n
        self.cluster=OrderedDict() #순서 지정 딕셔너리

    def init_center(self):
        index=random.randint(0,self.n)
        index_list=[]
        for i in range(self.n):
            while index in index_list:
                index=random.randint(0,self.n)
            index_list.append(index)
            self.cluster[i]={'center': self.data[index],'data':[]}


    def clustering(self,cluster):
        for i in range(len(data)):
            e = np.zeros(self.n)
            for j in range(self.n):
                e[j]=(euclid(self.data[i],self.cluster[j]['center']))
            index=np.argmin(e)
            cluster[index]['data'].append(data[i])
        return cluster


    #각 그룹의 아이템의 평균값으로 센터값 갱신
    def update_center(self):
        points=[]
        self.cluster=self.clustering(self.cluster)
        for i in range(self.n):
            c_data=self.cluster[i]['data']
            for j in range(len(c_data)):
                points.append(c_data[j])
                self.cluster[i]['center']=np.mean(points,axis=0)
        return self.cluster


    #갱신된 센터값으로 다시 clustering 하고 기존 센터값이랑 비교, 같으면 중지
    def update(self):
        self.cluster = self.update_center()
        old_cluster = copy.deepcopy(self.cluster)
        self.cluster = self.clustering(self.cluster)
        for i in range(self.n):
            if np.any(self.cluster[i]['center'] == old_cluster[i]['center']):
                break
            else:
                self.update()
        return self.cluster


    def fit(self):
        self.init_center()
        self.cluster=self.clustering(self.cluster)
        self.update()

        result,labels =self.get_result(self.cluster)
        draw_graph(result,labels)

    def get_result(self,cluster):
        result=[]
        labels=[]
        for key,value in cluster.items():
            for item in value['data']:
                labels.append(key)
                result.append(item)
        return np.array(result), labels


if __name__=='__main__':
    data=read_data()
    data=norm(data)
    dbscan(data)
    hierarchical(data)

    model=KMeans(data,8)
    model.fit()