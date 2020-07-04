import codecs

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy.linalg as la


def draw_graph(data,n):
    plt.figure()
    if n==1:
        plt.scatter(data[:,0],data[:,1],cmap='rainbow')
    else:
        plt.scatter(data,[0]*len(data),cmap='rainbow')
    plt.show()


def read_data():
    data=[]
    with codecs.open('seoul_student.txt','r','utf-8') as file:
        lines=file.readlines()
        del lines[0]
        for index,line in enumerate(lines):
            line=line.rstrip().split('\t')
            data.append(line)
        return data

def norm(data):
    scaler=MinMaxScaler()
    norm_data=scaler.fit_transform(data)
    return norm_data


def sklearn_pca(data,dim):
    x=data
    pca=PCA(n_components=dim) #사용할 주성분 갯수
    pca.fit(x)
    print('sklearn 공분산\n', pca.get_covariance())
    print('sklearn 고유벡터\n',pca.components_, '\n')
    p=PCA(n_components=dim)
    x=p.fit_transform(x)
    draw_graph(x,2)



#공분산 행렬 구함
def get_covariance(data):
    x=[]
    y=[]
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][1])
    x=np.array(x)
    y=np.array(y)
    ex= get_expected_value(data, x)
    ey= get_expected_value(data, y)
    z=np.zeros((2,2))

    for i in range(len(x)):
        z[0,0]+=(x[i]-ex)*(x[i]-ex)
        z[0,1]+=(x[i]-ex)*(y[i]-ey)
        z[1,0]+=(y[i]-ey)*(x[i]-ex)
        z[1,1]+=(y[i]-ey)*(y[i]-ey)
    z=z/(len(x)-1)
    print('my 공분산\n',z)
    return z

#두 열의 각각의 평균값 구해줌-아래 공분산 구할때 ex,ey이가 됨
def get_expected_value(data, x):
    return x.sum(axis=0) / len(data)


#내림차순 정렬하는 함수(공분산매트릭스 구한거를 eig안에 넣으면 w(val):고유값 v(vec):고유벡터
def e_sort(val,vec):
    a=val.argsort()[::-1]
    return vec[a]
    
#첫번째 고유벡터 값 (얘가 제일 큰거니까)과 data dot(행렬곱)
def pca(data,dim=1):
    w,v=la.eig(get_covariance(data))
    v=e_sort(w,v)[1]
    v = e_sort(w, v)
    print('my 고유벡터\n',v)
    pca=np.dot(data,v)
    draw_graph(pca,2)
    


if __name__ == '__main__':
    reduce_dim =1
    data=read_data()
    data=norm(data)
    draw_graph(data,1)
    sklearn_pca(data, reduce_dim)
    pca(data,reduce_dim)

