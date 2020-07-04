import codecs
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data_dict={}
categories=[]
names=[]
with codecs.open('seoul_tax.txt','r','utf-8') as file:
    categories=file.readline().rstrip().split('\t')
    lines=file.readlines()
    for line in lines:
        line=line.rstrip()
        line=line.replace(',','')
        line=line.split('\t')
        if line[0] not in data_dict.keys():
            data_dict[line[0]]={}
            data_dict[line[0]]=np.array(line[1:],dtype=np.int64)
            names=data_dict.keys()


def manhattan(x,y):
    distance=0
    for i in range(len(x)):
        distance+=abs(x[i]-y[i])
    return distance


def euclid(x,y):
    distance=0
    for i in range(len(x)):
        distance +=(x[i]-y[i])**2
    return distance**0.5

def fbytype(x, y, ftype):
    if ftype == "cos_dist":
        return cosine_distances(x,y)
    if ftype == "euclid":
        return euclid(x,y)
    if ftype == "manhattan":
        return manhattan(x,y)

for key in data_dict:
    data_dict[key]=data_dict[key].reshape(1,-1)


datal_dict = [data_dict[x] for x in data_dict]
print(datal_dict)

za=np.zeros((25,25))
for x1 in range(len(names)):
    for y1 in range(len(names)):
        za[x1,y1] = fbytype( datal_dict[x1],datal_dict[y1], "cos_dist")

datal_dict = [data_dict[x][0] for x in data_dict]

zb=np.zeros((25,25))
for x1 in range(len(names)):
    for y1 in range(len(names)):
        zb[x1,y1] = fbytype( datal_dict[x1],datal_dict[y1], "euclid")

zc=np.zeros((25,25))
for x1 in range(len(names)):
    for y1 in range(len(names)):
        zc[x1,y1] = fbytype( datal_dict[x1],datal_dict[y1], "manhattan")


plt.figure(figsize=(12,6))

plt.subplot(231)
plt.pcolor(za)
plt.colorbar()

plt.subplot(232)
plt.pcolor(zb)
plt.colorbar()

plt.subplot(233)
plt.pcolor(zc)
plt.colorbar()

#-----------정규화--------------------------------------

sc = MinMaxScaler()

T = sc.fit_transform(za.reshape(625,1))
T = T.reshape((25,25))
plt.subplot(234)
plt.pcolor(T)
plt.colorbar()

T = sc.fit_transform(zb.reshape(625,1))
T = T.reshape((25,25))
plt.subplot(235)
plt.pcolor(T)
plt.colorbar()

T = sc.fit_transform(zc.reshape(625,1))
T = T.reshape((25,25))
plt.subplot(236)
plt.pcolor(T)
plt.colorbar()
plt.show()