import numpy as np
import matplotlib.pyplot as plt
import codecs

names = []
data_list = []
with open('seoul.txt', 'r', encoding='utf-8') as file:
    names = file.readline().rstrip().split('\t')
    lines = file.readlines()
    for line in lines:
        line = line.rstrip()
        data_list.append(line.split('\t'))


data_np = np.array(data_list)
print(data_np)

data_np = np.delete(data_np, [0, 1], 1)
print(data_np)

tmp = data_np.tolist()
names = names[2:]
print(names)

for i in range(75):
    tmp[i] = [int(x) for x in tmp[i]]
print(tmp)

total = []
man = []
woman = []
for i in range(75):
    if i % 3 == 0:
        total.append(tmp[i])
    elif i % 3 == 1:
        man.append(tmp[i])
    else:
        woman.append(tmp[i])

total = np.array(total)
man = np.array(man)
woman = np.array(woman)

print(total)
print(man)
print(woman)


total=total.sum(axis=0)
man=man.sum(axis=0)
woman=woman.sum(axis=0)


total=total.tolist()
man=man.tolist()
woman=woman.tolist()

print(total)
print(man)
print(woman)

plt.figure(figsize=(12,3))
plt.subplot(131)
plt.bar(range(101),total,width=0.6)
plt.subplot(132)
plt.bar(range(101),man,width=0.6)
plt.subplot(133)
plt.bar(range(101),woman,width=0.6)
plt.show()

t=[]
m=[]
w=[]
for i in range(101):
    t.append(str(total[i]))
    w.append(str(man[i]))
    m.append(str(woman[i]))

total = np.array(total)
man = np.array(man)
woman = np.array(woman)

print('\n')
print('계 :',' '.join(t))
print("계 총합 : ",np.sum(total))
print("계 평균 : ",int(np.mean(total)))
print("계 분산 : ",int(np.var(total)),'\n')

print('남자 :',' '.join(t))
print("남자 총합 : ",np.sum(man))
print("남자 평균 : ",int(np.mean(man)))
print("남자 분산 : ",int(np.var(man)),'\n')

print('여자 :',' '.join(t))
print("여자 총합: ",np.sum(woman))
print("여자 평균: ",int(np.mean(woman)))
print("여자 분산: ",int(np.var(woman)),'\n')

