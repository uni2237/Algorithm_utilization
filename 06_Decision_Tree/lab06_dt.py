import pandas as pd
import matplotlib.pyplot as plt
import pydot as pydot
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score


data=pd.read_csv('data/train.csv')
print(data)


print(data.info())
print('null 개수',data.isnull().sum()) #age,cabin,embarked에만 null 있음
data['Age'].fillna(data['Age'].mean(),inplace=True)
data['Cabin'].fillna('N',inplace=True)
data['Embarked'].fillna('N',inplace=True)
print('null 개수',data.isnull().sum())

print(data['Cabin'])
data['Cabin']=data['Cabin'].str[:1]
print(data['Cabin'].value_counts())

sns.barplot(
    data=data,
    x='Sex',
    y='Survived'
)
plt.show()

sns.barplot(
    data=data,
    x='Pclass',
    y='Survived',
    hue='Sex'
)
plt.show()

sns.barplot(
    data=data,
    x='Embarked',
    y='Survived',
    hue='Sex'
)
plt.show()

def get_category(age):
    c = ''
    if age <= 5:
        c = '0~5'
    elif age <= 12:
        c = '~12'
    elif age <= 18:
        c = '~18'
    elif age <= 25:
        c = '~25'
    elif age <= 35:
        c = '~35'
    elif age <= 60:
        c = '~60'
    else:
        c = 'old'
    return c


age_group = ['0~5', '~12', '~18', '~25', '~35', '~60', 'old']

data['Age_group'] = data['Age'].apply(lambda x: get_category(x))
sns.barplot(data=data, x='Age_group', y='Survived', hue='Sex', order=age_group)
plt.show()
data.drop('Age_group', axis=1, inplace=True)



def fillna(data):
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Cabin'].fillna('N', inplace=True)
    data['Embarked'].fillna('N', inplace=True)
    data['Fare'].fillna(0, inplace=True)
    return data


def drop(data):
    data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return data

#Label Encoding 활용
def format(data):
    data['Cabin'] = data['Cabin'].str[:1]
    label = ['Cabin','Sex','Embarked']
    for i in label:
        le = LabelEncoder()
        data[i]=le.fit_transform(data[i])
    return data


def transform(data):
    data = fillna(data)
    data = drop(data)
    data = format(data)
    return data

train=pd.read_csv('data/train.csv')
test=pd.read_csv('data/test.csv')
sub=pd.read_csv('data/gender_submission.csv')

y_train=train['Survived']
x_train=train.drop('Survived',axis=1)

x_train=transform(x_train)
test=transform(test)

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.1,random_state=7)

dt=DecisionTreeClassifier(random_state=7)
dt.fit(x_train,y_train)
pred=dt.predict(test)
print('정확도:{0:.4f}'.format(accuracy_score(sub['Survived'],pred)))

export_graphviz(dt,out_file='dt.dot',class_names=['No','Yes'],
                feature_names=x_train.columns,impurity=False,filled=True)
(graph, )=pydot.graph_from_dot_file('dt.dot',encoding='utf-8')
graph.write_png('dt.png')
submission=pd.DataFrame({"PassengerID":sub['PassengerId'],
                         "Survived":pred})
submission.to_csv('my_submission.csv',index=False)