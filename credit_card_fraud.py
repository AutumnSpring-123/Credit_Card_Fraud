#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import re


# In[4]:


import sklearn


# In[83]:


import seaborn as sns


# In[84]:


import matplotlib.pyplot as plt


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import warnings


# In[9]:


warnings.filterwarnings('ignore')


# In[10]:


from collections import Counter


# In[153]:


from sklearn.ensemble import RandomForestClassifier, VotingClassifier


# In[148]:


from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split


# In[13]:


from sklearn.feature_selection import SelectFromModel, SelectKBest


# In[14]:


from sklearn.pipeline import make_pipeline


# In[15]:


from sklearn.model_selection import StratifiedKFold


# In[16]:


from sklearn.model_selection import cross_val_score


# In[207]:


from imblearn.over_sampling import RandomOverSampler


# In[17]:


from sklearn.model_selection import GridSearchCV


# In[18]:


sns.set(style='white', context='notebook', palette='deep')


# In[19]:


pd.options.display.max_columns = 100


# In[190]:


df = pd.read_csv('D:\Datasets\\Dataset\\creditcard.csv')
df.head()


# In[192]:


df.describe()


# In[193]:


df.info()


# In[194]:


df.isnull().sum()*100/df.shape[0]


# In[195]:


dt = list(df[((df.isnull().sum(axis=1)/df.shape[1])*100)>5].index)
print(dt)


# In[71]:


get_ipython().system('pip install mlxtend')


# In[72]:


get_ipython().system('pip install xgboost')


# In[74]:


from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score


# In[151]:


from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[76]:


from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, RandomForestRegressor
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor


# In[281]:


from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA


# In[197]:


sns.countplot(data=df,x="Class")
count = df["Class"].value_counts()
plt.yticks(count)
plt.show()


# In[198]:


sc = StandardScaler()
amount = df['Amount'].values
df['Amount'] = sc.fit_transform(amount.reshape(-1,1))


# In[199]:


df.drop(['Time'],axis=1,inplace=True)


# In[200]:


df.shape


# In[201]:


df.drop_duplicates(inplace=True)


# In[202]:


df.shape


# In[108]:


from sklearn.preprocessing import StandardScaler


# In[203]:


X = df.drop('Class',axis=1).values
Y = df['Class'].values


# In[204]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=1)


# In[208]:


ros = RandomOverSampler()


# In[209]:


X_train1,Y_train1 = ros.fit_resample(X_train,Y_train)


# In[210]:


X_test1,Y_test1 = ros.fit_resample(X_test,Y_test)


# In[211]:


lr = LogisticRegression()


# In[126]:


X_train , X_test ,y_train , y_test = train_test_split(X,y,test_size = 0.2 ,random_state = 42 ,stratify=y)


# In[212]:


lr = create_model(lr)


# In[213]:


lr.fit(X_train1,Y_train1)


# In[214]:


Y_pred = lr.predict(X_test1)


# In[215]:


print(classification_report(Y_test1,Y_pred))


# In[216]:


def create_model1(model1):
    model1.fit(X_train1,Y_train1)
    Y_pred1=model1.predict(X_test1)
    print(classification_report(Y_test1,Y_pred1))
    print('Confustion Matrix')
    print(confusion_matrix(Y_test1,Y_pred1))
    return model1


# In[217]:


lr = LogisticRegression()


# In[218]:


lr = create_model1(lr)


# In[219]:


dt = DecisionTreeClassifier()


# In[220]:


dt = create_model1(dt)


# In[221]:


dt1 = DecisionTreeClassifier(max_depth=4,criterion="entropy") 


# In[222]:


dt1 = create_model1(dt1)


# In[223]:


lr = LogisticRegression()
dt2 = DecisionTreeClassifier()
dt3 = DecisionTreeClassifier(criterion="entropy")


# In[224]:


model_list=[('Logistic',lr),('Decision_Tree_Gini',dt2),('Decision_Tree_Entropy',dt3)]


# In[225]:


vc1 = VotingClassifier(estimators=model_list)


# In[226]:


model = create_model1(vc1)


# In[227]:


vc2 = VotingClassifier(estimators=model_list,voting="soft")


# In[228]:


model = create_model1(vc2)


# In[229]:


bc = BaggingClassifier(LogisticRegression(),n_estimators=10,max_samples=50,random_state=1)


# In[230]:


model = create_model1(bc)


# In[231]:


bc1 = BaggingClassifier(LogisticRegression(),n_estimators=10,max_samples=500,random_state=1,bootstrap=False)


# In[232]:


model = create_model1(bc1)


# In[233]:


rf = RandomForestClassifier(max_depth = 4)


# In[234]:


rf = create_model1(rf)


# In[235]:


lr = LogisticRegression()
dt2 = DecisionTreeClassifier() # by-default Gini Index
dt3 = DecisionTreeClassifier(criterion="entropy")


# In[236]:


model_list = [lr,dt1,dt2]


# In[237]:


meta = LogisticRegression()


# In[238]:


sc = StackingClassifier(classifiers=model_list,meta_classifier=meta)


# In[239]:


model = create_model1(sc)


# In[240]:


ada = AdaBoostClassifier(n_estimators=100)


# In[241]:


model = create_model1(ada)


# In[242]:


xgb = XGBClassifier(max_depth = 4)


# In[243]:


xgb = create_model1(xgb)


# In[244]:


svc = LinearSVC(random_state=3)


# In[245]:


svc = create_model1(svc)


# In[246]:


svc = LinearSVC(random_state=1,C=0.05)


# In[247]:


svc=create_model1(svc)


# In[248]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[249]:


anova = SelectKBest(score_func=f_regression,k=15)


# In[250]:


X_train_imp = anova.fit_transform(X_train1,Y_train1)
X_test_imp = anova.transform(X_test1)


# In[251]:


anova.get_support()


# In[252]:


lr1=LinearRegression()


# In[253]:


lr1.fit(X_train_imp,Y_train1)


# In[254]:


lr1.score(X_test_imp,Y_test1)


# In[255]:


df.head(10)


# In[256]:


X = df.drop('Class',axis=1) #input variable
Y = df['Class']


# In[257]:


X.columns


# In[258]:


columns = []
for col in X:
    columns.append(col)
    print(columns)


# In[259]:


columns = []
for col in X:
    columns.append(col)
    X_new = df[columns]
    X_train1,X_test1,Y_train1,Y_test1 = train_test_split(X_new,Y,test_size=0.3,random_state=1)
    lr = LinearRegression()
    lr.fit(X_train1,Y_train1)
    score = lr.score(X_test1,Y_test1)
    print('Column : ',col, ' Score : ',score)


# In[260]:


columns = []
X_new = X
n_col = X_new.shape[1]
for i in range(n_col,0,-1): 
    columns.append(X_new)
    X_train1,X_test1,y_train1,y_test1 = train_test_split(X_new,Y,test_size=0.3,random_state=1)
    lr = LinearRegression()
    lr.fit(X_train1,y_train1)
    score1 = lr.score(X_test1,y_test1)
    print('Column : ',i, 'Score : ',score1)
    X_new = X_new.iloc[:,:-1]
    print('After Remove Column : ', i)


# In[261]:


lr = LinearRegression()


# In[262]:


lr.fit(X_train1,Y_train1)


# In[263]:


training_score = lr.score(X_train1,Y_train1)
print('Training Score : ',training_score)


# In[264]:


testing_score = lr.score(X_test1,Y_test1)
print('Testing Score : ',testing_score)


# In[265]:


m = lr.coef_
m = np.round(m,2)
print(m)


# In[266]:


lr.intercept_


# In[267]:


l1 = Lasso(1000)
l1.fit(X_train1,Y_train1)


# In[268]:


l1.coef_


# In[269]:


m = lr.coef_
a  =[]
for i in m:
    i = np.round(i,2)
    a.append(i)
print(a)


# In[270]:


m1 = l1.coef_
b = []
for i in m1:
    i = np.round(i,3)
    b.append(i)
print(b)


# In[271]:


c = X.columns
L = list(zip(c,a,b))

df1 = pd.DataFrame(L, columns=['Column','Original_Slope','Lasso_Slope'])
df1


# In[272]:


print('Score : ')
for i in range(200,1001,50):
    l1 = Lasso(i)
    l1.fit(X_train1,Y_train1)
    score = l1.score(X_test1,Y_test1)
    print(np.round(score,2))


# In[273]:


pca = PCA(n_components=1,random_state=1)
X_train_pca = pca.fit_transform(X_train1,Y_train1)


# In[274]:


X_test_pca = pca.transform(X_test1)


# In[275]:


lr = LinearRegression()


# In[276]:


lr.fit(X_train_pca,Y_train1)


# In[277]:


lr.score(X_test_pca,Y_test1)

