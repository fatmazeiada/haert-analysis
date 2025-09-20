#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


data=pd.read_csv(r"C:\Users\AS\Downloads\heart.csv")


# ## Explore Data

# In[ ]:


data.head(7)


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# In[ ]:


sns.heatmap(data.isnull())


# In[ ]:


data.duplicated().sum ()


# ## Analysis

# In[ ]:


data["trestbps"].value_counts()


# In[ ]:


data.hist(figsize=(20,15))
plt.show()


# In[ ]:


cor=data.corr()


# In[ ]:


sns.heatmap(cor)


# In[ ]:


sns.heatmap(cor.rank(axis="columns"), annot=True, fmt=".1f" ,linewidth=.5)




# ## processing Data

# In[ ]:


get_ipython().system('C:\\Users\\AS\\AppData\\Local\\Programs\\Python\\Python313\\python.exe -m pip install --upgrade pip')


# In[ ]:


get_ipython().system('pip install scikit-learn')


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()


# In[ ]:


obj=data.select_dtypes(include ="object")
non_obj=data.select_dtypes(exclude="object")


# In[ ]:


for i in range(0,obj.shape[1]):
    obj.iloc[:, i] = lab.fit_transform(obj.iloc[:, i])


# In[ ]:


df=pd.concat([obj,non_obj],axis=1)


# In[ ]:


df


# ## Model

# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[ ]:


x=df.drop(["target"],axis=1)
y=df["target"]


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)


# In[ ]:


def all(model):
    model.fit(x_train, y_train)           # تدريب الموديل
    pre = model.predict(x_test)           # التنبؤ بالبيانات الجديدة
    print(confusion_matrix(y_test, pre))  # مصفوفة الالتباس
    print(classification_report(y_test, pre))  # تقرير التصنيف


# In[ ]:


model1 = KNeighborsClassifier()
all(model1)


# In[ ]:


model2=DecisionTreeClassifier()
all(model2)



# In[ ]:


model3=GaussianNB()
all(model3)


# In[ ]:


model4=SVC()
all(model4)


# In[ ]:


model5=RandomForestClassifier()
all(model5)


# In[ ]:


model6=GradientBoostingClassifier()
all(model6)


# In[ ]:




