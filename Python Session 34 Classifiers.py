#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import warnings
warnings.filterwarnings('ignore')


# In[2]:


iris=load_iris()


# In[3]:


iris


# In[5]:


iris.feature_names


# In[6]:


iris.keys()


# In[7]:


iris.DESCR


# In[8]:


iris.target


# In[9]:


ds=pd.DataFrame(data=iris.data)
ds
sb.pairplot(ds)


# In[28]:


x=iris.data
y=iris.target
y


# In[ ]:





# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=22,random_state=43)


# In[34]:


mnb=MultinomialNB()
mnb.fit(x_train,y_train)
predmnb=mnb.predict(x_test)
print(accuracy_score(y_test,predmnb))
print(confusion_matrix(y_test,predmnb))
print(classification_report(y_test,predmnb))


# In[37]:


from sklearn.neighbors import KNeighborsClassifier


# In[38]:


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
knn.score(x_train,y_train)
predknn=knn.predict(x_test)
print(accuracy_score(y_test,predknn))
print(confusion_matrix(y_test,predknn))
print(classification_report(y_test,predknn))

