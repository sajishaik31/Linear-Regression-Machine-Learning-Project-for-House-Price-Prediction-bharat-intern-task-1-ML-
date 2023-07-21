#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


HouseDF = pd.read_csv('USA_Housing.csv')
HouseDF.head() 


# In[4]:


HouseDF.info()


# In[5]:


HouseDF.describe()


# In[6]:


HouseDF.columns


# In[8]:


sns.pairplot(HouseDF)


# In[10]:


sns.distplot(HouseDF['Price'])


# In[11]:


sns.heatmap(HouseDF.corr(), annot=True)


# In[14]:


X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]

y = HouseDF['Price']


# In[16]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)  


# In[17]:


from sklearn.linear_model import LinearRegression 

lm = LinearRegression() 

lm.fit(X_train,y_train) 


# In[26]:


print(lm.intercept_)


# In[28]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient']) 
coeff_df


# In[22]:


predictions = lm.predict(X_test)  
plt.scatter(y_test,predictions)


# In[23]:


sns.distplot((y_test-predictions),bins=50); 


# In[ ]:




