#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression


# In[2]:


#load in data
data = pd.read_csv('C://Users/alyss/Desktop/DG/Week-4/toy_dataset.csv')
data.head()


# In[3]:


data.drop(columns='Number', inplace=True)


# In[4]:


data['City'].value_counts() #check city values


# In[5]:


#convert cities to number values
def convert_numeric(word):
    word_dict = {'New York City':1, 'Los Angeles':2, 'Dallas':3, 'Mountain View':4, 'Austin':5, 'Boston':6, 'Washington D.C.':7, 'San Diego':8,}
    return word_dict[word]

data['City'] = data['City'].apply(lambda x : convert_numeric(x))
data.head()


# In[6]:


#convert gender to numeric
gender = []
for i in range(0, len(data)):
    if data['Gender'][i] == 'Male':
        gender.append(0)
    else:
        gender.append(1)


# In[7]:


data.drop(columns='Gender', inplace=True)
data['Gender'] = gender
data.head()


# In[8]:


#convert illness to boolean
illness = []
for i in range(0, len(data)):
    if data['Illness'][i] == 'No':
        illness.append(0)
    else:
        illness.append(1)


# In[9]:


data.drop(columns='Illness', inplace=True)
data['Illness'] = illness
data.head()


# In[10]:


#define features and target
features = data.drop(columns='Illness')
target = data['Illness']


# In[11]:


#basic classification model
model = LogisticRegression()


# In[12]:


#fit model
model.fit(features, target)


# In[13]:


#save model to disk
pickle.dump(model, open('model.pkl', 'wb'))


# In[14]:


#load model
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[3,50,40000,0]]))

