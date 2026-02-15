#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Financial_ Application_ Behavior_ Dataset.csv")

df.head()


# In[2]:


import os
os.getcwd()


# In[4]:


df.shape
df.info()
df.isnull().sum()


# In[5]:


enroll_rate = df['enrolled'].mean()
enroll_rate


# In[29]:


plt.figure()
plt.bar(['Enroll Rate'], [enroll_rate])
plt.ylim(0,1)
plt.title("Overall Enrollment Rate")
plt.ylabel("Rate")
plt.show()


# In[6]:


df['age_group'] = pd.cut(df['age'],
                         bins=[0,20,30,40,50,60,100],
                         labels=['10s','20s','30s','40s','50s','60+'])

age_enroll = df.groupby('age_group')['enrolled'].mean()
age_enroll


# In[9]:


hour_enroll = df.groupby('hour')['enrolled'].mean().sort_index()
hour_enroll.plot(kind='line', marker='o')
plt.title("Enrollment Rate by Hour")
plt.xlabel("Hour")
plt.ylabel("Enrollment Rate")
plt.xticks(range(0,24))
plt.grid(True)
plt.show()


# In[10]:


df.groupby('enrolled')['numscreens'].mean()


# In[11]:


df['screen_group'] = pd.cut(df['numscreens'],
                            bins=[0,3,6,10,20,100],
                            labels=['1-3','4-6','7-10','11-20','20+'])

screen_enroll = df.groupby('screen_group')['enrolled'].mean()
screen_enroll


# In[25]:


screen_counts = df.groupby('screen_group')['enrolled'].agg(['mean','count'])
screen_counts


# In[12]:


screen_enroll.plot(kind='bar')
plt.title("Enrollment Rate by Screen Count")
plt.show()


# In[40]:


df['screen_list'] = df['screen_list'].astype(str)

df['screens'] = df['screen_list'].str.split(',')

enrolled_screens = df[df['enrolled']==1]['screens'].explode()
not_enrolled_screens = df[df['enrolled']==0]['screens'].explode()
top_enrolled = enrolled_screens.value_counts().head(10)
top_not_enrolled = not_enrolled_screens.value_counts().head(10)

from IPython.display import display

display(top_enrolled)
display(top_not_enrolled)


# In[41]:


comparison = pd.DataFrame({
    "Enrolled Top Screens": top_enrolled,
    "Not Enrolled Top Screens": top_not_enrolled
})

comparison


# In[14]:


df['has_loan'] = df['screen_list'].str.contains("Loan", case=False)
df['has_credit'] = df['screen_list'].str.contains("Credit", case=False)


# In[15]:


df.groupby('has_loan')['enrolled'].mean()
df.groupby('has_credit')['enrolled'].mean()


# In[30]:


df.groupby('has_credit')['enrolled'].mean().plot(kind='bar')
plt.title("Enrollment Rate by Credit Screen Access")
plt.ylabel("Enrollment Rate")
plt.show()


# In[16]:


df.groupby('used_premium_feature')['enrolled'].mean()


# In[17]:


df['first_open'] = pd.to_datetime(df['first_open'], errors='coerce')
df['enrolled_date'] = pd.to_datetime(df['enrolled_date'], errors='coerce')


# In[18]:


enrolled_df = df[df['enrolled'] == 1].copy()
enrolled_df['enroll_time_days'] = (enrolled_df['enrolled_date'] - enrolled_df['first_open']).dt.days

enrolled_df['enroll_time_days'].describe()


# In[34]:


plt.figure()
enrolled_df['enroll_time_days'].hist(bins=20)
plt.title("Distribution of Enrollment Time (Days)")
plt.xlabel("Days to Enroll")
plt.ylabel("Frequency")
plt.show()


# In[35]:


enrolled_df['speed_group'] = pd.cut(enrolled_df['enroll_time_days'],
                                     bins=[-1,1,3,7,100],
                                     labels=['0-1d','2-3d','4-7d','7d+'])


# In[21]:


enrolled_df.groupby('speed_group').size()
enrolled_df.groupby('speed_group')['numscreens'].mean()


# In[38]:


speed_summary = enrolled_df.groupby('speed_group').agg(
    users=('user', 'count'),
    avg_screens=('numscreens', 'mean'),
    premium_rate=('used_premium_feature', 'mean'),
    avg_days=('enroll_time_days', 'mean')
)
speed_summary


# In[36]:


enrolled_df.groupby('speed_group')['numscreens'].mean().plot(kind='bar')
plt.title("Average Screens by Enrollment Speed")
plt.ylabel("Average Screen Count")
plt.show()


# In[37]:


enrolled_df.groupby('speed_group')['used_premium_feature'].mean()


# In[23]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = df[['age','numscreens','used_premium_feature']]
y = df['enrolled']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

model.coef_


# In[39]:


coef = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
coef


# In[24]:


model.score(X_test, y_test)


# In[ ]:




