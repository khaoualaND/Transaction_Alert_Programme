#!/usr/bin/env python
# coding: utf-8

# In[13]:


from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


import pandas as pd


# In[2]:


df = pd.read_csv('data.csv')


# In[3]:


df['amount_ratio'] = df['amount'] / df['oldbalanceOrg'].replace(0, 1)
df['balance_diff'] = df['newbalanceDest'] - df['oldbalanceDest']
df['origin_balance_change'] = df['oldbalanceOrg'] - df['newbalanceOrig']


# In[4]:


X_clustering = df[['amount', 'amount_ratio', 'balance_diff', 'origin_balance_change']]


# In[5]:


kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_clustering)


# In[6]:


for cluster in range(3):
    print(f"Cluster {cluster}:")
    print(df[df['cluster'] == cluster]['isFraud'].value_counts(normalize=True))
    print()


# In[7]:


# After analyzing the clusters, identify which ones look suspicious
suspicious_clusters = [0]  # Replace with the clusters you've determined as suspicious


# In[8]:


# Label the data
df['label'] = df['isFraud']
df.loc[df['cluster'].isin(suspicious_clusters) & (df['isFraud'] == 0), 'label'] = 2


# In[9]:


# Prepare data for supervised learning
X = X_clustering
y = df['label']


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[11]:


# Train a classifier on labeled data
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[14]:


# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[15]:


# Count the number of transactions in each category
category_counts = df['label'].value_counts()

# Display the counts
print("Number of transactions in each category:")
print(f"Non-Fraudulent (0): {category_counts.get(0, 0)}")
print(f"Fraudulent (1): {category_counts.get(1, 0)}")
print(f"Suspicious (2): {category_counts.get(2, 0)}")


# In[16]:


# Alert Program Function
def alert_system(transaction):
    pred = model.predict(transaction)
    if pred == 1:
        return "ALERT: Fraudulent Transaction!"
    elif pred == 2:
        return "ALERT: Suspicious Transaction!"
    else:
        return "Transaction is Normal."


# In[20]:


import numpy as np

index =  45
transaction = X_test.iloc[[index]]  

# Test the alert_system function
alert = alert_system(transaction)
print(f"Transaction {index}: {alert}")


# In[ ]:




