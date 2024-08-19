#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


# Load and preprocess the data
df = pd.read_csv('data.csv')
df['amount_ratio'] = df['amount'] / df['oldbalanceOrg'].replace(0, 1)
df['balance_diff'] = df['newbalanceDest'] - df['oldbalanceDest']
df['origin_balance_change'] = df['oldbalanceOrg'] - df['newbalanceOrig']


# In[3]:


# Features for Isolation Forest
X = df[['amount', 'amount_ratio', 'balance_diff', 'origin_balance_change']]


# In[4]:


# Apply Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
df['anomaly'] = iso_forest.fit_predict(X)


# In[5]:


# Convert anomaly labels to match your labeling convention
# -1 for outliers (anomalies), 1 for inliers (normal data)
df['label'] = df['isFraud']
df.loc[df['anomaly'] == -1, 'label'] = 2  # Mark anomalies as suspicious


# In[6]:


# Prepare data for supervised learning
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.3, random_state=42)


# In[9]:


# Train a classifier on labeled data
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[10]:


# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[11]:


# Count the number of transactions in each category
category_counts = df['label'].value_counts()

# Display the counts
print("Number of transactions in each category:")
print(f"Non-Fraudulent (0): {category_counts.get(0, 0)}")
print(f"Fraudulent (1): {category_counts.get(1, 0)}")
print(f"Suspicious (2): {category_counts.get(2, 0)}")


# In[12]:


# Alert Program Function
def alert_system(transaction):
    pred = model.predict(transaction)
    if pred == 1:
        return "ALERT: Fraudulent Transaction!"
    elif pred == 2:
        return "ALERT: Suspicious Transaction!"
    else:
        return "Transaction is Normal."


# In[13]:


sample_transaction = pd.DataFrame({
    'amount': [50000],  # Set an example amount
    'amount_ratio': [0.8],  # Set example ratio
    'balance_diff': [30000],  # Set example balance difference
    'origin_balance_change': [-20000]  # Set example origin balance change
})
alert = alert_system(sample_transaction)
print(alert)


# In[20]:


import numpy as np

index =   420
transaction = X_test.iloc[[index]]  

# Test the alert_system function
alert = alert_system(transaction)
print(f"Transaction {index}: {alert}")


# In[22]:


import numpy as np

index =  3
transaction = X_test.iloc[[index]]  

# Test the alert_system function
alert = alert_system(transaction)
print(f"Transaction {index}: {alert}")


# In[18]:


import pandas as pd
import numpy as np

for index in range(len(X_test)):
 
    transaction = X_test.iloc[[index]]  

    
    alert = alert_system(transaction)
    print(f"Transaction {index}: {alert}")


# In[ ]:




