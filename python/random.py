#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[3]:


# Load data
df = pd.read_csv('data.csv')


# In[4]:


# Feature Engineering
df['amount_ratio'] = df['amount'] / df['oldbalanceOrg'].replace(0, 1)
df['balance_diff'] = df['newbalanceDest'] - df['oldbalanceDest']
df['origin_balance_change'] = df['oldbalanceOrg'] - df['newbalanceOrig']


# In[5]:


# Define labels
df['label'] = df['isFraud'] # Start with the fraud labels
df.loc[(df['label'] == 0) & (df['amount'] > 200000), 'label'] = 2  # Mark large non-fraudulent transactions as suspicious


# In[6]:


# Train/Test Split
X = df[['amount', 'amount_ratio', 'balance_diff', 'origin_balance_change']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[7]:


# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[8]:


# Prediction
y_pred = model.predict(X_test)


# In[9]:


# Classification Report
print(classification_report(y_test, y_pred))


# In[10]:


# Assuming 'label' column contains 0 for non-fraudulent, 1 for fraudulent, and 2 for suspicious transactions

# Count the number of transactions in each category
category_counts = df['label'].value_counts()

# Display the counts
print("Number of transactions in each category:")
print(f"Non-Fraudulent (0): {category_counts.get(0, 0)}")
print(f"Fraudulent (1): {category_counts.get(1, 0)}")
print(f"Suspicious (2): {category_counts.get(2, 0)}")


# In[11]:


# Alert Program Function
def alert_system(transaction):
    pred = model.predict(transaction)
    if pred == 1:
        return "ALERT: Fraudulent Transaction!"
    elif pred == 2:
        return "ALERT: Suspicious Transaction!"
    else:
        return "Transaction is Normal."


# In[12]:


sample_transaction = pd.DataFrame({
    'amount': [50000],  # Set an example amount
    'amount_ratio': [0.8],  # Set example ratio
    'balance_diff': [30000],  # Set example balance difference
    'origin_balance_change': [-20000]  # Set example origin balance change
})
alert = alert_system(sample_transaction)
print(alert)


# In[15]:


sample_transaction = pd.DataFrame({
    'amount': [58230000],  # Set an example amount
    'amount_ratio': [0.7],  # Set example ratio
    'balance_diff': [80014],  # Set example balance difference
    'origin_balance_change': [385105]  # Set example origin balance change
})
alert = alert_system(sample_transaction)
print(alert)


# In[ ]:




