#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


# Load data
df = pd.read_csv('data.csv')


# In[3]:


# Feature Engineering
df['amount_ratio'] = df['amount'] / df['oldbalanceOrg'].replace(0, 1)
df['balance_diff'] = df['newbalanceDest'] - df['oldbalanceDest']
df['origin_balance_change'] = df['oldbalanceOrg'] - df['newbalanceOrig']


# In[4]:


# Define labels
df['label'] = df['isFraud'] # Start with the fraud labels
df.loc[(df['label'] == 0) & (df['amount'] > 200000), 'label'] = 2  # Mark large non-fraudulent transactions as suspicious


# In[5]:


# Train/Test Split
X = df[['amount', 'amount_ratio', 'balance_diff', 'origin_balance_change']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[6]:


# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[7]:


# Prediction
y_pred = model.predict(X_test)


# In[8]:


# Classification Report
print(classification_report(y_test, y_pred))


# In[9]:


# Assuming 'label' column contains 0 for non-fraudulent, 1 for fraudulent, and 2 for suspicious transactions

# Count the number of transactions in each category
category_counts = df['label'].value_counts()

# Display the counts
print("Number of transactions in each category:")
print(f"Non-Fraudulent (0): {category_counts.get(0, 0)}")
print(f"Fraudulent (1): {category_counts.get(1, 0)}")
print(f"Suspicious (2): {category_counts.get(2, 0)}")


# In[10]:


# Alert Program Function
def alert_system(transaction):
    pred = model.predict(transaction)
    if pred == 1:
        return "ALERT: Fraudulent Transaction!"
    elif pred == 2:
        return "ALERT: Suspicious Transaction!"
    else:
        return "Transaction is Normal."


# In[11]:


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


# In[16]:


from sklearn.model_selection import train_test_split

# Assuming df is your DataFrame and you have already defined X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Access the test data
print("Features in the test set:")
print(X_test)

print("Labels in the test set:")
print(y_test)


# In[17]:


print("First 5 rows of Test Data (X_test):")
print(X_test.head())



# In[18]:


sample_transaction = pd.DataFrame({
    'amount': [45000000],  # Set an example amount
    'amount_ratio': [0.1],  # Set example ratio
    'balance_diff': [-5005],  # Set example balance difference
    'origin_balance_change': [-1452]  # Set example origin balance change
})
alert = alert_system(sample_transaction)
print(alert)


# In[19]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Predictions on the test set
y_pred = model.predict(X_test)


# In[20]:


# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[21]:


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


# In[22]:


# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[23]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score

# Perform 5-fold cross-validation
cross_val_scores = cross_val_score(model, X, y, cv=5, scoring=make_scorer(accuracy_score))

print("Cross-Validation Accuracy Scores:", cross_val_scores)
print("Mean Cross-Validation Accuracy:", cross_val_scores.mean())


# In[24]:


from sklearn.metrics import classification_report

# Predict on the test data
y_pred = model.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred, target_names=['Non-Fraudulent', 'Fraudulent', 'Suspicious'])
print(report)


# In[25]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# Binarize the labels for multi-class ROC
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_binarized.shape[1]

# Train the OneVsRest model
model_ovr = OneVsRestClassifier(model)
y_score = model_ovr.fit(X_train, y_train).predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class')
plt.legend(loc='lower right')
plt.show()


# In[1]:


# Count the number of transactions in each category
category_counts = df['label'].value_counts()

# Display the counts
print("Number of transactions in each category:")
print(f"Non-Fraudulent (0): {category_counts.get(0, 0)}")
print(f"Fraudulent (1): {category_counts.get(1, 0)}")
print(f"Suspicious (2): {category_counts.get(2, 0)}")


# In[ ]:




