#!/usr/bin/env python
# coding: utf-8

# In[35]:


#importing libraries 
import numpy as np
import pandas as pd
import matplotlib as plt


# In[36]:


# loading the data 
Data = pd.read_csv("/content/bcr1203-S3_edited.csv") 


# 
# 
# 
# # Processing the data 
# 
# 

# In[37]:


Data


# In[38]:


#Transposing the Data
Data=Data.T
Data


# In[39]:


header = Data.iloc[0] 
Data = Data[1:] 
Data.columns = header 
Data


# In[40]:


#plotting a bar graph for class C and N where C stands for Cancer and N stands for Normal
Data['Class Labels'].value_counts().plot(kind='bar')


# In[41]:


X= Data.iloc[:,2:1371] 


# In[42]:


X


# In[43]:


X.columns


# In[44]:


Y=Data['Class Labels']
Y


# In[ ]:


X["Labels"] = Y


# In[46]:


#converting the labes to 0 and 1 where C is 1 and N is 0
type_dict = {'C': 1,'N': 0}       
X['Labels'] = X.Labels.map(type_dict)
X


# In[47]:


X_final = X.iloc[:, :-1]
Y_final = X.iloc[:,-1]


# In[48]:


Y_final


# 
# 
# #Applying LOOCV model
# 
# 
# 

# In[49]:


#importing librraies and applying model
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final,Y_final, test_size=0.33)
# create loocv procedure
cv = LeaveOneOut()
# create model
model = RandomForestClassifier(random_state=1)
model.fit(X_train,y_train)
# evaluate model
scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# Here after applying the LOOCV model we got the 70% accuracy 

# In[50]:


prediction = model.predict(X_test)
from sklearn.metrics import accuracy_score
scores = accuracy_score(y_test,prediction)
scores


# After applying the model on test data we got the accuracy of 85%

# In[51]:


#classification report
# here will get the precision, recall and f1-score
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))


# The score we get for 0 (N) is precision = 0.83, recall = 0.95, f1-score = 0.89
# 
# The score we get for 1 (C) is precision = 0.90, recall = 0.69,
# f1-score = 0.78 

# In[52]:


rf_probs = model.predict_proba(X_test)
rf_probs = rf_probs[:, 1]


# In[53]:


from sklearn.metrics import roc_curve, roc_auc_score
rf_auc_m = roc_auc_score(y_test, rf_probs)
print('AUROC = %.3f' % (rf_auc_m))


# In[54]:


# Calculating AUC
from sklearn import metrics
fpr , tpr, thresholds = metrics.roc_curve(y_test, rf_probs)
auc=metrics.auc(fpr, tpr)
print('AUC= %.3f'% (auc))


# The AUC score we got is 0.87

# In[55]:


import matplotlib.pyplot as plt 
rf_fpr_m, rf_tpr_m, _ = roc_curve(y_test, rf_probs)
plt.plot(rf_fpr_m, rf_tpr_m, marker='.', label='(AUROC = %0.3f)' % rf_auc_m)
# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()


# In[56]:


#calculating Kappa score
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(prediction,y_test,weights='quadratic')
kappa


# The Kappa score we got is 0.67
