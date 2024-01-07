#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest,f_classif,chi2
file_path = pd.read_csv('nba2021.csv')
df=pd.DataFrame(file_path)

selected_features = ['G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
       '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
       'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

# selected_features = ['G', 'GS', 'MP', 'FG', 'FGA', '3P',
#        '3PA','2P', '2PA','FT', 'FTA', 'FT%', 'ORB',
#        'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

#
X_df= df[selected_features]
Y = df['Pos']
model=SelectKBest(f_classif,k=15)
new=model.fit(X_df,Y)
X_new=new.transform(X_df)
col=new.get_support(indices=True)
X=X_df.iloc[:,col]
print(X.columns)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(Y)
onehot_encoder = OneHotEncoder(sparse=False)


scaler = StandardScaler()
X_encoded = scaler.fit_transform(X)


# Split the data into training and test sets (75% for training, 25% for testing)
train_feature, test_feature, train_class, test_class = train_test_split(
    X_encoded, y_encoded, test_size=0.25, random_state=0)
# linearsvm = LinearSVC(random_state=0).fit(train_feature, train_class)
linear_svm = SVC(kernel='linear', max_iter=10000000)
linear_svm.fit(train_feature, train_class)


print("Training set accuracy: {:.3f}".format(linear_svm.score(train_feature, train_class)))
print("Test set accuracy: {:.3f}".format(linear_svm.score(test_feature, test_class)))

linear_svm_predictions = linear_svm.predict(test_feature)
#linear_svm_confusion_matrix = confusion_matrix(test_class, linear_svm_predictions)

print("LinearSVM Confusion Matrix:")
print(pd.crosstab(test_class, linear_svm_predictions, rownames=['True'], colnames=['Predicted'], margins=True))

scores = cross_val_score(linear_svm, X_encoded, y_encoded, cv=10)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("Max cross-validation scores: {}".format(max(scores)))
print("Min cross-validation scores: {}".format(min(scores)))

svm1_class_report = classification_report(test_class, linear_svm_predictions, target_names = ['PG', 'SG', 'SF', 'PF', 'C'])
print(svm1_class_report)


# In[ ]:




