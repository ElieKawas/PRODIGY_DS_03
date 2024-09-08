#Task 3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
     
df=pd.read_csv("C:/Users/hp/Desktop/Intership/Prodigy Task 3/bank.csv",sep=';')


#Data Pre-processing and Cleaning
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())

#missing values
missing_values=df.isnull().sum()
print(missing_values)

#duplicates
duplicates=df.duplicated().sum()
print(duplicates)

#Visualization
#heatmap
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.title("Correlation Map")
plt.show()

#y=deposit

#histogram

#age
sns.histplot(df['age'])
plt.title("Age distribution")
plt.show()

#job
sns.histplot(df['job'])
plt.title("Job Distribution")
plt.show()

#education
sns.histplot(df['education'])
plt.title("Education Distribution")
plt.show()

#education and deposit
sns.countplot(x='education',data=df,hue="y")
plt.title("Education Distribution")
plt.show()

#marital distribution
sns.countplot(x="marital",data=df,hue="y")
plt.title("Marital Distribution")
plt.show()

#loan distribution
sns.countplot(x="loan",data=df,hue="y")
plt.title("Personal Loan Distribution and Deposit")
plt.show()

sns.countplot(x="housing",data=df,hue="y")
plt.title("Housing Loan Distribution and Deposit")
plt.show()


#Decision Tree

#Model building
#train - test split

X=df.drop(['job','marital','education','contact','month','default','housing','loan','poutcome','y'],axis=1)
y=df.y

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_X, train_y)
print('Train Score: {}'.format(decision_tree.score(train_X, train_y)))
print('Test Score: {}'.format(decision_tree.score(test_X, test_y)))
ypred = decision_tree.predict(test_X)

clf = DecisionTreeClassifier(criterion= 'gini', max_depth= 5, min_samples_leaf = 3)
clf.fit(train_X, train_y)

#Confusion Matrix
pred_y=clf.predict(test_X)
cm=confusion_matrix(pred_y,test_y)
ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot()
plt.show()

#Visualizing the Tree
from sklearn import tree
fig=plt.figure(figsize=(25,20))
t=tree.plot_tree(clf,filled=True,feature_names=X.columns)
plt.show()
