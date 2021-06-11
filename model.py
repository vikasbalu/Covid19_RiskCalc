import pandas as pd
import pickle
import seaborn as sns

covid = pd.read_csv("data.csv")

from sklearn.preprocessing import LabelEncoder
e=LabelEncoder()

covid['Breathing Problem']=e.fit_transform(covid['Breathing Problem'])
covid['Fever']=e.fit_transform(covid['Fever'])
covid['Dry Cough']=e.fit_transform(covid['Dry Cough'])
covid['Sore throat']=e.fit_transform(covid['Sore throat'])
covid['Running Nose']=e.fit_transform(covid['Running Nose'])
covid['Asthma']=e.fit_transform(covid['Asthma'])
covid['Chronic Lung Disease']=e.fit_transform(covid['Chronic Lung Disease'])
covid['Headache']=e.fit_transform(covid['Headache'])
covid['Heart Disease']=e.fit_transform(covid['Heart Disease'])
covid['Diabetes']=e.fit_transform(covid['Diabetes'])
covid['Hyper Tension']=e.fit_transform(covid['Hyper Tension'])
covid['Abroad travel']=e.fit_transform(covid['Abroad travel'])
covid['Contact with COVID Patient']=e.fit_transform(covid['Contact with COVID Patient'])
covid['Attended Large Gathering']=e.fit_transform(covid['Attended Large Gathering'])
covid['Visited Public Exposed Places']=e.fit_transform(covid['Visited Public Exposed Places'])
covid['Family working in Public Exposed Places']=e.fit_transform(covid['Family working in Public Exposed Places'])
covid['Wearing Masks']=e.fit_transform(covid['Wearing Masks'])
covid['Sanitization from Market']=e.fit_transform(covid['Sanitization from Market'])
covid['COVID-19']=e.fit_transform(covid['COVID-19'])
covid['Dry Cough']=e.fit_transform(covid['Dry Cough'])
covid['Sore throat']=e.fit_transform(covid['Sore throat'])
covid['Gastrointestinal ']=e.fit_transform(covid['Gastrointestinal '])
covid['Fatigue ']=e.fit_transform(covid['Fatigue '])


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

x=covid.drop('COVID-19',axis=1)
y=covid['COVID-19']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

corelation = covid.corr()
sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
acc_logreg=model.score(x_test, y_test)*100
print("logisticRegression is=",acc_logreg)
print(confusion_matrix(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
acc_randomforest=rf.score(x_test, y_test)*100
print("RandomForestClassifier is =",acc_randomforest)
print(confusion_matrix(y_test, y_pred))

from sklearn.ensemble import GradientBoostingClassifier
GBR = GradientBoostingClassifier(n_estimators=100, max_depth=4)
GBR.fit(x_train, y_train)
y_pred = GBR.predict(x_test)
acc_gbk=GBR.score(x_test, y_test)*100
print("GradientBoostingClassifier is =",acc_gbk)
print(confusion_matrix(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn=knn.score(x_test, y_test)*100
print("KNeighborsClassifier is =",acc_knn)
print(confusion_matrix(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
t = DecisionTreeClassifier()
t.fit(x_train,y_train)
y_pred = t.predict(x_test)
acc_decisiontree=t.score(x_test, y_test)*100
print("Decisiontree is =",acc_decisiontree)
print(confusion_matrix(y_test, y_pred))

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc_gaussian= model.score(x_test, y_test)*100
print("GaussianNB is =",acc_gaussian)
print(confusion_matrix(y_test, y_pred))

from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc_svc=clf.score(x_test, y_test)*100
print("SVM is =",acc_svc)
print(confusion_matrix(y_test, y_pred))

dtc=rf.fit(x,y)

pickle.dump(dtc, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

covid.describe()

covid.shape

covid.columns

covid.isnull().sum()

sns.heatmap(covid.isnull(), yticklabels=False)

corelation = covid.corr()
sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True)

symptoms1 = covid.drop(['Chronic Lung Disease', 'Headache', 'Heart Disease', 
                        'Diabetes', 'Hyper Tension', 'Fatigue ', 'Gastrointestinal ',
                        'Abroad travel', 'Contact with COVID Patient', 
                        'Attended Large Gathering', 'Visited Public Exposed Places', 
                        'Family working in Public Exposed Places', 'Wearing Masks', 
                        'Sanitization from Market'], axis=1)
corelation1 = symptoms1.corr()
sns.heatmap(corelation1, xticklabels=corelation1.columns, yticklabels=corelation1.columns, annot=True)

symptoms2 = covid.drop(['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat', 
                        'Running Nose', 'Asthma', 'Gastrointestinal ', 'Abroad travel', 
                        'Contact with COVID Patient', 'Attended Large Gathering', 
                        'Visited Public Exposed Places', 
                        'Family working in Public Exposed Places', 'Wearing Masks', 
                        'Sanitization from Market'], axis=1)
corelation2 = symptoms2.corr()
sns.heatmap(corelation2, xticklabels=corelation2.columns, yticklabels=corelation2.columns, annot=True)

symptoms3 = covid.drop(['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat', 
                        'Running Nose', 'Asthma', 'Chronic Lung Disease', 'Headache', 
                        'Heart Disease', 'Diabetes', 'Hyper Tension', 'Fatigue ', 
                        'Wearing Masks', 'Sanitization from Market'], axis=1)
corelation3 = symptoms3.corr()
sns.heatmap(corelation3, xticklabels=corelation3.columns, yticklabels=corelation3.columns, annot=True)

sns.regplot(x='Abroad travel',y='COVID-19',data=covid)

sns.regplot(x='Running Nose',y='COVID-19',data=covid)

sns.displot(covid['Fever'])

sns.displot(covid['Family working in Public Exposed Places'])

sns.displot(covid['COVID-19'])

sns.heatmap(xticklabels=corelation.columns, yticklabels=False, data=covid)