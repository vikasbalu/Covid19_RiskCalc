import pandas as pd
import numpy as np
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

covid.describe()

covid.shape

covid.columns

covid.isnull().sum()

sns.heatmap(covid.isnull(), yticklabels=False)

corelation = covid.corr()
sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True)

symptoms1 = covid.drop(['Chronic Lung Disease', 'Headache', 'Heart Disease', 'Diabetes', 'Hyper Tension', 'Fatigue ', 'Gastrointestinal ','Abroad travel', 'Contact with COVID Patient', 'Attended Large Gathering', 
'Visited Public Exposed Places', 'Family working in Public Exposed Places', 
'Wearing Masks', 'Sanitization from Market'], axis=1)
corelation1 = symptoms1.corr()
sns.heatmap(corelation1, xticklabels=corelation1.columns, yticklabels=corelation1.columns, annot=True)

symptoms2 = covid.drop(['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat', 'Running Nose', 'Asthma', 'Gastrointestinal ', 'Abroad travel', 'Contact with COVID Patient', 'Attended Large Gathering', 
'Visited Public Exposed Places', 'Family working in Public Exposed Places', 
'Wearing Masks', 'Sanitization from Market'], axis=1)
corelation2 = symptoms2.corr()
sns.heatmap(corelation2, xticklabels=corelation2.columns, yticklabels=corelation2.columns, annot=True)

symptoms2 = covid.drop(['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat', 'Running Nose', 'Asthma',
                        'Chronic Lung Disease', 'Headache', 'Heart Disease', 'Diabetes', 'Hyper Tension', 
                        'Fatigue ', 'Wearing Masks', 'Sanitization from Market'], axis=1)
corelation2 = symptoms2.corr()
sns.heatmap(corelation2, xticklabels=corelation2.columns, yticklabels=corelation2.columns, annot=True)

sns.regplot(x='Abroad travel',y='COVID-19',data=covid)

sns.regplot(x='Running Nose',y='COVID-19',data=covid)

sns.displot(covid['Fever'])

sns.displot(covid['Family working in Public Exposed Places'])

sns.displot(covid['COVID-19'])

sns.heatmap(xticklabels=corelation.columns, yticklabels=False, data=covid)