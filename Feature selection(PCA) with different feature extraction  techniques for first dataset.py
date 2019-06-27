#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing pandas library

import pandas as pd 


# In[1062]:


data=open("Machine2.csv","r")  #open the csv file


# In[1063]:


data=pd.read_csv("Machine2.csv",encoding = "ISO-8859-1") 


# In[1064]:


data.shape ##check the shape of your data 


# In[1065]:


data.isnull().head() ##Check all null value in dataset


# In[1066]:


data.isnull().sum() #check how many null values present in each column


# In[1067]:


modifieddata=data.fillna(" ") #Fill all null or empty cells in original DataFrame with an empty space and set that to a new DataFrame variabl


# In[1068]:


modifieddata.isnull().sum() #Verify that you no longer have any null values by running 


# In[1069]:


modifieddata.to_csv('modifieddata12.csv',index=False) ##Save modified dataset to a new CSV


# In[1070]:


md=pd.read_csv("modifieddata12.csv") #save modified csv into new variable in order to perform ferther operation


# In[1071]:


md['word_count'] = md['commentText'].apply(lambda x: len(str(x).split(" ")))
md[['commentText','word_count']].head()   #count the number of words in each comment


# In[1072]:


md['char_count'] = md['commentText'].str.len() ## this also includes spaces
md[['commentText','char_count']].head()


# In[1073]:


#Average words
def avg_word(sentence):
  words = sentence.split()
  try:
    return (sum(len(word) for word in words)/len(words))
  except:
      print(sentence)

data['avg_word'] = data['commentText'].apply(lambda x: avg_word(x))
print(data[['commentText','avg_word']])


# In[ ]:


import numpy as np  #import numpy library


# In[ ]:


import pandas as pd


# In[ ]:


from nltk.corpus import stopwords


# In[ ]:


import nltk #import natural language processing toolkit


# In[ ]:


from nltk.corpus import stopwords     ###Count number of stopwords in each comment
stop = stopwords.words('english')

md['stopwords'] = md['commentText'].apply(lambda x: len([x for x in x.split() if x in stop]))
md[['commentText','stopwords']].head()


# In[ ]:


import pandas as pd   ##Count number of special character in each comments
from nltk.stem import PorterStemmer, WordNetLemmatizer
porter_stemmer = PorterStemmer()


# In[ ]:


md['commentText'] = md['commentText'].apply(lambda x: " ".join(x.lower() for x in x.split()))
md['commentText'].head()    


# In[1074]:


md['hastags'] = md['commentText'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
md[['commentText','hastags']].head() #count number of special character in ech comment


# In[1075]:


md['upper'] = md['commentText'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
md[['commentText','upper']].head() #identifying the number of uppercase letters in comment


# In[1076]:


md['commentText'] = md['commentText'].apply(lambda x: " ".join(x.lower() for x in x.split()))
md['commentText'].head() #transform all the comments in to lower case


# In[1077]:


md['commentText'] = md['commentText'].str.replace('[^\w\s]','')
md['commentText'].head()    ##removing punctuations 


# In[1078]:


from nltk.corpus import stopwords  ###removal of stop words
stop = stopwords.words('english')
md['commentText'] = md['commentText'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
md['commentText'].head()  ###removal of stop words


# In[ ]:


tok_comments= md['commentText'].apply(lambda x: x.split())  #performing tokenization
tok_comments


# In[ ]:


from nltk.stem import PorterStemmer  ##stemming 
st = PorterStemmer()
md['commentText'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


# In[1080]:


from textblob import Word ##lemmatization
md['commentText'] = md['commentText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
md['commentText']


# In[1081]:


#import numpy as np  
#import re  
#import nltk  
#from sklearn.datasets import load_files  
#nltk.download('stopwords')  
#import pickle  
#from nltk.corpus import stopwords  


# In[1082]:


import numpy as np  
import re  
import nltk 


# In[1083]:


pip install sklearn #installing Scikit learning library


# In[1084]:



from sklearn.datasets import load_files 


# In[1085]:


X, y = md.commentText, md.Lables  #assign the variable to each attribute 


# In[1086]:


documents = []


# In[1087]:


for sen in range(0, len(X)): 
    
    document = re.sub(r'\W', ' ', str(X[sen]))


# In[1088]:


document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

for text in md.commentText:
   documents.append(text)


# In[1089]:


#import library for count vectorization similarly we can import for TF-IDF vectorization .

from sklearn.feature_extraction.text import CountVectorizer

#from sklearn.feature_extraction.text import TfidfVectorizer


# In[1090]:


#Code for countvectorization
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))  
X = vectorizer.fit_transform(documents).toarray()

#similarly we can run below code for TF-IDF and frequency vectorization

#Code for TF-IDF vectorization

#tfidfconverter = TfidfVectorizer( stop_words=stopwords.words('english')) 
#X = tfidfconverter.fit_transform(documents).toarray()

#Code for Frequency vectorization

#vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
#X = vectorizer.fit_transform(documents).toarray()


# In[1091]:


#from sklearn.feature_selection import SelectKBest

#from sklearn.decomposition import PCA  #Import library for pricimple component analysis


# In[1092]:


array = md.values


# In[ ]:





# In[1093]:


from sklearn.model_selection import train_test_split 


# In[1094]:


#split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 


# In[1095]:


from sklearn.decomposition import PCA


# In[1097]:



pca = PCA(n_components=2) #We can icrease and decrease the number of components as per requirement
fit = pca.fit(X)


# In[1098]:


print(fit.components_)


# In[1103]:


from sklearn.model_selection import GridSearchCV


# In[1104]:


#############################Random forest###############################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


# In[1105]:


rfc=RandomForestClassifier(random_state=0)
param_grid = {
    'n_estimators': [1,2,3,4,5,6,7,8,9,10]
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train, y_train)


r=CV_rfc.best_params_
print(r)


# In[1106]:


classifier = RandomForestClassifier(n_estimators=100, random_state=0)  #we can change n_estimtors and random state as per our requirement
print ((cross_val_score(classifier, X_train, y_train, scoring='accuracy',cv=10)))


# In[1017]:


#train the model using training dataset 
classifier.fit(X_train, y_train)


# In[1018]:


#predict the output on test dataset 
y_pred = classifier.predict(X_test)  


# In[1019]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#print the result 

print(confusion_matrix(y_test,y_pred))  
print ('F1 score:', f1_score(y_test, y_pred))
print ('Recall:', recall_score(y_test, y_pred))
print ('Precision:', precision_score(y_test, y_pred))

print(accuracy_score(y_test, y_pred))  


# In[1107]:


#################################Logistic Regression #############################################


from sklearn.model_selection import GridSearchCV


# In[1021]:



C = [1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.0,8.0]
param_grid = dict(C=C),


# In[1022]:


import time
lr = LogisticRegression(penalty='l2')
grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 3, n_jobs=-1)

start_time = time.time()
grid_result = grid.fit(X_train, y_train)
#Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# In[1108]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=4.0, dual=True)  #We can change number of parameter in order to get best accuracy for the model
print((cross_val_score(model,X_train, y_train, cv=10)))  # to print cross validation scores 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Logistic Regression")
print(accuracy_score(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred))
print ('Recall:', recall_score(y_test, y_pred))
print ('Precision:', precision_score(y_test, y_pred))


# In[1026]:


##########################Linear SVM#############################################

from sklearn.model_selection import GridSearchCV

from sklearn.svm import LinearSVC


# In[1027]:


lsvm=LinearSVC(random_state=0)
param_grid = {
    'C': [1.0,0.1,0.01,0.01,0.001,2,3,4,5,6,7,8,9,10]
}
CV_rfc = GridSearchCV(estimator=lsvm, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train, y_train)

r=CV_rfc.best_params_
print(r)


# In[1109]:


from sklearn import svm


# In[1110]:


clfr = svm.SVC(C=1,kernel='linear') # can change the number of parameter i.e. value of C in order to get best accuracy of the model
print((cross_val_score(clfr,X_train, y_train, cv=10)))

#train the model using training dataset
clfr.fit(X_train, y_train)

#predict the output 
y_pred = clfr.predict(X_test)

 


# In[1030]:


from sklearn import metrics


# In[1031]:


#Model Accuracy: how often is the classifier correct?
print("SVM Linear:",metrics.accuracy_score(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred))
print ('Recall:', recall_score(y_test, y_pred))
print ('Precision:', precision_score(y_test, y_pred))


# In[1112]:


####################################Decision tree#######################################

from sklearn.tree import DecisionTreeClassifier  #import library


# In[1113]:


from sklearn import metrics 


# In[1114]:



clf = DecisionTreeClassifier()
print((cross_val_score(clf,X_train, y_train, cv=10)))  #Print cross validation scores 


# In[1049]:


#Train the model using train dataset
clf = clf.fit(X_train,y_train)


# In[1050]:


#predict the output for test datset
y_pred = clf.predict(X_test)


# In[1051]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred))
print ('Recall:', recall_score(y_test, y_pred))
print ('Precision:', precision_score(y_test, y_pred))


# In[1115]:


######################### Bernoulli Naive Bayes ################################################

from sklearn.naive_bayes import BernoulliNB #import libraries 


# In[1117]:


bnb = BernoulliNB(alpha=0.1)  #Value of alpha can be change as per requiremnt 
print((cross_val_score(bnb,X_train, y_train, cv=10)))


# In[1054]:


#train the model using train dataset
bnb.fit(X_train, y_train)

#predict the output 
y_pred=bnb.predict(X_test)

#print the result 
print("Bernoulli nayive bayes")
print(accuracy_score(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred))
print ('Recall:', recall_score(y_test, y_pred))
print ('Precision:', precision_score(y_test, y_pred))


# In[1055]:


#------------------Grid search for bernoulli naive bayes----------

from sklearn.model_selection import GridSearchCV


# In[1056]:


# #
clf = BernoulliNB()
param_grid = {'alpha': [0.1,0.001,0.0001,0.00001,1,10,20,30,100] }
CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train, y_train)
r=CV_rfc.best_params_
print(r)


# In[1118]:


#-----------------Gaussian Naive Bayes--------------------
from sklearn.naive_bayes import GaussianNB


# In[1119]:


gnb= GaussianNB()
print((cross_val_score(gnb,X_train, y_train, cv=10)))  #printing cross validation scores 

#train the model using training set
gnb.fit(X_train, y_train)

#predict the output 
y_pred=gnb.predict(X_test)

#print the results 
print("Gaussian naive bayes")
print(accuracy_score(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred))
print ('Recall:', recall_score(y_test, y_pred))
print ('Precision:', precision_score(y_test, y_pred))
#0.651


# In[1120]:


from sklearn import svm
clfpol = svm.SVC(C=0.001, decision_function_shape='ovr', degree=3, gamma= 10, kernel='poly',max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)
print((cross_val_score(clfpol,X_train, y_train, cv=10)))
#65.07
# Train the model using the training sets
clfpol.fit(X_train, y_train)

# #Predict the response for test dataset
y_pred = clfpol.predict(X_test)
print("SVM Polynomial:",metrics.accuracy_score(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred))
print ('Recall:', recall_score(y_test, y_pred))
print ('Precision:', precision_score(y_test, y_pred))


# In[1121]:


clfgs = svm.SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)
print((cross_val_score(clfgs,X_train, y_train, cv=10)))
#0.7239
# #Train the model using the training sets
clfgs.fit(X_train, y_train)

# #Predict the response for test dataset
y_pred = clfgs.predict(X_test)
#
from sklearn import metrics
#
# Model Accuracy: how often is the classifier correct?
print("SVM gaussian:",metrics.accuracy_score(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred))
print ('Recall:', recall_score(y_test, y_pred))
print ('Precision:', precision_score(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




