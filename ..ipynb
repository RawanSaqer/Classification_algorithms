import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold,GridSearchCV
 
    

features_names = ['','ID','LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
                  'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3',
                 'PAY_AMT4','PAY_AMT5','PAY_AMT6','target']

db = pd.read_csv('dataset_train_woed.csv',header=0,names=features_names)
db.head()
print(db.shape)
array = db.values
x = array[:,0:25]
y = array[:,25]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6)


models=[]

model1= LogisticRegression(solver='liblinear',multi_class='ovr')
models.append(('Logistic Regression',model1))

model2= GaussianNB()
models.append(('Naive Bayes',model2))

model3= KNeighborsClassifier()
models.append(('K-Nearest Neighbor',model3))

model4= DecisionTreeClassifier()
models.append(('Decision Tree',model4))


print("Before Applying any Features:")


for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("%s:" % (name))
    print("Accuracy Score is %s" % (accuracy_score(y_test, y_pred)), "%")
    print("Confusion Matrix is \n %s" % (confusion_matrix(y_test, y_pred)))
    print("The classification report is \n %s" % (classification_report(y_test, y_pred)))
    print("%s: %f" % ("The Precision Score is", precision_score(y_test, y_pred, average='weighted')))
    print("%s: %f" % ("The Recall Score is", recall_score(y_test, y_pred, average='weighted')))
    print("%s: %f" % ("The F1 Score is", f1_score(y_test, y_pred, average='weighted')))
    
results=[]
names=['LR','NB','KNN','DT']
for name,model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=7)
    cv_results=model_selection.cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)

fig=plt.figure()
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

best=SelectKBest(f_classif,k=10)
xfeature=best.fit_transform(x_train,y_train)
x_newUnivariate=array[:,[3,4,7,10,11,12,14,16,17,18]]
x_train1,x_test1,y_train1,y_test1=train_test_split(x_newUnivariate,y,test_size=0.2,random_state=6)

for name,model in models:
    model.fit(x_train1,y_train1)
    y_pred1=model.predict(x_test1)

results1=[]
for name, model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=7)
    cv_results=model_selection.cross_val_score(model,x_train1,y_train1,cv=kfold,scoring='accuracy')
    results1.append(cv_results)
    names.append(name)

fig=plt.figure()
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
plt.boxplot(results1)
ax.set_xticklabels(names)
plt.show()

