"""
Kaggle Competition: Titanic: Machine Learning from Disaster

@author: Shuai Wang <info.shuai@gmail.com>
"""

########################  Data munging  #######################################
import pandas as pd
import numpy as np
import re
import csv
import pylab as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation


# header=0 when you know row 0 is the header row
train = pd.read_csv('train.csv', header=0)
train_x = train.drop(['Survived'], axis=1)
test_x = pd.read_csv('test.csv', header=0)
combi_x = train_x.append(test_x)
ids = test_x['PassengerId'].values

# female = 0, male = 1
combi_x['Gender'] = combi_x['Sex'].map({'female':0, 'male':1}).astype(int)

# missing Embarked: just make them embark from most common place
# and then create dummy variables
combi_x.Embarked[combi_x.Embarked.isnull()] = combi_x.Embarked.dropna().mode().values
ports = list(enumerate(np.unique(combi_x['Embarked'])))    
ports_mapping = {name: i for i, name in ports}              
combi_x.Embarked = combi_x.Embarked.map(lambda x: ports_mapping[x]).astype(int)

# missing Age: median of all the people on board
median_age = combi_x['Age'].dropna().median()
combi_x.loc[combi_x.Age.isnull(), 'Age'] = median_age

# FamilySize = SibSp + Parch + 1
combi_x['FamilySize'] = combi_x['SibSp'] + combi_x['Parch'] + 1

# Alone = 1 if travelled alone; 0 otherwise
combi_x.loc[combi_x.FamilySize == 1, 'Alone'] = 1
combi_x.loc[combi_x.FamilySize > 1, 'Alone'] = 0

# missing Fare: median of their respective class
median_fare = np.zeros(3)
for f in range(0,3):                                              
    median_fare[f] = combi_x[combi_x.Pclass == f+1]['Fare'].dropna().median()                                            # loop 0 to 2
    combi_x.loc[(combi_x.Fare.isnull()) & (combi_x.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Name feature engineering: extract title from Name
def find_title(name):
    return re.split('[,.]', name)[1].strip()
    
combi_x['Title'] = combi_x['Name'].map(lambda x: find_title(x))
title_list = np.unique(combi_x.Title) 
  
# replace all titles and create dummies
def replace_title(title):
    if title in ['Capt', 'Col', 'Don', 'Major', 'Master', 'Rev']:
        return 'Sir'
    elif title in ['the Countess', 'Dona', 'Jonkheer']:
        return 'Lady'      
    elif title == 'Mlle':
        return 'Miss'
    elif title == 'Mme':
        return 'Ms'
    else:
        return title
        
combi_x['Title'] = combi_x['Title'].apply(replace_title)
combi_x['Title'] = combi_x['Title'].map({'Lady': 1, 'Miss': 2, 'Ms': 3, 'Mrs': 4, 
                                         'Dr': 5, 'Sir': 6, 'Mr': 7}).astype(int)

# interaction terms betwween Age and Class                 
combi_x['Age*Class'] = combi_x.Age * combi_x.Pclass


# subset useful features for learning algorithm
combi_x = combi_x[['Pclass', 
                   'Age', 
                   'SibSp',
                   'Parch',
                   'Fare',
                   'Embarked',
                   'Gender',
                   'FamilySize',
                   'Alone',
                   'Title',
                   'Age*Class'
                    ]]

# Split the merged data back into training and testing, in array format
train_x = combi_x.values[:len(train_x)]
test_x = combi_x.values[len(train_x):]
train_y = train['Survived']                    


########################  Machine learning  ###################################
def forest_cv():
    n_estimators = [10, 50, 100, 200, 400, 600, 800]
    scores = list()
    scores_std = list()
    
    for n_estimator in n_estimators:
        np.random.seed(0)
        forest = RandomForestClassifier(n_estimators=n_estimator)
        this_score = cross_validation.cross_val_score(
                forest, train_x, train_y, cv=10)
        scores.append(np.mean(this_score))
        scores_std.append(np.std(this_score))
        print 'Model with {0} trees finished'.format(n_estimator)
       
    pl.plot(n_estimators, scores)
    pl.plot(n_estimators, np.array(scores) + np.array(scores_std), 'b--')
    pl.plot(n_estimators, np.array(scores) - np.array(scores_std), 'b--')
    pl.xlabel('Number of trees')
    pl.ylabel('CV scores')
    pl.show()
    print 
    print 'The highest CV socre is {0} with {1} trees'.format(
            max(scores), n_estimators[scores.index(max(scores))]) 


########################  Output results  #####################################
print 'Training...'
clf = GradientBoostingClassifier(n_estimators=200).fit(train_x, train_y)

print 'Predicting...'
output = clf.predict(test_x).astype(int)

predictions_file = open("forest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'