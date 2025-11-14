import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import torch
from scipy.stats import wasserstein_distance
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#This corresponds to the analyst releasing a subset of the columns of the NIST dataset in the clear
def release():
    data = pd.read_csv('insert address here') #Replace the address here accordingly
    datarelease = data[['AGEP', 'DENSITY', 'INDP_CAT', 'INDP', 'SEX', 'EDU', 'HISP', 'RAC1P', 'DREM']]
    datarelease = datarelease.replace('N', '-10')

    return datarelease

#Function that randomly splits the data into two halves, which is used in the demographic coherence framework
def split(dataset):
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    training = dataset.loc[1:13626].reset_index(drop=True)
    test = dataset.loc[13627:27252].reset_index(drop=True)
    return training, test


#Function that processes raw data release to produce a dataset fit for a supervised learing algorithm
def preplabeled(training):
    #Extract the label to be predicted as a separate dataframe
    ytrain = training['DREM']
    training = training.drop(columns=['DREM'], axis=1)

    #Change labels to be -1: existence of a disability, and 1: absence of a disability
    ytrain = ytrain.replace('1','-1')
    ytrain = ytrain.replace('2','1')


    xtrain = training.to_numpy()
    ytrain = ytrain.to_numpy()

    return xtrain, ytrain


####Prepares datasets for evaluation, by defining the arrays
####on which prediction distributions are defined
def evalprep(training, test):

    ytrain = training['DREM']
    ytest = test['DREM']

    training = training.drop(columns=['DREM'], axis=1)
    test = test.drop(columns=['DREM'], axis=1)

    xtrain = training.to_numpy()
    xtest = test.to_numpy()

    ####Extracting training and test set indices corresponding to individuals
    ####with cognitive disability and locating corresponding records

    idx = ytrain.index[ytrain == '1']
    xgrptraining = training.loc[idx]
    xgrptraining = xgrptraining.to_numpy()

    idx_test = ytest.index[ytest == '1']
    xgrptest = test.loc[idx_test]
    xgrptest = xgrptest.to_numpy()

    return xtrain, xtest, xgrptraining, xgrptest


#Processes the data release, and then applies a learning algorithm on it to produce
#a classifer. We use a random forest classifier.
def adversary(training):
    xtrain, ytrain = preplabeled(training)
    clf = RandomForestClassifier(n_estimators=50)
    clf = clf.fit(xtrain, ytrain)
    return clf

###########################################################
####The main script is executed by the data analyst########
####and is an instantiation of our framework###############
###########################################################

###The analyst simply releases a subset of the columns on the clear
datarelease = release()

numiter=100

##Used to track results for cognitive disability subgroup
wasslist = []
#Used to track results for entire dataset
wassfull = []

for i in range(numiter):
    #Splits dataset into two halves
    training, test = split(datarelease)

    #Adversary trains classifier on training data
    clf = adversary(training)

    #Creates datasets for prediction distributions to be defined on- both subgroup and whole dataset
    xtrain, xtest, xgrptraining, xgrptest = evalprep(training, test)

####Create the prediction distributions
    p1 = clf.predict(xgrptraining)
    p2 = clf.predict(xgrptest)
    p3 = clf.predict(xtrain)
    p4 = clf.predict(xtest)


    ############## Graph the prediction distributions (of subgroups) #############
    ############## for one run of this to show how it looks.################
    ############## To run this, set num_iter = 1, and uncomment the below lines ###############

    # p1 = np.sort(p1)
    # p2 = np.sort(p2)
    # fig, axs = plt.subplots(2)
    # fig.subplots_adjust(hspace=0.5)
    # fig.suptitle('Prediction Distributions for Cognitive Disability Subgroup')
    # axs[0].plot(p1, color='blue')
    # axs[0].set_xlabel('Data Points in Category')
    # axs[0].set_ylabel('Prediction Values')
    # axs[0].set_title('Prediction distribution for Training Data')
    # axs[1].plot(p2, color='red')
    # axs[1].set_xlabel('Data Points in Category')
    # axs[1].set_ylabel('Prediction Values')
    # axs[1].set_title('Prediction distribution for Test Data')
    # plt.show()

    ###################################################
    ###################################################
    ###################################################

    #Store wasserstein distance for cognitive disability subgroup
    wassvalgrp = wasserstein_distance(p1,p2)
    wasslist.append(wassvalgrp)

    #Store wasserstein distance for full dataset
    wassval_full = wasserstein_distance(p3,p4)
    wassfull.append(wassval_full)

#####Plot Wasserstein distance vs iteration distributions for subgroup
wasslist = np.sort(wasslist)
plt.plot(wasslist)
plt.xlabel('Iteration')
plt.ylabel('Wasserstein distance')

yticks = np.linspace(0, 2, 20, endpoint=True)
plt.yticks(yticks)
plt.title('Wasserstein Distance Plot on Cognitive Disability Subgroup')
plt.grid()
plt.show()

#####Print average Wasserstein distances
print('The average Wasserstein distance among people with a cognitive disability is', np.average(wasslist))
print('The average Wasserstein distance is the whole dataset is', np.average(wassfull))
