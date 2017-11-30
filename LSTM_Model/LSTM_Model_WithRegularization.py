
# coding: utf-8

# In[1]:


import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.metrics import categorical_accuracy, binary_accuracy
from keras.models import load_model
from keras.regularizers import l1_l2
import pandas as pd

import pickle


# In[2]:


##Read X

# f = open("pickle_files/group1_x_sparse.pkl", 'r')
# group1_x = pickle.load(f)
# f.close()

# f = open("pickle_files/group2_x_sparse.pkl", 'r')
# group2_x = pickle.load(f)
# f.close()

f = open("pickle_files/group3_x_sparse.pkl", 'r')
group3_x = pickle.load(f)
f.close()

f = open("pickle_files/group4_x_sparse.pkl", 'r')
group4_x = pickle.load(f)
f.close()

f = open("pickle_files/group5_x_sparse.pkl", 'r')
group5_x = pickle.load(f)
f.close()

f = open("pickle_files/group6_x_sparse.pkl", 'r')
group6_x = pickle.load(f)
f.close()

f = open("pickle_files/group7_x_sparse.pkl", 'r')
group7_x = pickle.load(f)
f.close()

f = open("pickle_files/group8_x_sparse.pkl", 'r')
group8_x = pickle.load(f)
f.close()

f = open("pickle_files/group9_x_sparse.pkl", 'r')
group9_x = pickle.load(f)
f.close()

f = open("pickle_files/group10_x_sparse.pkl", 'r')
group10_x = pickle.load(f)
f.close()


# In[3]:


##Read Y

# f = open("pickle_files/group1_y_sparse.pkl", 'r')
# group1_y = pickle.load(f)
# f.close()

# f = open("pickle_files/group2_y_sparse.pkl", 'r')
# group2_y = pickle.load(f)
# f.close()

f = open("pickle_files/group3_y_sparse.pkl", 'r')
group3_y = pickle.load(f)
f.close()

f = open("pickle_files/group4_y_sparse.pkl", 'r')
group4_y = pickle.load(f)
f.close()

f = open("pickle_files/group5_y_sparse.pkl", 'r')
group5_y = pickle.load(f)
f.close()

f = open("pickle_files/group6_y_sparse.pkl", 'r')
group6_y = pickle.load(f)
f.close()

f = open("pickle_files/group7_y_sparse.pkl", 'r')
group7_y = pickle.load(f)
f.close()

f = open("pickle_files/group8_y_sparse.pkl", 'r')
group8_y = pickle.load(f)
f.close()

f = open("pickle_files/group9_y_sparse.pkl", 'r')
group9_y = pickle.load(f)
f.close()

f = open("pickle_files/group10_y_sparse.pkl", 'r')
group10_y = pickle.load(f)
f.close()


# In[4]:


def split_train_test_validation(data_X, data_Y, train_perc = 0.6, valid_perc = 0.1, test_perc = 0.3):
    #Get the total number of rows
    nrow = len(data_X)
    n_train = int(nrow*train_perc)
    n_valid = int(nrow*valid_perc)
    n_test  = int(nrow*test_perc)
    
    #Shuffle the dataset
    new_order = np.random.choice(nrow, nrow, replace=False)

    #Convert dataset into numpy array for multi-indexing
    x_array = np.array(data_X)
    y_array = np.array(data_Y)

    #The first n_train are the training set
    x_train = x_array[new_order[0:n_train]]
    y_train = y_array[new_order[0:n_train]]
    
    #The following n_valid are the validation set
    x_valid = x_array[new_order[(n_train):(n_train+n_valid+1)]]
    y_valid = y_array[new_order[(n_train):(n_train+n_valid+1)]]
    
    #The following n_test are the test set
    x_test = x_array[new_order[(n_train+n_valid+1):(n_train+n_valid+n_test+2)]]
    y_test = y_array[new_order[(n_train+n_valid+1):(n_train+n_valid+n_test+2)]]
    
    #Return arrays
    return x_train, y_train, x_valid, y_valid, x_test, y_test


# In[5]:


#Split sets for each group

# #Group 1
# group1_x_train, group1_y_train, group1_x_valid, group1_y_valid, group1_x_test, group1_y_test = split_train_test_validation(group1_x, group1_y)


# #Group 2
# group2_x_train, group2_y_train, group2_x_valid, group2_y_valid, group2_x_test, group2_y_test = split_train_test_validation(group2_x, group2_y)


#Group 3
group3_x_train, group3_y_train, group3_x_valid, group3_y_valid, group3_x_test, group3_y_test = split_train_test_validation(group3_x, group3_y)

#Group 4
group4_x_train, group4_y_train, group4_x_valid, group4_y_valid, group4_x_test, group4_y_test = split_train_test_validation(group4_x, group4_y)


#Group 5
group5_x_train, group5_y_train, group5_x_valid, group5_y_valid, group5_x_test, group5_y_test = split_train_test_validation(group5_x, group5_y)


#Group 6
group6_x_train, group6_y_train, group6_x_valid, group6_y_valid, group6_x_test, group6_y_test = split_train_test_validation(group6_x, group6_y)


#Group 7
group7_x_train, group7_y_train, group7_x_valid, group7_y_valid, group7_x_test, group7_y_test = split_train_test_validation(group7_x, group7_y)


#Group 8
group8_x_train, group8_y_train, group8_x_valid, group8_y_valid, group8_x_test, group8_y_test = split_train_test_validation(group8_x, group8_y)


#Group 9
group9_x_train, group9_y_train, group9_x_valid, group9_y_valid, group9_x_test, group9_y_test = split_train_test_validation(group9_x, group9_y)


#Group 10
group10_x_train, group10_y_train, group10_x_valid, group10_y_valid, group10_x_test, group10_y_test = split_train_test_validation(group10_x, group10_y)


# In[6]:


def chooseGroup_x(number):
    if number == 1:
        return group1_x_train
    elif number == 2:
        return group2_x_train
    elif number == 3:
        return group3_x_train
    elif number == 4:
        return group4_x_train
    elif number == 5:
        return group5_x_train
    elif number == 6:
        return group6_x_train
    elif number == 7:
        return group7_x_train
    elif number == 8:
        return group8_x_train
    elif number == 9:
        return group9_x_train
    elif number == 10:
        return group10_x_train
    
def chooseGroup_y(number):
    if number == 1:
        return group1_y_train
    elif number == 2:
        return group2_y_train
    elif number == 3:
        return group3_y_train
    elif number == 4:
        return group4_y_train
    elif number == 5:
        return group5_y_train
    elif number == 6:
        return group6_y_train
    elif number == 7:
        return group7_y_train
    elif number == 8:
        return group8_y_train
    elif number == 9:
        return group9_y_train
    elif number == 10:
        return group10_y_train


# In[7]:


#Create probabilities of each group, so each group is randomly selected proportionally to their number of items
total_train = len(group3_x_train) + len(group4_x_train) + len(group5_x_train) + len(group6_x_train) + len(group7_x_train) + len(group8_x_train) + len(group9_x_train) + len(group10_x_train)
p = []
p.append(len(group3_x_train)/float(total_train))
p.append(len(group4_x_train)/float(total_train))
p.append(len(group5_x_train)/float(total_train))
p.append(len(group6_x_train)/float(total_train))
p.append(len(group7_x_train)/float(total_train))
p.append(len(group8_x_train)/float(total_train))
p.append(len(group9_x_train)/float(total_train))
p.append(len(group10_x_train)/float(total_train))



def correctInputShape(data_x):
    size = data_x.shape[0]
    num_features = data_x[0].shape[1]
    num_drugs = data_x[0].shape[0]
    
    x_shaped = np.empty((size, num_drugs, num_features))
    
    for i in range(size):
        x_shaped[i] = np.nan_to_num(data_x[i].todense())
    
    return x_shaped

def getRecall(pred, true):
    recall = len(pred.intersection(true))/float(len(true))
    return recall

def getPrecision(pred,true):
    precision = len(pred.intersection(true))/float(len(pred))
    return precision

def getFBetaScore(precision, recall, beta=1):
    if (precision ==0) & (recall == 0):
        return 0
    else:
        return (1+ beta**2)*precision*recall/((beta**2 * precision) + recall)


def top10prediction(y_true, y_pred, groupNumber):
    #put both arrays in the same shape, flatten
    pred = y_pred.reshape(-1, y_pred.shape[-1])
    true_flat = np.asarray(map(lambda x: np.asarray(x.todense()),y_true))
    true = true_flat.reshape(-1, y_pred.shape[-1])
    
    scores = []
    for i in range(len(pred)):
        p = pred[i]
        t = true[i]
        
        #Extract top values
        actual_ones = set(np.where(t==1)[0])
        top10_pred = set(np.argsort(p)[-10:])
        
        
        if len(actual_ones)>0:
            recall = getRecall(top10_pred, actual_ones)
            precision = getPrecision(top10_pred, actual_ones)
            F1Score = getFBetaScore(precision, recall)

            scores.append([recall, precision, F1Score, groupNumber])
    
    scores_df = pd.DataFrame(scores)
    scores_df.columns = ['Recall', 'Precision', 'F1Score', 'GroupNumber']
    return scores_df

def allGroupsPredictions(model):
#     group1_x_valid_shape = correctInputShape(group1_x_valid)
#     group2_x_valid_shape = correctInputShape(group2_x_valid)
    group3_x_valid_shape = correctInputShape(group3_x_valid)
    group4_x_valid_shape = correctInputShape(group4_x_valid)
    group5_x_valid_shape = correctInputShape(group5_x_valid)
    group6_x_valid_shape = correctInputShape(group6_x_valid)
    group7_x_valid_shape = correctInputShape(group7_x_valid)
    group8_x_valid_shape = correctInputShape(group8_x_valid)
    group9_x_valid_shape = correctInputShape(group9_x_valid)
    group10_x_valid_shape = correctInputShape(group10_x_valid)
    
    #Predict
#     group1_predicted = model.predict(group1_x_valid_shape)
#     group2_predicted = model.predict(group2_x_valid_shape)
    group3_predicted = model.predict(group3_x_valid_shape)
    group4_predicted = model.predict(group4_x_valid_shape)
    group5_predicted = model.predict(group5_x_valid_shape)
    group6_predicted = model.predict(group6_x_valid_shape)
    group7_predicted = model.predict(group7_x_valid_shape)
    group8_predicted = model.predict(group8_x_valid_shape)
    group9_predicted = model.predict(group9_x_valid_shape)
    group10_predicted = model.predict(group10_x_valid_shape)
    
    
    #Measure predictions
    # group1Pred = top10prediction(group1_y_valid, group1_predicted)
    # group2Pred = top10prediction(group2_y_valid, group2_predicted)
    group3Pred = top10prediction(group3_y_valid, group3_predicted, 'Group3')
    group4Pred = top10prediction(group4_y_valid, group4_predicted, 'Group4')
    group5Pred = top10prediction(group5_y_valid, group5_predicted, 'Group5')
    group6Pred = top10prediction(group6_y_valid, group6_predicted, 'Group6')
    group7Pred = top10prediction(group7_y_valid, group7_predicted, 'Group7')
    group8Pred = top10prediction(group8_y_valid, group8_predicted, 'Group8')
    group9Pred = top10prediction(group9_y_valid, group9_predicted, 'Group9')
    group10Pred = top10prediction(group10_y_valid, group10_predicted, 'Group10')
    
    groupList = pd.DataFrame()
    groupList = groupList.append(group3Pred)
    groupList = groupList.append(group4Pred)
    groupList = groupList.append(group5Pred)
    groupList = groupList.append(group6Pred)
    groupList = groupList.append(group7Pred)
    groupList = groupList.append(group8Pred)
    groupList = groupList.append(group9Pred)
    groupList = groupList.append(group10Pred)
    
    #Average
    F1Score = groupList['F1Score'].mean()
    Recall = groupList['Recall'].mean()
    Precision = groupList['Precision'].mean()
    
    return F1Score, Recall, Precision



numFeatures = group10_x_train[0].shape[1]
numOutput = group10_y_train[0].shape[1]

def generator(batch_size):
    
    while True: #Infinite Loop
    
        #Randomly select a group
        groupNumber = np.random.choice(range(3,11), p=p)

        #Extract group
        group_x = chooseGroup_x(groupNumber)
        group_y = chooseGroup_y(groupNumber)

        # Create empty arrays to contain batch of features and labels
        batch_features = np.empty((batch_size, groupNumber, numFeatures))
        batch_labels = np.empty((batch_size, groupNumber, numOutput))
        
        #Generate random index
        batch_indices = np.random.randint(0,len(group_x)-1, size=batch_size)
        i = 0
        for index in batch_indices:
            #np.nan_to_num transforms nan to 0. Without removing nan, the loss function is nan
            batch_features[i] = np.nan_to_num(group_x[index].todense())
            #batch_features[i] = np.nan_to_num(group[index])          
            #Randomly generated output
            batch_labels[i] = group_y[index].todense()
            i += 1
            
        yield batch_features, batch_labels


# In[ ]:


#Define Data Parameters
batch_size = 128

#Model Parameters
epochs = 10
steps_per_epochs = total_train/batch_size
#steps_per_epochs = 1
loss_function = "binary_crossentropy"

#Regularization parameters
l1_par = [10 ** i for i in np.arange(-2.0,2,1)]
l2_par = [10 ** i for i in np.arange(-2.0,2,1)]

#Append 0 to par
l1_par.append(0)
l2_par.append(0)

#Initialize Dictionary to store results
f1scores = []
bestF1 = -1
bestModel = None

#Grid Search
for l1 in l1_par:
    for l2 in l2_par:
        l1l2key = str(l1) + "-" + str(l2)
        print(l1l2key)
        model = Sequential()
        print(" Sequential")
        model.add(LSTM(numOutput, input_shape= (None,numFeatures), return_sequences=True, activation='sigmoid', recurrent_activation='sigmoid', kernel_regularizer=l1_l2(l1,l2)))
        # model.add(LSTM(output_size, input_shape= (None,number_features), return_sequences=True, activation='sigmoid', recurrent_activation='sigmoid'))
        print(" LSTM")
        model.add(Dense(numOutput, activation = 'sigmoid', kernel_regularizer=l1_l2(l1,l2)))
        print(" Dense")
        model.compile(loss=loss_function, optimizer='adam', metrics=[categorical_accuracy, binary_accuracy])
        print(" compile")
        model.fit_generator(generator=generator(batch_size), steps_per_epoch=steps_per_epochs, epochs=epochs, verbose=2)

        F1Score, Recall, Precision = allGroupsPredictions(model)
        print(" F1Score:" + str(F1Score) + ", Precision:" + str(Precision) + ", Recall: " + str(Recall))
        f1scores.append([l1, l2, F1Score, Precision, Recall])

        if F1Score>bestF1:
            bestF1 = F1Score
            bestModel = model

print(bestF1)

#Save average F1 Scores for all combinations
f1_df = pd.DataFrame(f1scores)
f1_df.columns = ['L1', 'L2', 'F1Score', 'Precision', 'Recall']
f1_df.to_csv('GridSearch_F1Scores.csv')


bestModel.save('Clindata_LSTM_Regularization.h5')
