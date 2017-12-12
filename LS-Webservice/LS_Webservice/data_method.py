import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import math

# load the unique_dict_categorical file for unique values for the feature matrix
file_name = open("unique_categorical_dict.pkl", "r")
unique_categorical_dict = pickle.load(file_name)
file_name.close()

# remove all nan within the dictionary 
ind = pd.isnull(unique_categorical_dict['dechal'])
unique_categorical_dict['dechal'][ind] = "nan"
ind = pd.isnull(unique_categorical_dict['age_grp'])
unique_categorical_dict['age_grp'][ind] = "nan"
ind = pd.isnull(unique_categorical_dict['indications'])
unique_categorical_dict['indications'][ind] = "nan"
ind = pd.isnull(unique_categorical_dict['rechal'])
unique_categorical_dict['rechal'][ind] = "nan"
ind = pd.isnull(unique_categorical_dict['sex'])
unique_categorical_dict['sex'][ind] = "nan"

#Create function to convert to dummy variables
def convert_to_dummies(data, categorical_columns, unique_dictionary):
    #Assign the categories for each column
    for column in categorical_columns:
        data[column] = data[column].astype('category', categories=unique_dictionary[column])
    #Convert to categorical
    dummy = pd.get_dummies(data, columns=categorical_columns)
    return dummy

def calculate_sigmod(x, k, x0):
    y = 1 / (1 + math.exp(-k * (x - x0)))
    return y

def calculate_sigmod(x, k, x0):
    y = 1 / (1 + math.exp(-k * (x - x0)))
    return y

# required columns for the dataset:
def data_process(data_json):
    categorical_columns = ['RxCUI', 'indications', 'role_cod', 'sex', 'age_grp', 
                       'dechal', 'rechal']    
    new_df = pd.DataFrame(columns=['RxCUI','indications','role_cod','sex','age','age_grp','wt','dechal','rechal'])
    drug_num = len(data_json['rxcuis'])
    for i in range(drug_num):
        if data_json['indications'][i] == '':
            data_json['indications'][i] = "nan"
        if data_json['dechals'][i] == '':
            data_json['dechals'][i] = "nan"
        if data_json['rechals'][i] == '':
            data_json['rechals'][i] = "nan"
        if data_json['gender'] == '':
            data_json['gender'] = "nan"
        if data_json['age_group'] == '':
            data_json['age_group'] = "nan"
            
        # scale the weight and age
        data_json['weight'] = calculate_sigmod(float(data_json['weight']), 10 ** -1.5, 170)
        data_json['age'] = calculate_sigmod(float(data_json['age']), 0.1, 50)

        new_df.loc[i] = [data_json['rxcuis'][i],data_json['indications'][i],data_json['reported_roles'][i],data_json['gender'],\
                        data_json['age'],data_json['age_group'],data_json['weight'],data_json['dechals'][i],data_json['rechals'][i]]
    new_df_dummies = convert_to_dummies(new_df, categorical_columns, unique_categorical_dict)
    return (new_df_dummies)
  
# Correct shape of test Patient
def correctInputShape(data_x):
    size = 1
    num_features = data_x.shape[1]
    num_drugs = data_x.shape[0]
    x_shaped = np.empty((size, num_drugs, num_features))
    for i in range(size):
        x_shaped[i] = np.nan_to_num(data_x)
    return x_shaped

def run_model(json_file, model):
    patient_feature = data_process(json_file)
    Patient_info = correctInputShape(patient_feature)
    predicted_proba = model.predict(Patient_info) #Predict probabilities
    last_pred = predicted_proba[len(predicted_proba)-1][0] #Extract probabilities from the last drug
    inds = np.argsort(-last_pred)[:10] # Extract top 10 indices
    # Read reaction names (this file is on the Github/Data Cleaning folder)
    reactions = pd.read_csv('Top2000Reactions.csv')
    reacts = [reactions['Reaction'][i] for i in inds]     # Extract top 10 reactions
    probs = [last_pred[i] for i in inds]# Extract top 10 probabilities
    # put everything together in a dataframe
    prediction_df = pd.DataFrame()
    prediction_df['Reaction'] = reacts
    prediction_df['Probability'] = probs
    prediction_df_json = prediction_df.to_json(orient='split')
    return (prediction_df_json)

def get_rxcui():
    return unique_categorical_dict['RxCUI'].tolist()

def get_indication():
    return unique_categorical_dict['indications'].tolist()