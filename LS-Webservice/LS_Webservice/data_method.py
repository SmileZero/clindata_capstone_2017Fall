import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from keras.models import load_model

# def read_dict():
# load the unique_dict_categorical file for unique values for the feature matrix
file_name = open("unique_categorical_dict.pkl", "r")
unique_categorical_dict = pickle.load(file_name)
file_name.close()

# remove all nan within the dictionary
ind = pd.isnull(unique_categorical_dict['dechal'])
unique_categorical_dict['dechal'] = unique_categorical_dict['dechal'][~ind]
ind = pd.isnull(unique_categorical_dict['age_grp'])
unique_categorical_dict['age_grp'] = unique_categorical_dict['age_grp'][~ind]
ind = pd.isnull(unique_categorical_dict['indications'])
unique_categorical_dict['indications'] = unique_categorical_dict['indications'][~ind]
ind = pd.isnull(unique_categorical_dict['rechal'])
unique_categorical_dict['rechal'] = unique_categorical_dict['rechal'][~ind]
ind = pd.isnull(unique_categorical_dict['sex'])
unique_categorical_dict['sex'] = unique_categorical_dict['sex'][~ind]

#Create function to convert to dummy variables
def convert_to_dummies(data, categorical_columns, unique_dictionary):
    #Assign the categories for each column
    for column in categorical_columns:
        data[column] = data[column].astype('category', categories=unique_dictionary[column])
    #Convert to categorical
    dummy = pd.get_dummies(data, columns=categorical_columns)
    return dummy

# required columns for the dataset:
def data_process(data_json):
    categorical_columns = ['RxCUI', 'indications', 'role_cod', 'sex', 'age_grp',
                       'dechal', 'rechal']
    new_df = pd.DataFrame(columns=['RxCUI','indications','role_cod','sex','age','age_grp','wt','dechal','rechal'])
    durg_num = len(data_json['rxcuis'])
    for i in range(durg_num):
        new_df.loc[i] = [data_json['rxcuis'][i],data_json['indications'][i].upper(),data_json['reported_roles'][i],data_json['gender'],\
                        data_json['age'],data_json['age_group'],data_json['weight'],data_json['dechals'][i],data_json['rechals'][i]]
    new_df_dummies = convert_to_dummies(new_df, categorical_columns, unique_categorical_dict)
    return (new_df_dummies)

# Correct shape of test Patient
def correctInputShape(data_x):
    size = data_x.shape[0]
    num_features = data_x[0].shape[1]
    num_drugs = data_x[0].shape[0]
    x_shaped = np.empty((size, num_drugs, num_features))
    for i in range(size):
        x_shaped[i] = np.nan_to_num(data_x[i].todense())
    return x_shaped

def run_model(json_file):
    patient_feature = data_process(json_file)
    model = load_model('Clindata_LSTM_Test.h5')
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