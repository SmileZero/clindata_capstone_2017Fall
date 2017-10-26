
# coding: utf-8

# In[13]:

import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[2]:

#read in the data
data = pd.read_csv("solr_us_10k.csv") 


# In[3]:

#get a hint of how data looks
col_names = data.columns.values 
print col_names
#print data.describe()
data.head()


# In[4]:

#get the column that needs to be encoded
gender = data['sex']
reactions = data['reactions']
indications = data['indications']


# In[5]:

def hot_pot_encoding(values):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    return label_encoder,onehot_encoded

def reverse_encoding(label_encoder,onehot_encoded):
    # invert first example
    #inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded)])
    print(inverted)
    return inverted


# In[6]:

#gender hotpot encoding
gender_values = array(gender)
print(gender_values)
gender_label_encoder,gender_onehot_encoded = hot_pot_encoding(gender_values)
#gender_inverted = reverse_encoding(gender_label_encoder,gender_onehot_encoded)

#reaction hotpot encoding
reactions_values = array(reactions)
print(reactions_values)
reactions_label_encoder,reactions_onehot_encoded = hot_pot_encoding(reactions_values)
#reactions_inverted = reverse_encoding(reactions_label_encoder,reactions_onehot_encoded)

#reaction hotpot encoding
indications_values = array(indications)
print(indications_values)
indications_label_encoder,indications_onehot_encoded = hot_pot_encoding(indications_values)
#indications_inverted = reverse_encoding(indications_label_encoder,indications_onehot_encoded)


# In[10]:

def split_coma(s):
    return s.split('\\,')
reactions_splited = map(split_coma, reactions)
reactions_splited[:2]


# In[28]:

reaction_flat_list = [item for sublist in reactions_splited for item in sublist]
#reactions_flat = np.asarray(reactions_splited).flatten('F')
#print reaction_flat_list
reaction_set = set(reaction_flat_list)
distinct_reaction = list(reaction_set)


# In[29]:

print len(distinct_reaction)


# In[33]:

def one_hot_encoding(data, distinct_val):
    encoding_list = []
    for line in data:
        line_list = [0 for i in xrange(len(distinct_val))]
        for item in line:
            idx = distinct_val.index(item)
            line_list[idx] = 1
        encoding_list.append(line_list)
    return encoding_list


# In[31]:

def get_distinct_value(splited_list):
    flat_list = [item for sublist in splited_list for item in sublist]
    flat_set = set(flat_list)
    return list(flat_set)


# In[37]:

test_reaction_split = reactions_splited[:10]
test_distinct_reactions = get_distinct_value(test_reaction_split)
print test_distinct_reactions
test_encoding_list = one_hot_encoding(test_reaction_split, test_distinct_reactions)
test_encoding_list


# In[38]:

test_reaction_split


# In[ ]:



