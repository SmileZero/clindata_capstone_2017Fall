# -*- coding: utf-8 -*-
"""
Capstone data analyze

"""
import pandas as pd
#import numpy as np
Faers =pd.read_csv('C:/Users/shaosijie/Desktop/solr_us_all.csv/solr_us_all.csv')
Faers.shape #(3017774, 56)

list(Faers) # have the all column names for Faers

Faers.primaryid.nunique() # 3013737
set(Faers.age_cod)
# {nan, 'DEC', 'DY', 'HR', 'MON', 'WK', 'YR', 'age_cod'}
set(Faers.wt_cod)
#{nan, 'KG', 'LBS', 'wt_cod'}
set(Faers.dose_unit)

type(Faers)


## scale the age column
DY_list = Faers.index[Faers.age_cod == "DY"].tolist()
MON_list = Faers.index[Faers.age_cod == "MON"].tolist()
WK_list = Faers.index[Faers.age_cod == "WK"].tolist()
DEC_list = Faers.index[Faers.age_cod == "DEC"].tolist()
HR_list = Faers.index[Faers.age_cod == "HR"].tolist()

Faers.age[DY_list] = pd.to_numeric(Faers.age[DY_list])/365.0
Faers.age[MON_list] = pd.to_numeric(Faers.age[MON_list])/12.0
Faers.age[WK_list] = pd.to_numeric(Faers.age[WK_list])/52.0
Faers.age[DEC_list] = pd.to_numeric(Faers.age[DEC_list])*10
Faers.age[HR_list] = pd.to_numeric(Faers.age[HR_list])/(365.0*24.0)


## Scale the weight column 
KG_list = Faers.index[Faers.wt_cod == "KG"].tolist()
Faers.wt[KG_list] = pd.to_numeric(Faers.wt[KG_list])*2.2046




