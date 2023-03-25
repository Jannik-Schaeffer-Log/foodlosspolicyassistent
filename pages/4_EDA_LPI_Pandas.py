########################################
# Import libraries
########################################
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

########################################
# Configure Page
########################################
st.set_page_config(layout="wide")

########################################
# Read Data
########################################
@st.cache
def get_data():
    data_LPI = pd.read_csv("data\LPI_Data.csv", encoding='latin-1')
    return data_LPI

df_LPI= get_data()

########################################
# Clean Data 
# Cange Data Types
########################################
df_LPI=df_LPI.replace(to_replace='..', value=None)
df_LPI_score=df_LPI[df_LPI['Series Name'].str.contains("score",na=False)]
convert_dtypes_dict={
    'Country Name':object,
    'Country Code':object,
    'Series Name':object,
    'Series Code':object,
    '2007 [YR2007]':float,
    '2010 [YR2010]':float,
    '2012 [YR2012]':float,
    '2014 [YR2014]':float,
    '2016 [YR2016]':float,
    '2018 [YR2018]':float
    }
df_LPI_score = df_LPI_score.astype(convert_dtypes_dict)

########################################
# Create EDA Report
########################################
pr=df_LPI_score.profile_report()
st_profile_report(pr)