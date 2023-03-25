
import pandas as pd
import pandas_profiling
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(layout="wide")

@st.cache
def get_data():
    data_FAO = pd.read_csv('data\Data.csv')
    return data_FAO

df=get_data()
pr = df.profile_report()
st_profile_report(pr) 