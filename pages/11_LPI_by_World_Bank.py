########################################
# Import libraries
########################################
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

########################################
# Configure Page
########################################
st.set_page_config(page_title="Test", page_icon="🥑",layout="wide")

########################################
# Add Title
########################################
st.header("Logistic Performance Index by World Bank")
st.sidebar.header("Logistic Performance Index by World Bank")

########################################
# Read Data
########################################
df = pd.read_csv("data\LPI_Data.csv", encoding='latin-1')

########################################
# Create Filters
########################################
years = st.select_slider(
    'Select years', 
    options=['2007', '2010', '2012', '2014', '2016', '2018'],
    value=('2007', '2018')
    )

col1, col2= st.columns([1,6])

with col1:
    series = st.radio(
        "Select score or rank.",
        ('score', 'rank'), horizontal= True)

# Create List for series_name selectbox
series_list_score=[
    'Logistics performance index: Overall score (1=low to 5=high)',
    'Ability to track and trace consignments, score (1=low to 5=high)',
    'Competence and quality of logistics services, score (1=low to 5=high)',
    'Ease of arranging competitively priced international shipments, score (1=low to 5=high)',
    'Efficiency of the clearance process, score (1=low to 5=high)',
    'Frequency with which shipments reach consignee within scheduled or expected time, score (1=low to 5=high)',
    'Quality of trade- and transport-related infrastructure, score (1=low to 5=high)'
]
series_list_rank=[
    'Logistics performance index: Overall rank (1=highest performance)',
    'Ability to track and trace consignments, rank (1=highest performer)',
    'Competence and quality of logistics services, rank (1=highest performer)',
    'Ease of arranging competitively priced international shipments, rank (1=highest performer)',
    'Efficiency of the clearance process, rank (1=highest performer)',
    'Frequency with which shipments reach consignee within scheduled or expected time, rank (1=highest performer)',
    'Quality- of trade and transport-related infrastructure, rank (1=highest performer)'
]
if series =='score':
    series_list=series_list_score
else:
    series_list=series_list_rank

with col2:
    series_name = st.selectbox(
            "Choose overall score or category", list(series_list)
        )

########################################
# Clean Data 
# Change Data Types
# Filter for rank or score
########################################
df=df.replace(to_replace='..', value=None)
df_series=df[df['Series Name'].str.contains(series,na=False)]
df_series=df_series[df_series['Series Name']==series_name]
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
df_series = df_series.astype(convert_dtypes_dict)

########################################
# Filter Dataset
# Build Mean for selected timeframe
########################################
year_dict= {
    '2007':'2007 [YR2007]',
    '2010':'2010 [YR2010]',
    '2012':'2012 [YR2012]',
    '2014':'2014 [YR2014]',
    '2016':'2016 [YR2016]',
    '2018':'2018 [YR2018]'
}

df_series['mean']=df_series.loc[:,year_dict[years[0]]:year_dict[years[1]]].mean(axis=1)

########################################
# Prepare Data for Visualisation
########################################
df_map=df_series[['Country Name','mean']]

########################################
# Create Visualisation
########################################
fig = go.Figure(data=go.Choropleth(
    locations=df_map['Country Name'], # Spatial coordinates
    z = df_map['mean'].astype(float), # Data to be color-coded
    locationmode = 'country names', # set of locations match entries in `locations`
    colorscale = 'tempo',
    colorbar_title = series,
))
fig.update_layout(margin={"r":0,"t":25,"l":0,"b":0})
fig.update_layout(
    title_text = series_name,
    title_x=0.5,
    geo_scope='world', 
    template='simple_white',
)


########################################
# Show Visualisation
########################################
st.plotly_chart(fig, use_container_width=True)


with st.expander('See Data Profil'):

    # my_bar = st.progress(0)
    # for percent_complete in range(100):
    #     time.sleep(0.1)
    # my_bar.progress(percent_complete + 1)
    if st.button('Start Profiling'):
        pr = df.profile_report()
        st_profile_report(pr)