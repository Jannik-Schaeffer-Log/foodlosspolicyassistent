########################################
# Import libraries
########################################
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
#import time


########################################
# Configure Page
########################################
st.set_page_config(page_title="Food Loss Database", page_icon="🥑",layout="wide")


########################################
# Add Title
########################################
st.header("Food Loss & Waste Database by FAO")
st.sidebar.header("Food Loss & Waste Database by FAO")
# col1, col2 = st.columns([1.5,0.5])
# with col1:
#     st.write('"The Food Loss and Waste database is the largest online collection of data on both food loss and food waste and causes reported in scientific journals, academic publications, grey literature, countries among others. The database contains data and information from openly accessible databases, reports and studies measuring food loss and waste across food products, stages of the value chain, and geographical areas. In November 2021, more than 700 publications and reports from various sources (e.g., subnational reports, academic studies, FAOSTAT and reports from national and international organizations such as the World Bank, GIZ, FAO, IFPRI, and other sources), which have produced more than 29 thousand data points, were included. Data can be queried, downloaded, and plotted in an interactive and structured way. The database can be used by anyone who wishes to know more about food losses and waste." Source: https://www.fao.org/platform-food-loss-waste/flw-data/en/')
#     #st.write('Source: https://www.fao.org/platform-food-loss-waste/flw-data/en/')

########################################
# Read Data
########################################
#FAO data
df = pd.read_csv("data\Data.csv")
# Country Categories by Worldbank 
country_categories_tidy= pd.read_csv('data/country_categories_tidy.csv', delimiter=';')
# Country Categories by Worldbank 
commodity_groups_tidy= pd.read_csv('data/commodity_groups.csv', delimiter=';')

########################################
# Create Filters
########################################
years = st.slider(
    'Select years', 
    min_value=min(df['year']), 
    value=(min(df['year']), max(df['year'])), 
     max_value=max(df['year']))
col3,col4 = st.columns(2)
with col3:
    food_supply_stage = st.selectbox(
            "Choose food supply chain stage", ['All']+list(df['food_supply_stage'].unique())
        )

with col4:
    commodity = st.selectbox(
            "Choose commodity", ['All']+list(df['commodity'].unique())
        )

########################################
# Filter Dataset
########################################
if food_supply_stage == 'All':
    if commodity == 'All':
        df_filtered = df[(df['year'] >= years[0]) & (df['year'] <= years[1])]

    else:     
        df_filtered = df[(df['year'] >= years[0]) & (df['year'] <= years[1]) & (df['commodity']==commodity) ]
    
else:     
    if commodity == 'All':
        df_filtered = df[(df['year'] >= years[0]) & (df['year'] <= years[1]) & (df['food_supply_stage']==food_supply_stage)]

    else:     
        df_filtered = df[(df['year'] >= years[0]) & (df['year'] <= years[1]) & (df['food_supply_stage']==food_supply_stage) & (df['commodity']==commodity) ]


########################################
# Prepare Data for Visualisation
########################################
df_map=pd.pivot_table(df_filtered,index='country', values='loss_percentage',aggfunc=np.mean, fill_value=None)
df_map=df_map.reset_index()

########################################
# Create Chloropleth Map Visualisation
########################################
fig = go.Figure(data=go.Choropleth(
    locations=df_map['country'], # Spatial coordinates
    z = df_map['loss_percentage'].astype(float), # Data to be color-coded
    locationmode = 'country names', # set of locations match entries in `locations`
    colorscale = 'tempo',
    colorbar_title = "Food Loss %",
))
fig.update_layout(margin={"r":0,"t":25,"l":0,"b":0})
fig.update_layout(
    title_text = 'Food Loss Percentage',
    title_x=0.5,
    geo_scope='world', 
    template='simple_white',
    

)

st.plotly_chart(fig, use_container_width=True)


########################################
#  Create Line chart  Visualisation
########################################

# prepare FAO Data for merge
df_FAO_pivot=pd.pivot_table(df ,index=['country','year','commodity','food_supply_stage'], values='loss_percentage',aggfunc=np.mean, fill_value=None)
df_FAO_pivot=df_FAO_pivot.reset_index()
# df_FAO_pivot['year']=pd.to_datetime(df_FAO_pivot['year'],format='%Y')

## Add Country Categories to the Dataframe
df_merged=pd.merge(df_FAO_pivot,country_categories_tidy, how='inner', left_on='country', right_on='Country')
df_merged=df_merged.drop(columns=['Country'])
df_merged=df_merged.ffill(axis = 0)

## Add Commodity Categories to the Dataframe
df_merged=pd.merge(df_merged,commodity_groups_tidy, how='inner', left_on='commodity', right_on='Commodities')
df_merged=df_merged.drop(columns=['Commodities'])
df_merged=df_merged.ffill(axis = 0)

if food_supply_stage == 'All':
    if commodity == 'All':
        df_merged_filtered = df_merged[(df_merged['year'] >= years[0]) & (df_merged['year'] <= years[1])]

    else:     
        df_merged_filtered = df_merged[(df_merged['year'] >= years[0]) & (df_merged['year'] <= years[1]) & (df_merged['commodity']==commodity) ]
    
else:     
    if commodity == 'All':
        df_merged_filtered = df_merged[(df_merged['year'] >= years[0]) & (df_merged['year'] <= years[1]) & (df_merged['food_supply_stage']==food_supply_stage)]

    else:     
        df_merged_filtered = df_merged[(df_merged['year'] >= years[0]) & (df_merged['year'] <= years[1]) & (df_merged['food_supply_stage']==food_supply_stage) & (df_merged['commodity']==commodity) ]



df_line_chart=df_merged_filtered.groupby('year').agg('mean')
#df_line_chart['year']=pd.to_datetime(df_line_chart['year'],format='%Y')
# Create figure
fig1 = go.Figure()

fig1.add_trace(
    go.Scatter(x=list(df_line_chart.index), y=list(df_line_chart['loss_percentage'])))

# Set title
fig1.update_layout(
    title_text="Mean Food Loss Percentage",
    template='seaborn',
    width=300, height=400
)

# Add range slider
fig1.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)
st.plotly_chart(fig1, use_container_width=True)

with st.expander('See selected Data'):
    st.dataframe(df_merged_filtered)
with st.expander('See Data Profil'):

    # my_bar = st.progress(0)
    # for percent_complete in range(100):
    #     time.sleep(0.1)
    # my_bar.progress(percent_complete + 1)
    if st.button('Start Profiling'):
        pr = df.profile_report()
        st_profile_report(pr)
