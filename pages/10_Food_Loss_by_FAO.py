########################################
# Import libraries
########################################
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import matplotlib.pyplot as plt

########################################
# Configure Page
########################################
st.set_page_config(page_title="Food Loss Database", page_icon="🥑",layout="wide")


########################################
# Add Title
########################################
st.header("Food Loss & Waste Database by FAO")
st.sidebar.header("Food Loss & Waste Database by FAO")

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
# Drop duplicates
########################################
#drop duplicates
df=df.drop_duplicates()

########################################
# Clean Country names 
########################################
#change country names
df["country"]=df["country"].replace(to_replace="United Republic of Tanzania",value=str("Tanzania"))
df["country"]=df["country"].replace(to_replace='Democratic Republic of the Congo',value=str('Congo, Dem. Rep.'))
df["country"]=df["country"].replace(to_replace='Iran (Islamic Republic of)',value=str('Iran'))
df["country"]=df["country"].replace(to_replace='Bolivia (Plurinational State of)',value=str('Bolivia'))
df["country"]=df["country"].replace(to_replace='Viet Nam',value=str('Vietnam'))
df["country"]=df["country"].replace(to_replace='China,Taiwan',value=str('Taiwan'))
df["country"]=df["country"].replace(to_replace="Democratic People's Republic of Korea",value=str("Korea, South"))
df["country"]=df["country"].replace(to_replace='Republic of Korea',value=str("Korea, North"))
df["country"]=df["country"].replace(to_replace="Lao People's Democratic Republic",value=str('Laos'))
df["country"]=df["country"].replace(to_replace="Venezuela (Bolivarian Republic of)",value=str('Venezuela'))
df["country"]=df["country"].replace(to_replace='The former Yugoslav Republic of Macedonia',value=str('North Macedonia'))

########################################
# Drop unwanted Region data
########################################
df=df[df["country"]!='Australia and New Zealand']
df=df[df["country"]!='Africa']
df=df[df["country"]!='Europe']
df=df[df["country"]!='Latin America and the Caribbean']
df=df[df["country"]!='Northern Africa']
df=df[df["country"]!='Northern America']
df=df[df["country"]!='South-Eastern Asia']
df=df[df["country"]!='Southern Asia']
df=df[df["country"]!='Sub-Saharan Africa']
df=df[df["country"]!='Sub-Saharan Africa']
df=df[df["country"]!='United Kingdom of Great Britain and Northern Ireland']
df=df[df["country"]!='Western Africa']
df=df[df["country"]!='Western Asia']
########################################
# Merge data
########################################
# Add Country Categories to the Dataframe
df_merged=pd.merge(df,country_categories_tidy, how='inner', left_on='country', right_on='Country')
df_merged=df_merged.drop(columns=['Country'])
df_merged=df_merged.ffill(axis = 0)
# Add Commodity Categories to the Dataframe
df_merged=pd.merge(df_merged,commodity_groups_tidy, how='inner', left_on='commodity', right_on='Commodities')
df_merged=df_merged.drop(columns=['Commodities'])
df_merged=df_merged.ffill(axis = 0)


########################################
# Create Filters
########################################
st.write("---")
st.write('Select filter')

years = st.slider(
    'Select years', 
    min_value=min(df['year']), 
    value=(min(df['year']), max(df['year'])), 
     max_value=max(df['year']))
col1, col2, col3,col4 = st.columns(4)
with col1:
    country = st.selectbox(
            "Choose country", ['All']+list(df['country'].unique())
        )
    
with col2:
    food_supply_stage = st.selectbox(
            "Choose food supply chain stage", ['All']+list(df['food_supply_stage'].unique())
        )

with col3:
    commodity = st.selectbox(
            "Choose commodity", ['All']+list(df['commodity'].unique())
        )

with col4:
    method_data_collection = st.selectbox(
            "Choose method of data collection", ['All']+list(df['method_data_collection'].unique())
        )   
    
colA,colB,colC = st.columns(3)

with colA:
    Region = st.selectbox(
            "Choose region", ['All']+list(df_merged['Region'].unique())
        )   
with colB:
    Commodity_Groups = st.selectbox(
            "Choose Commodity Group", ['All']+list(df_merged['Commodity Groups'].unique())
        )    
with colC:
    Income = st.selectbox(
            "Choose Income Group", ['All']+list(df_merged['Income'].unique())
        ) 
st.write("---")


########################################
# Filter Dataset
########################################

#filter years
df_filtered = df_merged[(df_merged['year'] >= years[0]) & (df_merged['year'] <= years[1])]
#filter row 1
if country != 'All':
    df_filtered=df_filtered[df_filtered['country']==country]
if food_supply_stage != 'All':
    df_filtered=df_filtered[df_filtered['food_supply_stage']==food_supply_stage]
if commodity != 'All':
    df_filtered=df_filtered[df_filtered['commodity']==commodity]
if method_data_collection != 'All':
    df_filtered=df_filtered[df_filtered['method_data_collection']==method_data_collection]
#filter row 2
if Region != 'All':
    df_filtered=df_filtered[df_filtered['Region']==Region]
if Commodity_Groups != 'All':
    df_filtered=df_filtered[df_filtered['Commodity Groups']==Commodity_Groups]  
if Income != 'All':
    df_filtered=df_filtered[df_filtered['Income']==Income]  


########################################
# Prepare Data for Visualisation
########################################
df_map=pd.pivot_table(df_filtered,index='country', values='loss_percentage',aggfunc=np.mean, fill_value=None)
df_map=df_map.reset_index()
########################################
# Create Chloropleth Map Visualisation
########################################
# create figure
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
#  Create Linechart  Visualisation
########################################

# prepare FAO Data for linechart
df_line_chart=df_filtered.groupby('year').agg('mean')

# create figure
fig1 = go.Figure()

fig1.add_trace(
    go.Scatter(x=list(df_line_chart.index), y=list(df_line_chart['loss_percentage'])))

# update figure
fig1.update_layout(
    title_text="Mean Food Loss Percentage",
    template='seaborn',
    width=300, height=400
)

# update figure
fig1.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

#show figure
st.plotly_chart(fig1, use_container_width=True)


########################################
#  Barchart  Visualisation
########################################
#select variable
option_barchart= st.radio('See value counts for selected variable:',['Region','Income', 'Commodity Groups','food_supply_stage','method_data_collection','country','commodity'], horizontal= True)

#prepare dataset
df_bar_chart=df_filtered[df_filtered["loss_percentage"]>0]
df_bar_chart=df_bar_chart[option_barchart].value_counts()
Total_counts=df_bar_chart.sum()
df_bar_chart=pd.DataFrame(df_bar_chart)
df_bar_chart["%"]=df_bar_chart[f'{option_barchart}']/Total_counts
other_total={"Others":(df_bar_chart[7:].sum(axis=0))}
other_total=pd.DataFrame(other_total)
other_total=other_total.transpose()
df_bar_chart=df_bar_chart[0:7]
if int(other_total[option_barchart])!= 0:
    df_bar_chart=df_bar_chart.append(other_total)
df_bar_chart["%"]=df_bar_chart["%"].mul(100).round(1).astype(str) + '%'

#show data table & visualization
col5,col6 = st.columns([1,2])
with col5:
    st.write(f'value counts table for {option_barchart}')
    st.dataframe(df_bar_chart)

df_bar_chart=df_bar_chart[0:7]
df_bar_chart=df_bar_chart[f'{option_barchart}'].sort_values(ascending=True)
if int(other_total[option_barchart])!= 0:
    df_bar_chart=pd.concat([other_total[option_barchart],df_bar_chart])

#filter Nan row in Income
if 'Nan' in df_bar_chart.index:
    df_bar_chart=df_bar_chart.drop(index='Nan')
fig2 = px.bar(df_bar_chart, x=f"{option_barchart}", title=f"Barchart of value counts for {option_barchart}",  orientation='h')
fig2.update_layout(xaxis_title="value counts", yaxis_title=f'{option_barchart}', template='seaborn',)

with col6:
    st.plotly_chart(fig2, use_container_width=True)



########################################
#  Correlation Visualisation
########################################
mulitselect_country=st.multiselect(f'Select specific categories for comparison',df_filtered[f"{option_barchart}"].unique())


col7,col8=st.columns([1,2])

df_corr=df_filtered.groupby([f"{option_barchart}","year"]).agg('mean')
df_corr=df_corr.reset_index().drop(columns=['m49_code','year'])
df_corr=df_corr[df_corr[f'{option_barchart}'].isin(mulitselect_country)]
df_corr=pd.get_dummies(df_corr)
df_corr=df_corr.corr()['loss_percentage'].sort_values(ascending=False).astype('float')
with col7:
    st.write(f'Correlation of {option_barchart} and Food Loss %')
    st.dataframe(df_corr)



########################################
#  Linechart Visualisation
########################################
df_line_chart_categories=df_filtered.groupby([f"{option_barchart}","year"]).agg('mean')
df_line_chart_categories=df_line_chart_categories.reset_index()
df_line_chart_categories=df_line_chart_categories[df_line_chart_categories[f'{option_barchart}'].isin(mulitselect_country)]
fig3 = px.line(df_line_chart_categories, x="year", y="loss_percentage", color=f"{option_barchart}", title=f"Linechart of food loss percentage for {option_barchart}",markers=True)
fig3.update_layout(xaxis_title="year", yaxis_title=f'Food Loss %', template='seaborn')
with col8:
    st.plotly_chart(fig3, use_container_width=True)

########################################
#  Histogram Visualisation
########################################
df_histogram_chart=df_filtered.groupby([f"{option_barchart}","year"]).agg('count')
df_histogram_chart=df_histogram_chart.reset_index()
df_histogram_chart=df_histogram_chart[df_histogram_chart[f'{option_barchart}'].isin(mulitselect_country)]
fig4=px.histogram(df_histogram_chart, x="year", y="loss_percentage", color=f"{option_barchart}" ,nbins=(years[1]-years[0]+1), marginal="rug", title=f'Histogramm of selected data',hover_data=df_filtered.columns)
fig4.update_layout(xaxis_title='year', yaxis_title='count of datapoints', template='seaborn')
st.plotly_chart(fig4, use_container_width=True)
########################################
#  Further Insight in raw Data
########################################
with st.expander('See selected Data'):
    st.dataframe(df_filtered)
with st.expander('See Data Profil'):
    if st.button('Start Profiling'):
        pr = df.profile_report()
        st_profile_report(pr)
