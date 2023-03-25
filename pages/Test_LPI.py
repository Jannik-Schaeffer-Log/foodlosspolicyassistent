########################################
# Import libraries
########################################
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px

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
country_categories_tidy= pd.read_csv('data/country_categories_tidy.csv', delimiter=';')
df_LPI_new_country_names= pd.read_csv("data/LPI_Countries_name_changes.csv", delimiter=';')
########################################
# Prepare Data
########################################
def Change_Country_Names( df_new_names,column_new_names ,df_to_change,column_name):
    name_dict=df_new_names.set_index(column_new_names).T.to_dict('list')
    df_to_change[column_name]=df_to_change[column_name].map(name_dict)
    df_to_change[column_name]=df_to_change[column_name].str.get(0)

Change_Country_Names(df_LPI_new_country_names,'LPI Countries',df, 'Country Name' )

df_LPI=df[["Country Name","Series Name","2007 [YR2007]","2010 [YR2010]","2012 [YR2012]","2014 [YR2014]","2016 [YR2016]","2018 [YR2018]"]]
df_LPI.columns=["country","Series Name","2007","2010","2012","2014","2016","2018"]
df_LPI=pd.melt(df_LPI,["country","Series Name"],var_name="year",value_name="score_rank")
## Add Country Categories to the Dataframe
df_LPI=pd.merge(df_LPI,country_categories_tidy, how='inner', left_on='country', right_on='Country')
df_LPI=df_LPI.drop(columns=['Country'])
df_LPI=df_LPI.ffill(axis = 0)

df_LPI=df_LPI.replace(to_replace='..', value=None)
convert_dtypes_dict={
    'country':object,
    'Series Name':object,
    'year':int
    }
df_LPI = df_LPI.astype(convert_dtypes_dict)


# ########################################
# # Create Filters
# ########################################
years = st.select_slider(
    'Select years', 
    options=['2007', '2010', '2012', '2014', '2016', '2018'],
    value=('2007', '2018')
    )
years=pd.DataFrame(years).astype("int")
df_LPI = df_LPI[(df_LPI['year'] >= years.loc[0,0]) & (df_LPI['year'] <= years.loc[1,0])]


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
            "Choose overall score or category", ['All']+list(series_list)
        )

col3, col4,col5= st.columns(3)
with col3:
    country = st.selectbox(
            "Choose country", ['All']+list(df_LPI['country'].unique())
        )
with col4:
    Region= st.selectbox(
            "Choose region", ['All']+list(df_LPI['Region'].unique())
        )
with col5:
    Income = st.selectbox(
            "Choose income-group", ['All']+list(df_LPI['Income'].unique())
        )
#filter data
df_filtered=df_LPI
if series_name != 'All':
    df_filtered=df_filtered[df_filtered['Series Name'].str.contains(series,na=False)]
if country != 'All':
    df_filtered=df_filtered[df_filtered['country']==country]
if Region != 'All':
    df_filtered=df_filtered[df_filtered['Region']==Region]
if Income != 'All':
    df_filtered=df_filtered[df_filtered['Income']==Income] 

df_filtered=df_filtered[df_filtered['Series Name'].isin(series_list)]

# ########################################
# # Clean Data 
# # Change Data Types
# # Filter for rank or score
# ########################################
df_filtered=df_filtered.replace(to_replace='..', value=None)
convert_dtypes_dict={
    'country':object,
    'Series Name':object,
    'year':int,
    'score_rank':float
    }
df_filtered = df_filtered.astype(convert_dtypes_dict)



########################################
# Prepare Data for Visualisation
########################################
if series_name == 'All':
    if series == 'score':
        df_map=df_filtered[df_filtered['Series Name']=='Logistics performance index: Overall score (1=low to 5=high)']
        #df_map=df_map.groupby('country').agg('mean')
    else:
        df_map=df_filtered[df_filtered['Series Name']=='Logistics performance index: Overall rank (1=highest performance)']
        #df_map=df_map.groupby('country').agg('mean')
else:   
    df_map=df_filtered
df_map=df_map[['country','score_rank']]
#df_map=df_map.astype({'country': 'category','score_rank':'float'})
df_map=df_map.groupby('country').mean()

########################################
# Create Choropleth Map Visualisation
########################################
fig1 = go.Figure(data=go.Choropleth(
    locations=df_map.index, # Spatial coordinates
    z = df_map['score_rank'], # Data to be color-coded
    locationmode = 'country names', # set of locations match entries in `locations`
    colorscale = 'tempo',
    colorbar_title = series,
))
fig1.update_layout(margin={"r":0,"t":25,"l":0,"b":0})
fig1.update_layout(
    title_text = series_name,
    title_x=0.5,
    geo_scope='world', 
    template='simple_white',
)

#show Visualisation
st.plotly_chart(fig1, use_container_width=True)


########################################
# Create Linechart
########################################
radio_option= st.radio('Select Category:',['Region','Income','country', 'Series Name'], horizontal= True)

if radio_option=='country':
    selected_countries=st.multiselect('Select countries',df_filtered['country'].unique())
    df_filtered=df_filtered[df_filtered['country'].isin(selected_countries)]

if radio_option=='Income':
    df_filtered=df_filtered[df_filtered['Income']!='Nan']
########################################
# Create Linechart
########################################
df_linechart=df_filtered.groupby([f'{radio_option}','year']).mean()
df_linechart=df_linechart.reset_index()

fig2 = px.line(df_linechart, x="year", y="score_rank", color=f"{radio_option}", title=f"Linechart of {series} for {radio_option}",markers=True)
fig2.update_layout(xaxis_title="year", yaxis_title=f'{series}', template='seaborn')
st.plotly_chart(fig2, use_container_width=True)

########################################
# Create Histogram
########################################
# df_histogram=df_filtered.groupby([f'{radio_option}','year']).agg('count')
# df_histogram=df_histogram.reset_index()

# fig3=px.histogram(df_histogram, x="year", y="score_rank", color=f"{radio_option}" , marginal="rug", title=f'Histogramm of selected data')
# fig3.update_layout(xaxis_title='year', yaxis_title='count of datapoints', template='seaborn')
# st.plotly_chart(fig3, use_container_width=True)

########################################
# Create Boxplot
########################################
df_boxplot=df_filtered
if radio_option!='Series Name': 
    selected_series=st.selectbox('Select series',df_boxplot['Series Name'].unique())
    df_boxplot=df_boxplot[df_boxplot['Series Name']==selected_series]




# df_boxplot=df_boxplot.groupby([f'{radio_option}']).agg('mean')
# df_boxplot=df_boxplot.reset_index()


fig4 = px.box(df_boxplot, x=f'{radio_option}', y="score_rank")
fig4.update_layout(xaxis_title=f'{radio_option}', yaxis_title=f'{series}', template='seaborn')
st.plotly_chart(fig4, use_container_width=True)
# # build dataframe for regression wit LPI-data
# st.write(df_series)
# df_LPI=df_series[["Country Name","Series Name","2007 [YR2007]","2010 [YR2010]","2012 [YR2012]","2014 [YR2014]","2016 [YR2016]","2018 [YR2018]"]]
# df_LPI.columns=["Country Name","Series Name","2007","2010","2012","2014","2016","2018"]
# st.write(df_LPI)
# df_LPI = df_series.pivot(index="Country Name",columns='Series Name', values=["2007","2010","2012","2014","2016","2018"])
# st.write(df_LPI)
# y=[]
# years=["2007","2010","2012","2014","2016","2018"]
# x=df_LPI[years[0]]
# x['year']=years[0]
# y=x
# for i in range(1,len(years)):
#     x=df_LPI[years[i]]
#     x['year']=years[i]
#     y=y.append(x)

# y=y.sort_values(['Country Name','year'])
# # changing datatype
# y['year']=y['year'].astype('int64')


# #df_LPI_complete=

with st.expander('See Data Profil'):
    if st.button('Start Profiling'):
        pr = df.profile_report()
        st_profile_report(pr)

with st.expander('See Data'):
    st.dataframe(df_filtered)