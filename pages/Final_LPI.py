########################################
# Importieren der Module
########################################
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px


########################################
# Kofiguration der Seite
########################################
st.set_page_config(page_title="Test", page_icon="🥑",layout="wide")


########################################
# Einfügen von Titeln
########################################
st.header("Logistic Performance Index by World Bank")
st.sidebar.header("Logistic Performance Index by World Bank")


########################################
# Daten einlesen und bereinigen
########################################
df = pd.read_csv("data\LPI_Data.csv", encoding='latin-1')
country_categories_tidy= pd.read_csv('data/country_categories_tidy.csv', delimiter=';')
df_LPI_new_country_names= pd.read_csv("data/LPI_Countries_name_changes.csv", delimiter=';')


########################################
# Vorbereitung der Daten
########################################
def Change_Country_Names( df_new_names,column_new_names ,df_to_change,column_name):
    name_dict=df_new_names.set_index(column_new_names).T.to_dict('list')
    df_to_change[column_name]=df_to_change[column_name].map(name_dict)
    df_to_change[column_name]=df_to_change[column_name].str.get(0)

Change_Country_Names(df_LPI_new_country_names,'LPI Countries',df, 'Country Name' )

df_LPI=df[["Country Name","Series Name","2007 [YR2007]","2010 [YR2010]","2012 [YR2012]","2014 [YR2014]","2016 [YR2016]","2018 [YR2018]"]]
df_LPI.columns=["country","Series Name","2007","2010","2012","2014","2016","2018"]
df_LPI=pd.melt(df_LPI,["country","Series Name"],var_name="year",value_name="score_rank")

# Hinzufügen der Einkommenguppen und Regionen zum LPI Datensatz
df_LPI=pd.merge(df_LPI,country_categories_tidy, how='inner', left_on='country', right_on='Country')
df_LPI=df_LPI.drop(columns=['Country'])
df_LPI=df_LPI.ffill(axis = 0)


dict_income_group_dt={
    'UPPER-MIDDLE-INCOME ECONOMIES ($4,256 TO $13,205)':'Hohes-Mittleres Einkommen',
    'LOW-INCOME ECONOMIES ($1,085 OR LESS) ':'Niedriges Einkommen',
    'LOWER-MIDDLE INCOME ECONOMIES ($1,086 TO $4,255)':'Niedriges-Mittleres Einkommen',
    'HIGH-INCOME ECONOMIES ($13,205 OR MORE) ':'Hohes Einkommen',
    'Nan':'Niedriges-Mittleres Einkommen',
}
dict_income_group={
    'UPPER-MIDDLE-INCOME ECONOMIES ($4,256 TO $13,205)':'Upper-Middle-Income Economies',
    'LOW-INCOME ECONOMIES ($1,085 OR LESS) ':'Low-Income Economies',
    'LOWER-MIDDLE INCOME ECONOMIES ($1,086 TO $4,255)':'Lower-Middle Income Economies',
    'HIGH-INCOME ECONOMIES ($13,205 OR MORE) ':'High-Income Economies',
    'Nan':'Lower-Middle Income Economies',
}

df_LPI['Income']=df_LPI['Income'].replace(dict_income_group_dt)

dict_region={
    'SUB-SAHARAN AFRICA': 'Sub-Saharan Africa',
    'MIDDLE EAST AND NORTH AFRICA':'Middle East and North Africa',
    'SOUTH ASIA':'South Asia',
    'LATIN AMERICA AND THE CARIBBEAN':'Latin America & the Caribbean',
    'EAST ASIA AND PACIFIC':'East Asia and Pacific',
    'EUROPE AND CENTRAL ASIA':'Europe and Central Asia',
    'NORTH AMERICA':'North America',
}
dict_region_dt={
    'SUB-SAHARAN AFRICA': 'Sub-Sahara Afrika',
    'MIDDLE EAST AND NORTH AFRICA':'Naher Osten und Nordafrika',
    'SOUTH ASIA':'Südasien',
    'LATIN AMERICA AND THE CARIBBEAN':'Lateinamerika und die Karibik',
    'EAST ASIA AND PACIFIC':'Ostasien und Pazifik',
    'EUROPE AND CENTRAL ASIA':'Europa und Zentralasien',
    'NORTH AMERICA':'Nord Amerika',
}

df_LPI['Region']=df_LPI['Region'].replace(dict_region_dt)




df_LPI=df_LPI.replace(to_replace='..', value=None)
convert_dtypes_dict={
    'country':object,
    'Series Name':object,
    'year':int
    }
df_LPI = df_LPI.astype(convert_dtypes_dict)


########################################
# Erstellen der Filteroptionen
########################################

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


########################################
# Filterung der Daten nach der gewählten Filtern
########################################
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

########################################
#Ändern der Datentypen
########################################
df_filtered=df_filtered.replace(to_replace='..', value=None)
convert_dtypes_dict={
    'country':object,
    'Series Name':object,
    'year':int,
    'score_rank':float
    }
df_filtered = df_filtered.astype(convert_dtypes_dict)


########################################
# Vorbereitung der Daten für die Visualisierung
########################################
if series_name == 'All':
    if series == 'score':
        df_map=df_filtered[df_filtered['Series Name']=='Logistics performance index: Overall score (1=low to 5=high)']
    else:
        df_map=df_filtered[df_filtered['Series Name']=='Logistics performance index: Overall rank (1=highest performance)']     
else:   
    df_map=df_filtered
df_map=df_map[['country','score_rank']]
df_map=df_map.groupby('country').mean()


########################################
# Erstellen der Chloroplethenkarte
########################################

# Erstellen Chloroplethenkarte
fig1 = go.Figure(data=go.Choropleth(
    locations=df_map.index, # Spatial coordinates
    z = df_map['score_rank'], # Data to be color-coded
    locationmode = 'country names', # set of locations match entries in `locations`
    colorscale = 'tempo',
    colorbar_title = 'Wertung',
))
fig1.update_layout(margin={"r":0,"t":25,"l":0,"b":0})
fig1.update_layout(
    title_text = 'Chloroplethenkarte der gemittelten LPI-Gesamtwertungen von 2007 bis 2018',
    title_x=0.5,
    geo_scope='world', 
    template='simple_white',
)

# Abbilden des Graphen
st.plotly_chart(fig1, use_container_width=True)

########################################
#  Änderung der LPI-Kategorienamen
########################################
dict_series={
    'Logistics performance index: Overall score (1=low to 5=high)':"LPI-Gesamtwertung",
    'Ability to track and trace consignments, score (1=low to 5=high)':"Tracking and tracing",
    'Competence and quality of logistics services, score (1=low to 5=high)':'Quality of log. services',
    'Ease of arranging competitively priced international shipments, score (1=low to 5=high)':'Ease of arranging shipments',
    'Efficiency of the clearance process, score (1=low to 5=high)':'Customs',
    'Frequency with which shipments reach consignee within scheduled or expected time, score (1=low to 5=high)':'Timeliness',
    'Quality of trade- and transport-related infrastructure, score (1=low to 5=high)':'Infrastructure'
}

df_filtered['Series Name']=df_filtered['Series Name'].replace(dict_series)


########################################
#  Erstellen des Liniendiagramm
########################################
# Auswahl der zu betrachtenden Datenreihe
radio_option= st.radio('Select Category:',['Region','Income','country', 'Series Name'], horizontal= True)

# Anpassen der Daten für die Visualisierung
if radio_option=='country':
    selected_countries=st.multiselect('Select countries',df_filtered['country'].unique())
    df_filtered=df_filtered[df_filtered['country'].isin(selected_countries)]

if radio_option=='Income':
    df_filtered=df_filtered[df_filtered['Income']!='Nan']

df_linechart=df_filtered[df_filtered['Series Name']!='LPI-Gesamtwertung'].groupby([f'{radio_option}','year']).mean()
df_linechart=df_linechart.reset_index()

# Erstellen des Liniendiagramms
fig2 = px.line(df_linechart, x="year", y="score_rank", color=f"{radio_option}", title=f"Liniendiagramm der LPI-Gesamtwertungen der Regionen",markers=True)
fig2.update_layout(xaxis_title="Jahr", yaxis_title=f'Wertung', template='seaborn')

# Abbilden des Graphen
st.plotly_chart(fig2, use_container_width=True)


########################################
#  Erstellen des Boxplot-Diagramms
########################################

# Anpassen der Daten für die Visualisierung
df_boxplot=df_filtered#[df_filtered['Series Name']!='LPI-Gesamtwertung']
if radio_option!='Series Name': 
    selected_series=st.selectbox('Select series',df_boxplot['Series Name'].unique())
    df_boxplot=df_filtered[df_filtered['Series Name']!='LPI-Gesamtwertung']

# Erstellen des Boxplot-Diagramms
ytitle_eng_box=series
ytitle_dt_box='Wertung'
xtitle_dt_box=radio_option
if radio_option == 'Series Name':
    xtitle_dt_box='LPI-Kategorien'
if radio_option == 'Income':
    xtitle_dt_box='Einkommensgruppe'

fig4 = px.box(df_boxplot, x=f'{radio_option}', y="score_rank", title='Boxplot-Diagramm der LPI-Gesamtwertungen nach Regionen' )
fig4.update_layout(yaxis_title=f'{ytitle_dt_box}', xaxis_title=f'{xtitle_dt_box}', template='seaborn')

# Abbilden des Graphen
st.plotly_chart(fig4, use_container_width=True)


########################################
# Further Insight in raw Data
########################################

# Zeigen der zugrundeliegenden gefilterten Daten
with st.expander('See Data Profil'):
    if st.button('Start Profiling'):
        pr = df.profile_report()
        st_profile_report(pr)

# Möglichkeit einer Datananalyse mittels Pandas selber für die Originaldaten durchzuführen
with st.expander('See Data'):
    st.dataframe(df_filtered)