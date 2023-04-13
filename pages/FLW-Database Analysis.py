########################################
# Importieren der Module
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
# Kofiguration der Seite
########################################
st.set_page_config(page_title="Food Loss Database", page_icon="🥑",layout="wide")


########################################
# Einfügen von Titeln
########################################
st.header("Food Loss & Waste Database by FAO")
st.sidebar.header("Food Loss & Waste Database by FAO")


########################################
# Daten einlesen und bereinigen
########################################

#FLW-Database einlesen
df = pd.read_csv("data/Data.csv")

# einlesen der Zuordnungsdatei von Ländern zu Einkommensgruppen und Regionen nach der Weltbank
country_categories_tidy= pd.read_csv('data/country_categories_tidy.csv', delimiter=';')

# einlesen der Zuordnungsdatei von Waren zu Warengruppen
commodity_groups_tidy= pd.read_csv('data/commodity_groups.csv', delimiter=';')

# Löschen von Duplikaten
df=df.drop_duplicates()

# Bereinigung der Ländernamen
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

# Löschen der Datenpunkte der Regionen in der FLW-Database
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
# Zusammenführen der Daten
########################################

# Hinzufügen der Einkommenguppen und Regionen zu der FLW-Database
df_merged=pd.merge(df,country_categories_tidy, how='inner', left_on='country', right_on='Country')
df_merged=df_merged.drop(columns=['Country'])
df_merged=df_merged.ffill(axis = 0)

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

df_merged['Income']=df_merged['Income'].replace(dict_income_group_dt)

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

df_merged['Region']=df_merged['Region'].replace(dict_region_dt)

# Hinzufügen der Warengruppen zu der FLW-Database
df_merged=pd.merge(df_merged,commodity_groups_tidy, how='inner', left_on='commodity', right_on='Commodities')
df_merged=df_merged.drop(columns=['Commodities'])
df_merged=df_merged.ffill(axis = 0)


########################################
# Erstellen der Filteroptionen
########################################

#Einfügen einer Handlungsaufforderung
st.write("---")
st.write('Select filter')

years = st.slider(
    'Select years', 
    min_value=min(df['year']), 
    value=(min(df['year']), max(df['year'])), 
     max_value=max(df['year']))

# Hinzufügen der ersten 4 Filteroptionen
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

# Hinzufügen der 3 weiterer Filteroptionen
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
# Filterung der Daten nach der gewählten Filtern
########################################

#Filterung des Zeitraums
df_filtered = df_merged[(df_merged['year'] >= years[0]) & (df_merged['year'] <= years[1])]

#Filterung nach den ersten 4 Filteroptionen
if country != 'All':
    df_filtered=df_filtered[df_filtered['country']==country]
if food_supply_stage != 'All':
    df_filtered=df_filtered[df_filtered['food_supply_stage']==food_supply_stage]
if commodity != 'All':
    df_filtered=df_filtered[df_filtered['commodity']==commodity]
if method_data_collection != 'All':
    df_filtered=df_filtered[df_filtered['method_data_collection']==method_data_collection]

#Filterung der 3 weiteren Filter 
if Region != 'All':
    df_filtered=df_filtered[df_filtered['Region']==Region]
if Commodity_Groups != 'All':
    df_filtered=df_filtered[df_filtered['Commodity Groups']==Commodity_Groups]  
if Income != 'All':
    df_filtered=df_filtered[df_filtered['Income']==Income]  


########################################
# Vorbereitung der Daten für die Visualisierung
########################################

df_map=pd.pivot_table(df_filtered,index='country', values='loss_percentage',aggfunc=np.mean, fill_value=None)
df_map=df_map.reset_index()


########################################
# Erstellen der Chloroplethenkarte
########################################

# Erstellen des Graphen
fig = go.Figure(data=go.Choropleth(
    locations=df_map['country'], # Spatial coordinates
    z = df_map['loss_percentage'].astype(float), # Data to be color-coded
    locationmode = 'country names', # set of locations match entries in `locations`
    colorscale = 'tempo',
    colorbar_title = "%",
))
fig.update_layout(margin={"r":0,"t":25,"l":0,"b":0})
titel_eng_map='Food Loss Percentage'
#titel_dt_map="Choroplethenkarte der gemittelten prozentualen Lebensmittelverluste und -abfälle"
fig.update_layout(
    title_text = titel_eng_map,
    title_x=0.5,
    geo_scope='world', 
    template='simple_white',
)

#Abbilden des Graphen
st.plotly_chart(fig, use_container_width=True)


########################################
#  Erstellen des Liniendiagramm 1
########################################

# Vorbereitung der Daten
df_line_chart=df_filtered.groupby('year').agg('mean')

# Erstellen des Graphen
fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(x=list(df_line_chart.index), y=list(df_line_chart['loss_percentage'])))
titel_eng_map="Mean Food Loss Percentage"
#titel_dt_map="Liniendiagramm gemittelter Lebensmittelverluste und -abfälle"
fig1.update_layout(
    title_text=titel_eng_map,
    template='seaborn',
    width=300, height=400,
    xaxis_title='Foodloss and waste in %', 
    yaxis_title= 'year'
)
fig1.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

# Abbilden des Graphen
st.plotly_chart(fig1, use_container_width=True)


########################################
#  Einfügen von weiteren Filteroptionen 
########################################

# Wahl der Filteroption
option_barchart= st.radio('See value counts for selected variable:',['Region','Income', 'Commodity Groups','food_supply_stage','method_data_collection','country','commodity'], horizontal= True)

# Vorbereitung der Daten nach gewählter Filteroption
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

# Datenpräsentation
col5,col6 = st.columns([1,2])
with col5:
    # Zeigen der Datentabelle
    st.write(f'value counts table for {option_barchart}')
    st.dataframe(df_bar_chart)

# Anpassen der Daten für die Visualisierung
df_bar_chart=df_bar_chart[0:7]
df_bar_chart=df_bar_chart[f'{option_barchart}'].sort_values(ascending=True)
if int(other_total[option_barchart])!= 0:
    df_bar_chart=pd.concat([other_total[option_barchart],df_bar_chart])
if 'Nan' in df_bar_chart.index:
    df_bar_chart=df_bar_chart.drop(index='Nan')

# Erstellen des Balkendiagramms
titel_eng_barchart="Barchart of value counts for"
#titel_dt_barchart="Balkendiagramm der Häufigkeiten nach Methoden der Datenerhebung"
fig2 = px.bar(df_bar_chart, x=f"{option_barchart}", title=f"{titel_eng_barchart} {option_barchart}",  orientation='h') #{option_barchart}
xaxis_eng_bar2="value counts"
xaxis_dt_bar2='Anzahl der Datenpunkte'
yaxis_dt_bar2='Methoden der Datenerhebung'
yaxis_eng_bar2={option_barchart}
if option_barchart=="Commodity Groups":
    yaxis_dt_bar2='Warengruppe'
    yaxis_eng_bar2='Commodity Groups'
fig2.update_layout(xaxis_title=xaxis_eng_bar2, yaxis_title=yaxis_eng_bar2, template='seaborn',)

# Abbilden des Graphen
with col6:
    st.plotly_chart(fig2, use_container_width=True)


########################################
#  Zeigen der Korrelation nach weiterer Filteroption
########################################

# Auswahl der Filteroption
mulitselect_country=st.multiselect(f'Select specific categories for comparison',df_filtered[f"{option_barchart}"].unique())

col7,col8=st.columns([1,2])

# Berechnung der Korrelation nach Filteroption
df_corr=df_filtered.groupby([f"{option_barchart}","year"]).agg('mean')
df_corr=df_corr.reset_index().drop(columns=['m49_code','year'])
df_corr=df_corr[df_corr[f'{option_barchart}'].isin(mulitselect_country)]
df_corr=pd.get_dummies(df_corr)
df_corr=df_corr.corr()['loss_percentage'].sort_values(ascending=False).astype('float')
with col7:
    # Zeigen der Datentabelle
    st.write(f'Correlation of {option_barchart} and Food Loss %')
    st.dataframe(df_corr)


########################################
#  Erstellen des Liniendiagramm 2 
########################################

# Anpassen der Daten für die Visualisierung
df_line_chart_categories=df_filtered.groupby([f"{option_barchart}","year"]).agg('mean')
df_line_chart_categories=df_line_chart_categories.reset_index()
df_line_chart_categories=df_line_chart_categories[df_line_chart_categories[f'{option_barchart}'].isin(mulitselect_country)]

# Erstellen des Liniendiagramms  
titel_eng_linechart2="Linechart of food loss percentage for"
titel_dt_linechart2="Liniendiagramm Lebensmittelverluste nach Stufen der Lebensmittlelieferkette"
fig3 = px.line(df_line_chart_categories, x="year", y="loss_percentage", color=f"{option_barchart}", title=f"{titel_eng_linechart2}",markers=True)
xaxis_eng_line2="year"
xaxis_dt_line2='Jahr'
yaxis_eng_line2="Food Loss "
yaxis_dt_line2='Lebensmittelverluste in'
fig3.update_layout(xaxis_title=xaxis_eng_line2, yaxis_title=f'{yaxis_eng_line2} %',legend_title_text=f'{option_barchart}', template='seaborn')

# Abbilden des Graphen
with col8:
    st.plotly_chart(fig3, use_container_width=True)


########################################
#  Erstellen des Histogramms 
########################################

# Anpassen der Daten für die Visualisierung
df_histogram_chart=df_filtered.groupby([f"{option_barchart}","year"]).agg('count')
df_histogram_chart=df_histogram_chart.reset_index()
df_histogram_chart=df_histogram_chart[df_histogram_chart[f'{option_barchart}'].isin(mulitselect_country)]

# Erstellen des Liniendiagramms 
titel_eng_hist="Histogramm of "
titel_dt_hist="Histogramm nach Stufen der Lebensmittlelieferkette"
xaxis_eng_hist="year"
xaxis_dt_hsit='Jahr'
yaxis_eng_hist="value counts "
yaxis_dt_hist='Anzahl der Datenpunkte'
fig4=px.histogram(df_histogram_chart, x="year", y="loss_percentage", color=f"{option_barchart}" ,nbins=(years[1]-years[0]+1), marginal="rug", title=f'{titel_eng_hist}',hover_data=df_filtered.columns)
fig4.update_layout(xaxis_title=xaxis_eng_hist, yaxis_title=yaxis_eng_hist,legend_title_text=f'{option_barchart}', template='seaborn')

# Abbilden des Graphen
st.plotly_chart(fig4, use_container_width=True)


########################################
# Weiter Einsichten in die Rohdaten
########################################

# Zeigen der zugrundeliegenden gefilterten Daten
with st.expander('See selected Data'):
    st.dataframe(df_filtered)

# Möglichkeit einer Datananalyse mittels Pandas selber für die Originaldaten durchzuführen
with st.expander('See Data Profil'):
    if st.button('Start Profiling'):
        pr = df.profile_report()
        st_profile_report(pr)
