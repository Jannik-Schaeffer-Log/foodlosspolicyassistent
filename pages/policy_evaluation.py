########################################
# Import libraries
########################################
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import scipy.stats

########################################
# Configure Page
########################################
st.set_page_config(page_title="Country Food Loss Policy Assistant", page_icon="🥑",layout="wide")

########################################
# Add Title
########################################
st.header("Country Food Loss Policy Assistant")
st.sidebar.header("Country Food Loss Policy Assistant")
########################################
# Read Data
########################################
# LPI Data
df_LPI= pd.read_csv("data/LPI_Data.csv", encoding='latin-1')
# FAO Data
df_FAO = pd.read_csv("data/Data.csv")
# Country Categories by Worldbank 
country_categories_tidy= pd.read_csv('data/country_categories_tidy.csv', delimiter=';')
# Commodity Groups by FAO 
commodity_groups_tidy= pd.read_csv('data/commodity_groups.csv', delimiter=';')
# LPI Country Name Changes
df_LPI_new_country_names= pd.read_csv("data/LPI_Countries_name_changes.csv", delimiter=';')
# FAO Country Name Changes
df_FAO_new_country_names= pd.read_csv("data/FAO_Countries_name_changes.csv", delimiter=';')
# FAO Country Name Changes
df_policies= pd.read_csv("data/All_Food_Loss_Policies.csv", delimiter=';')

########################################
# Clean Data 
# Change Data Types
########################################
# Drop duplicates & name change
df=df_FAO.drop_duplicates()

#get loss-percentage in range 0-100
with st.expander('See loss_percentage plot'):
    st.write(df['loss_percentage'].max() , df['loss_percentage'].min(),df['loss_percentage'].mean() )
    data_swarmplot = df['loss_percentage']
    fig_swarm = px.box(data_swarmplot, y='loss_percentage', title='Boxplot-Diagramm der prozentualen Lebensmittelverluste und -abfälle')
    fig_swarm.update_layout(xaxis_title='Lebensmittelverluste und -abfälle', yaxis_title='%')
    st.plotly_chart(fig_swarm, use_container_width=True)

#drop food supply stages which aren't relevant for food loss
food_supply_stages=[
    'Whole supply chain',
    'Storage',
    'Processing',
    'Trader',
    'Wholesale',
    'Post-harvest',
    'Transport',
    'Export',
    '<NA>',
    'Distribution',
    'Market',
    'Stacking',
    'Grading',
    'Packing'
    ]

#st.write(df['food_supply_stage'].unique())
df=df[df['food_supply_stage'].isin(food_supply_stages)]


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

# Drop unwanted Region data
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

#change name Back
df_FAO=df

#align country names for later merge
def Change_Country_Names( df_new_names,column_new_names ,df_to_change,column_name):
    name_dict=df_new_names.set_index(column_new_names).T.to_dict('list')
    df_to_change[column_name]=df_to_change[column_name].map(name_dict)
    df_to_change[column_name]=df_to_change[column_name].str.get(0)

Change_Country_Names(df_LPI_new_country_names,'LPI Countries',df_LPI, 'Country Name' )

#Change_Country_Names(df_FAO_new_country_names,'FAO Countries',df_FAO, 'country' )

# select columns from LPI-data
df_LPI_reg=df_LPI[["Country Name","Series Name","2007 [YR2007]","2010 [YR2010]","2012 [YR2012]","2014 [YR2014]","2016 [YR2016]","2018 [YR2018]"]]

# rename columns
df_LPI_reg.columns=["Country Name","Series Name","2007","2010","2012","2014","2016","2018"]

# select only score data from LPI
df_LPI_reg=df_LPI_reg[df_LPI_reg["Series Name"].str.contains('score',na=False)]
df_LPI_reg=df_LPI_reg.dropna()
# build dataframe for regression wit LPI-data
df_LPI_reg = df_LPI_reg.pivot(index="Country Name",columns='Series Name', values=["2007","2010","2012","2014","2016","2018"])
y=[]
years=["2007","2010","2012","2014","2016","2018"]
x=df_LPI_reg[years[0]]
x['year']=years[0]
y=x
for i in range(1,len(years)):
    x=df_LPI_reg[years[i]]
    x['year']=years[i]
    y=y.append(x)

y=y.sort_values(['Country Name','year'])
# changing datatype
y['year']=y['year'].astype('int64')
st.write(len(y.index.unique()))

# prepare FAO Data for merge
df_FAO_pivot=pd.pivot_table(df_FAO,index=['country','year','commodity','food_supply_stage'], values='loss_percentage',aggfunc=np.mean, fill_value=None)
df_FAO_pivot=df_FAO_pivot.reset_index()

st.write(len(df_FAO_pivot['country'].unique()))
colA, colB= st.columns(2)
with colA:
    st.write(df_FAO_pivot['country'].unique())
with colB:
    st.write(y.index.unique())
# merge filtered data of both data source
df_reg=pd.merge(y,df_FAO_pivot,how='inner', left_on=['Country Name','year'],right_on=['country','year'])
df_reg=df_reg.replace("..",np.NaN)

st.write(len(df_reg['country'].unique()))

## Add Country Categories to the Dataframe
df_reg=pd.merge(df_reg,country_categories_tidy, how='inner', left_on='country', right_on='Country')
df_reg=df_reg.drop(columns=['Country'])
df_reg=df_reg.ffill(axis = 0)

## Add Commodity-Groups to the Dataframe
df_reg=pd.merge(df_reg,commodity_groups_tidy, how='inner', left_on='commodity', right_on='Commodities')
df_reg=df_reg.drop(columns=['Commodities'])
df_reg=df_reg.ffill(axis = 0)

##Add missing years to Dataframe
df_reg['year']=pd.to_datetime(df_reg['year'],format='%Y')
df_reg = df_reg.set_index('year')
country_list=df_reg['country'].unique()
df_reg_extended=pd.DataFrame()
for i in range(len(country_list)):
    test=df_reg[df_reg['country']==country_list[i]].resample('Y').last()
    df_reg_extended=df_reg_extended.append(test)
df_reg_extended=df_reg_extended.ffill(axis = 0)

#Add Year Column again
df_reg=df_reg_extended.reset_index()
df_reg['year']=df_reg['year'].astype('string')
df_reg['year']=df_reg['year'].apply(lambda x: x[0:4])
df_reg['year']=df_reg['year'].astype('int')

#Remerge foodloss_percentage
df_reg=df_reg.drop(columns=['loss_percentage'])
df_reg=pd.merge(df_reg,df_FAO_pivot,how='inner', left_on=['country','year','commodity','food_supply_stage'],right_on=['country','year','commodity','food_supply_stage'])
df_reg=df_reg.set_index('year')
df_reg=df_reg.drop(columns=['commodity'])

dict_income_group={
    'UPPER-MIDDLE-INCOME ECONOMIES ($4,256 TO $13,205)':'Upper-Middle-Income Economies',
    'LOW-INCOME ECONOMIES ($1,085 OR LESS) ':'Low-Income Economies',
    'LOWER-MIDDLE INCOME ECONOMIES ($1,086 TO $4,255)':'Lower-Middle Income Economies',
    'HIGH-INCOME ECONOMIES ($13,205 OR MORE) ':'High-Income Economies'
}

df_reg['Income']=df_reg['Income'].replace(dict_income_group)

# select Country
# selected_country=st.selectbox('Select country',options=df_reg['country'].unique())

# filter df for dependent Income-Group
# selected_income_group=df_reg[df_reg['country']==selected_country]['Income'].unique()
# selected_income_group=str(selected_income_group)[2:-2]

# # filter df for dependent Region
# selected_region=df_reg[df_reg['country']==selected_country]['Region'].unique()
# selected_region=str(selected_region)[2:-2]

#select Series
dict_series={
    'Ability to track and trace consignments, score (1=low to 5=high)':0,
    'Competence and quality of logistics services, score (1=low to 5=high)':1,
    'Ease of arranging competitively priced international shipments, score (1=low to 5=high)':2,
    'Efficiency of the clearance process, score (1=low to 5=high)':3,
    'Frequency with which shipments reach consignee within scheduled or expected time, score (1=low to 5=high)':4,
    'Logistics performance index: Overall score (1=low to 5=high)':5,
    'Logistics performance index: Overall score (1=low to 5=high), lower bound':6,
    'Logistics performance index: Overall score (1=low to 5=high), upper bound':7,
    'Quality of trade- and transport-related infrastructure, score (1=low to 5=high)':8,
}

########################################
# Correlation Test
########################################

with st.expander('Correlation of LPI-categories and Food Loss'):
    #prepare arrays for correlation test
    cor_array_x= np.array(df_reg['loss_percentage'].astype(float))
    cor_array_y_1= np.array(df_reg['Ability to track and trace consignments, score (1=low to 5=high)'].astype(float))
    cor_array_y_2= np.array(df_reg['Competence and quality of logistics services, score (1=low to 5=high)'].astype(float))
    cor_array_y_3= np.array(df_reg['Ease of arranging competitively priced international shipments, score (1=low to 5=high)'].astype(float))
    cor_array_y_4= np.array(df_reg['Efficiency of the clearance process, score (1=low to 5=high)'].astype(float))
    cor_array_y_5= np.array(df_reg['Frequency with which shipments reach consignee within scheduled or expected time, score (1=low to 5=high)'].astype(float))
    cor_array_y_6= np.array(df_reg['Quality of trade- and transport-related infrastructure, score (1=low to 5=high)'].astype(float))
    
    #Correlation Test
    r_1,p_1 = scipy.stats.pearsonr(cor_array_x, cor_array_y_1)
    r_2,p_2 = scipy.stats.pearsonr(cor_array_x, cor_array_y_2)
    r_3,p_3 = scipy.stats.pearsonr(cor_array_x, cor_array_y_3)
    r_4,p_4 = scipy.stats.pearsonr(cor_array_x, cor_array_y_4)
    r_5,p_5 = scipy.stats.pearsonr(cor_array_x, cor_array_y_5)
    r_6,p_6 = scipy.stats.pearsonr(cor_array_x, cor_array_y_6)
    corr_dict={"Pearson's r":[r_1,r_2,r_3,r_4,r_5,r_6,],'p-value':[p_1,p_2,p_3,p_4,p_5,p_6]}
    corr_index_names=['Tracking and Tracing', 'Quality of log services','Ease of arranging shipments','Customs','Timeliness','Infrastructure' ]
    cor_df=pd.DataFrame(data=corr_dict, index=corr_index_names)

    st.write('Correlation between Food Loss percentages and LPI Categories for all datapoints.')
    st.write(cor_df)


########################################
# Regression
########################################

#### Prepare Data for Income Goup Regressions
def Income_group_data(Income_Group):
    df_income_group=df_reg[df_reg['Income']==f'{Income_Group}']
    return df_income_group



def build_model(model_data):
    X=model_data.drop([
        "loss_percentage",
        'country',
        'Logistics performance index: Overall score (1=low to 5=high)',
        "Logistics performance index: Overall score (1=low to 5=high), lower bound",
        "Logistics performance index: Overall score (1=low to 5=high), upper bound",
        'Region',
        'Income',
        'Commodity Groups',
        'food_supply_stage'
        ], axis=1).values
    y=model_data["loss_percentage"]

    names=model_data.drop([
        "loss_percentage",
        'country',
        'Logistics performance index: Overall score (1=low to 5=high)',
        "Logistics performance index: Overall score (1=low to 5=high), lower bound",
        "Logistics performance index: Overall score (1=low to 5=high), upper bound",
        'Region',
        'Income',
        'Commodity Groups',
        'food_supply_stage'
        ], axis=1).columns

    reg = LinearRegression()
    Regression_model=reg.fit(X,y)


    #print(Regression_model.intercept_)
    global df_reg_values
    df_reg_values=pd.DataFrame(list(Regression_model.coef_))
    df_reg_values.index=names

LPI_cat_names=[
    'Ability to track and trace consignments, score (1=low to 5=high)',
    'Competence and quality of logistics services, score (1=low to 5=high)',
    'Ease of arranging competitively priced international shipments, score (1=low to 5=high)',
    'Efficiency of the clearance process, score (1=low to 5=high)',
    'Frequency with which shipments reach consignee within scheduled or expected time, score (1=low to 5=high)',
    'Quality of trade- and transport-related infrastructure, score (1=low to 5=high)'
]

Income_Groups=[
    'Low-Income Economies',
    'Lower-Middle Income Economies',
    'Upper-Middle-Income Economies',
    'High-Income Economies'
    ]

def perform_Normalitytest(country,actual_income_group, nearest_income_group):
        pval=0.05
        #Country
        df_normality_test_country=pd.DataFrame()
        df_normality_test_country.index=['statistic','pval']
        for i in range(len(LPI_cat)): 
            df_normality_test_country_LPI_cat=pd.DataFrame(stats.shapiro(df_reg[df_reg['country']==country][LPI_cat[i]]))
            df_normality_test_country_LPI_cat.index=['statistic','pval']
            df_normality_test_country_LPI_cat.columns=[f'{LPI_cat[i]}']
            df_normality_test_country=df_normality_test_country.join(df_normality_test_country_LPI_cat,lsuffix='_Country', rsuffix='_Income_group')
        df_normality_test_country=df_normality_test_country.transpose() 
        df_normality_test_country['Normality_check']=(df_normality_test_country['pval']>=pval)

        #Actual Income Group
        df_normality_test_actual_income=pd.DataFrame()
        df_normality_test_actual_income.index=['statistic','pval']
        for i in range(len(LPI_cat)): 
            df_normality_test_actual_income_LPI_cat=pd.DataFrame(stats.shapiro(df_reg[df_reg['Income']==actual_income_group][LPI_cat[i]]))
            df_normality_test_actual_income_LPI_cat.index=['statistic','pval']
            df_normality_test_actual_income_LPI_cat.columns=[f'{LPI_cat[i]}']
            df_normality_test_actual_income=df_normality_test_actual_income.join(df_normality_test_actual_income_LPI_cat,lsuffix='_Country', rsuffix='_Income_group')
        df_normality_test_actual_income=df_normality_test_actual_income.transpose() 
        df_normality_test_actual_income['Normality_check']=(df_normality_test_actual_income['pval']>=pval)

        #Nearest Income Group
        df_normality_test_nearest_income=pd.DataFrame()
        df_normality_test_nearest_income.index=['statistic','pval']
        for i in range(len(LPI_cat)): 
            df_normality_test_nearest_income_LPI_cat=pd.DataFrame(stats.shapiro(df_reg[df_reg['Income']==nearest_income_group][LPI_cat[i]]))
            df_normality_test_nearest_income_LPI_cat.index=['statistic','pval']
            df_normality_test_nearest_income_LPI_cat.columns=[f'{LPI_cat[i]}']
            df_normality_test_nearest_income=df_normality_test_nearest_income.join(df_normality_test_nearest_income_LPI_cat,lsuffix='_Country', rsuffix='_Income_group')
        df_normality_test_nearest_income=df_normality_test_nearest_income.transpose() 
        df_normality_test_nearest_income['Normality_check']=(df_normality_test_nearest_income['pval']>=pval)

        return df_normality_test_country,df_normality_test_actual_income, df_normality_test_nearest_income

def perform_Ttest(country,actual_income_group, nearest_income_group):
        pval=0.05
        #Actual_Income & Country
        df_ttest_actual_income=pd.DataFrame()
        df_ttest_actual_income.index=['statistic','pval']
        for i in range(len(LPI_cat)): 
            df_ttest_actual_income_LPI_cat=pd.DataFrame(stats.ttest_ind(df_reg[df_reg['country']==country][LPI_cat[i]],df_reg[df_reg['Income']==actual_income_group][LPI_cat[i]],equal_var=False))
            df_ttest_actual_income_LPI_cat.index=['statistic','pval']
            df_ttest_actual_income_LPI_cat.columns=[f'{LPI_cat[i]}']
            df_ttest_actual_income=df_ttest_actual_income.join(df_ttest_actual_income_LPI_cat,lsuffix='_Country', rsuffix='_Income_group')
        df_ttest_actual_income=df_ttest_actual_income.transpose() 
        df_ttest_actual_income['Ttest_check']=(df_ttest_actual_income['pval']<pval)

        #Nearest_Income & Country
        df_ttest_nearest_income=pd.DataFrame()
        df_ttest_nearest_income.index=['statistic','pval']
        for i in range(len(LPI_cat)): 
            df_ttest_nearest_income_LPI_cat=pd.DataFrame(stats.ttest_ind(df_reg[df_reg['country']==country][LPI_cat[i]],df_reg[df_reg['Income']==nearest_income_group][LPI_cat[i]],equal_var=False))
            df_ttest_nearest_income_LPI_cat.index=['statistic','pval']
            df_ttest_nearest_income_LPI_cat.columns=[f'{LPI_cat[i]}']
            df_ttest_nearest_income=df_ttest_nearest_income.join(df_ttest_nearest_income_LPI_cat,lsuffix='_Country', rsuffix='_Income_group')
        df_ttest_nearest_income=df_ttest_nearest_income.transpose() 
        df_ttest_nearest_income['Ttest_check']=(df_ttest_nearest_income['pval']<pval)
        return df_ttest_actual_income, df_ttest_nearest_income


#################
all_countries=df_reg['country'].unique()
all_countries_results=pd.DataFrame(columns=['True Income Group', 'closest Income Group', 'income Group equal','chosen Datapoints']) #index=all_countries
st.write(all_countries_results)
#################
st.write(len(all_countries))
num_country=0
for i in range(len(all_countries)):
    
    selected_country= all_countries[i]
    
    # filter df for dependent Income-Group
    selected_income_group=df_reg[df_reg['country']==selected_country]['Income'].unique()
    selected_income_group=str(selected_income_group)[2:-2]

    # filter df for dependent Region
    # selected_region=df_reg[df_reg['country']==selected_country]['Region'].unique()
    # selected_region=str(selected_region)[2:-2]


    df_reg_income_total=pd.DataFrame()
    df_reg_income_total.index=LPI_cat_names
    for i in range(len(Income_Groups)):
        reg_data=Income_group_data(Income_Groups[i])
        build_model(reg_data)
        df_reg_values.columns=[Income_Groups[i]]
        df_reg_income_total=df_reg_income_total.join(df_reg_values)
    # st.write('Datatable with Regression Coef. per Income Group:')
    # st.dataframe(df_reg_income_total)

    build_model(df_reg[df_reg['country']==selected_country])
    df_reg_country=pd.DataFrame()
    df_reg_country.index=LPI_cat_names
    df_reg_country=df_reg_country.join(df_reg_values)
    df_reg_country.columns=[f'{selected_country}']
    # st.write('country regression')
    # st.dataframe(df_reg_country)

    df_for_Means=df_reg
    LPI_cat=[
        'Ability to track and trace consignments, score (1=low to 5=high)',
        'Competence and quality of logistics services, score (1=low to 5=high)',
        'Ease of arranging competitively priced international shipments, score (1=low to 5=high)',
        'Efficiency of the clearance process, score (1=low to 5=high)',
        'Frequency with which shipments reach consignee within scheduled or expected time, score (1=low to 5=high)',
        'Quality of trade- and transport-related infrastructure, score (1=low to 5=high)'
        ]

    for i in [0,1,2,3,4,6,7,8,14]:
        df_for_Means.iloc[:,i]=pd.to_numeric(df_for_Means.iloc[:,i])
    for i in [9,10,11,12,13]:
        df_for_Means.iloc[:,i]=df_for_Means.iloc[:,i].astype("category")

    Income_mean=pd.pivot_table(df_reg,index=['Income'], values=LPI_cat,aggfunc=np.mean, fill_value=None)
    Income_mean=Income_mean.drop(index=['Nan'])
    # st.write('Mean per Income Group')
    # st.dataframe(Income_mean)


    df_selected_country=df_for_Means[df_for_Means['country']==selected_country]
    Country_mean=pd.pivot_table(df_selected_country,index=['country'], values=LPI_cat,aggfunc=np.mean, fill_value=None)
    # st.write('Mean of Country')
    # st.dataframe(Country_mean)

    # st.write('Diff. of Means (Country & Income Groups)')
    Diff_Mean_Income= Income_mean.sub(Country_mean.iloc[0, :])
    Diff_Mean_Income['Abs_Sum_Mean_Diff']= abs(Diff_Mean_Income.sum(axis = 1))
    Diff_Mean_Income=Diff_Mean_Income.sort_values('Abs_Sum_Mean_Diff', ascending=True)
    # st.dataframe(Diff_Mean_Income)


    dict_comparison={'Actual Group': [df_selected_country['Income'].value_counts().idxmax()],'Nearest Group': [Diff_Mean_Income.index[0]]}#,df_selected_country['Region'].value_counts().idxmax(),df_selected_country['Commodity Groups'].value_counts().idxmax(),df_selected_country['food_supply_stage'].value_counts().idxmax()], 'Nearest Group': [Diff_Mean_Income.index[0],Diff_Mean_Region.index[0],Diff_Mean_Commodity_group.index[0],Diff_Mean_Supply_stage.index[0]]}
    df_comparison=pd.DataFrame(dict_comparison)
    df_comparison.index=['Income Group']#, 'Region', 'Commodity Group', 'Food Supply Stage']
    df_comparison['is_equal']=(df_comparison['Nearest Group']==df_comparison['Actual Group'])



    # st.write('Is Actual Income Group in LPI-Categories also the closest?')
    # st.dataframe(df_comparison)


    # with st.expander('Details Regression'):
    #     st.write('Datatable with Regression Coef. per Income Group:')
    #     st.dataframe(df_reg_income_total)

    #     st.write('country regression')
    #     st.dataframe(df_reg_country)

    # with st.expander('Details Mean Comparison'):
    #     st.write('Mean per Income Group')
    #     st.dataframe(Income_mean)

    #     st.write('Mean of Country')
    #     st.dataframe(Country_mean)

    #     st.write('Diff. of Means (Country & Income Groups)')
    #     st.dataframe(Diff_Mean_Income)

    #     st.write('Is Actual Income Group in LPI-Categories also the closest?')
    #     st.dataframe(df_comparison)

    ################
    #TEST STATISTICS
    ################
    # st.write('Tests')

    try:
        df_normality_test_country, df_normality_test_actual_income, df_normality_test_nearest_income =perform_Normalitytest(selected_country, selected_income_group,df_comparison.iloc[0,1])
    except:
        df_normality_test_country=[]
        df_normality_test_actual_income=[]
        df_normality_test_nearest_income=[]
        #st.info(f'{selected_country} does not have sufficient datapoints to run the model properly.')
    # st.write('Normality Test Country')
    # st.dataframe(df_normality_test_country)
    # st.write('Normality Test Actual Income Group')
    # st.dataframe(df_normality_test_actual_income)
    # st.write('Normality Test Nearest Income Group')
    # st.dataframe(df_normality_test_nearest_income)

    try:
        df_ttest_actual_income, df_ttest_nearest_income =perform_Ttest(selected_country, selected_income_group,df_comparison.iloc[0,1])
    except:
        df_ttest_actual_income=[]
        df_ttest_nearest_income=[]
        #st.info(f'{selected_country} does not have sufficient datapoints to run the model properly.')


    # st.write('Ttest Actual Income Group')
    # st.dataframe(df_ttest_actual_income)
    # st.write('Ttest Nearest Income Group')
    # st.dataframe(df_ttest_nearest_income)





    # with st.expander('Details Tests'):
    #     st.write('Normality Test Country')
    #     st.dataframe(df_normality_test_country)
    #     st.write('Normality Test Actual Income Group')
    #     st.dataframe(df_normality_test_actual_income)
    #     st.write('Normality Test Nearest Income Group')
    #     st.dataframe(df_normality_test_nearest_income)
    #     st.write('Ttest Actual Income Group')
    #     st.dataframe(df_ttest_actual_income)
    #     st.write('Ttest Nearest Income Group')
    #     st.dataframe(df_ttest_nearest_income)



    ################
    #LPI-category Selection
    ################


    Income_group_actual=selected_income_group
    nearest_income_group=df_comparison.iloc[0,1]
    LPI_cat_order_country=df_reg_country[df_reg_country[f'{selected_country}']<0]

    #Auswahl ob Regression country oder Income-group gewählt wird und welche Income Group gewählt wird.
    datapoints=len(df_reg[df_reg['country']==selected_country])
    num_country+=1
    #st.write(num_country)
    country_results=pd.DataFrame({'True Income Group':[df_comparison.iloc[0,0]], 
                                        'closest Income Group':[df_comparison.iloc[0,1]], 
                                        'income Group equal':[df_comparison.iloc[0,2]],
                                        'chosen Datapoints':['none']
                                        },
                                        index = [selected_country])
    #st.write(country_results)
    if (len(df_reg[df_reg['country']==selected_country])<=5)|(sum(df_reg_country[selected_country])==0):
        if (df_comparison['is_equal'].bool()==True) :
            #LPI_cat_order_country=df_reg_income_total[df_reg_income_total[Income_group_actual]<0][Income_group_actual]
            #st.info(f'There are only {datapoints} datapoints for {selected_country} so its Income-group:"{Income_group_actual}" was used to determine the policy-areas.')
            country_results=pd.DataFrame({'True Income Group':[df_comparison.iloc[0,0]], 
                                        'closest Income Group':[df_comparison.iloc[0,1]], 
                                        'income Group equal':[df_comparison.iloc[0,2]],
                                        'chosen Datapoints':[df_comparison.iloc[0,0]]
                                        },
                                        index = [selected_country])
            
            #pd.concat([all_countries_results, country_results],ignore_index=True)
            # if (sum(df_reg_country[selected_country])==0):
            #st.info(f'Additional Info: {selected_country}s datapoints were not interpretable for the model, so the Income-group data were used instead.')
        if (df_comparison['is_equal'].bool()==False) & (df_ttest_actual_income['Ttest_check'].sum()>=df_ttest_nearest_income['Ttest_check'].sum()):
            #LPI_cat_order_country=df_reg_income_total[df_reg_income_total[nearest_income_group]<0][nearest_income_group]
            #st.info(f'There are only {datapoints} datapoints for {selected_country} so its closest Income-group:"{nearest_income_group}" was used to determine the policy-areas. The actual Income-group is: {Income_group_actual}.')
            country_results=pd.DataFrame({'True Income Group':df_comparison.iloc[0,0], 
                                        'closest Income Group':df_comparison.iloc[0,1], 
                                        'income Group equal':df_comparison.iloc[0,2],
                                        'chosen Datapoints':df_comparison.iloc[0,1]
                                        },
                                        index = [selected_country])
            
            #pd.concat([all_countries_results, country_results],ignore_index=True)
            #if (sum(df_reg_country[selected_country])==0):
                #st.info(f'Additional Info: {selected_country}s datapoints were not interpretable for the model, so the Income-group data were used instead.')
        else:
            #LPI_cat_order_country=df_reg_income_total[df_reg_income_total[nearest_income_group]<0][nearest_income_group]
            #st.info(f'There are only {datapoints} datapoints for {selected_country} so its closest Income-group:"{nearest_income_group}" was used to determine the policy-areas. The actual Income-group is: {Income_group_actual}.')
            country_results=pd.DataFrame({'True Income Group':df_comparison.iloc[0,0], 
                                        'closest Income Group':df_comparison.iloc[0,1], 
                                        'income Group equal':df_comparison.iloc[0,2],
                                        'chosen Datapoints':df_comparison.iloc[0,0]
                                        },
                                        index = [selected_country])
            
            #pd.concat([all_countries_results, country_results],ignore_index=True)
            #if (sum(df_reg_country[selected_country])==0):
                #st.info(f'Additional Info: {selected_country}s datapoints were not interpretable for the model, so the Income-group data were used instead.')
        all_countries_results=all_countries_results.append(country_results)
    else:
        country_results=pd.DataFrame({'True Income Group':[df_comparison.iloc[0,0]], 
                                        'closest Income Group':[df_comparison.iloc[0,1]], 
                                        'income Group equal':[df_comparison.iloc[0,2]],
                                        'chosen Datapoints':[selected_country]
                                        },
                                        index = [selected_country])
        #pd.merge(all_countries_results, country_results, left_index=True, right_index=True)
        all_countries_results=all_countries_results.append(country_results)
        #pd.concat([all_countries_results, country_results],axis=0)
    #st.write(country_results)

    
st.write(all_countries_results)
st.write(len(all_countries_results))
#st.write(all_countries_results)
all_countries_results.to_csv('Policy_assistent_Results.csv',sep=';')

final_evaluation_dict={'High-Income Economies':[16, 10, 0],
                       'Upper-Middle Income Economies':[9, 13, 5],
                       'Lower-MiddleIncome Economies':[4, 15, 8],
                       'Low-Income Economies':[7, 9, 0],
                       }
final_evaluation_dict_dt={'Hohes Einkommen':[16, 10, 0],
                       'Hohes-Mittleres Einkommen':[9, 13, 5],
                       'Niedriges-Mittleres Einkommen':[4, 15, 8],
                       'Niedriges Einkommen':[7, 9, 0],
                       }
final_evaluation=pd.DataFrame(final_evaluation_dict_dt, index=['Land', 'echte Einkommensgruppe', 'ähnlichste Einkommensgruppe'])
st.write(final_evaluation)
final_evaluation=final_evaluation.reset_index()
final_evaluation=final_evaluation.melt(id_vars=['index'])
fig_final_evaluation=px.bar(final_evaluation, x='index', y='value', 
                 color="variable", barmode="group")
fig_final_evaluation.update_layout(xaxis_title='Datenbasis', yaxis_title='Anzahl', title='Säulendiagramm zur Anzahl der Länder je Einkommensgruppe',legend_title_text='Einkommensgruppe', template='seaborn')
st.plotly_chart(fig_final_evaluation, use_container_width=True)


deep_dive_evaluation_dict={'High-Income Economies':[0,1],
                       'Upper-Middle Income Economies':[5,4],
                       'Lower-MiddleIncome Economies':[8,4],
                       'Low-Income Economies':[0,5],
                       }
deep_dive_evaluation_dict_dt={'Hohes Einkommen':[0,1],
                       'Hohes-Mittleres Einkommen':[5,4],
                       'Niedriges-Mittleres Einkommen':[8,4],
                       'Niedriges Einkommen':[0,5],
                       }
deep_dive_evaluation=pd.DataFrame(deep_dive_evaluation_dict_dt, index=['echte Einkommensgruppe', 'ausgewählte Einkommensgruppe'])
st.write(deep_dive_evaluation)
deep_dive_evaluation=deep_dive_evaluation.reset_index()
deep_dive_evaluation=deep_dive_evaluation.melt(id_vars=['index'])
fig_deep_dive_evaluation=px.bar(deep_dive_evaluation, x='index', y='value', 
                 color="variable", barmode="group")
fig_deep_dive_evaluation.update_layout(xaxis_title='', yaxis_title='Anzahl',title='Säulendiagramm zur Änderung der Einkommensgruppe',legend_title_text='Einkommensgruppe', template='seaborn')
st.plotly_chart(fig_deep_dive_evaluation, use_container_width=True)


fig_sankey = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = ["Hohes Einkommen (von)", "Hohes-Mittleres Einkommen (von)", "Niedriges-Mittleres Einkommen (von)", "Niedriges Einkommen (von)","Hohes Einkommen (nach)", "Hohes-Mittleres Einkommen (nach)", "Niedriges-Mittleres Einkommen (nach)", "Niedriges Einkommen (nach)"],
      #label = ["High-Income Economies (1)", "Upper-Mid Income Economies (1)", "Lower-Mid Income Economies (1)", "Low-Income Economies (1)","High-Income Economies (2)", "Upper-Mid Income Economies (2)", "Lower-Mid Income Economies (2)", "Low-Income Economies (2)"],
      color = "blue"
    ),
    link = dict(
      source = [1, 1, 1, 2, 2], # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = [4, 6, 7, 5, 7],
      value = [1, 3, 1, 4, 5]
  ))])

fig_sankey.update_layout(title_text="Sankey-Diagramm des Wechsel zur ähnlichsten Einkommensgruppe", font_size=10)
st.plotly_chart(fig_sankey, use_container_width=True)

# LPI_cat_order_country=pd.DataFrame(LPI_cat_order_country)
# LPI_cat_order_country.columns=[f'{selected_country}']


# LPI_cat_order_country=LPI_cat_order_country.sort_values(f'{selected_country}',ascending=False)
# LPI_cat_order_country['weight']=np.arange(len(LPI_cat_order_country))+1
# LPI_cat_order_country['weight']=pd.to_numeric(LPI_cat_order_country['weight'])
# LPI_cat_order_country=LPI_cat_order_country.sort_values(by=['weight'], ascending = False)
# LPI_cat_order_country=LPI_cat_order_country.reset_index()
# LPI_cat_order_country.columns=['LPI-Category ordered by Priority (highest first)','Regression coefficent', 'Priority']

# LPI_cat_name={
#     'Ability to track and trace consignments, score (1=low to 5=high)':'Tracking and Tracing',
#     'Competence and quality of logistics services, score (1=low to 5=high)':'Quality of log services',
#     'Ease of arranging competitively priced international shipments, score (1=low to 5=high)':'Ease of arranging shipments',
#     'Efficiency of the clearance process, score (1=low to 5=high)':'Customs',
#     'Frequency with which shipments reach consignee within scheduled or expected time, score (1=low to 5=high)':'Timeliness',
#     'Quality of trade- and transport-related infrastructure, score (1=low to 5=high)':'Infrastructure'
#     }
# LPI_cat_order_country=LPI_cat_order_country.replace({'LPI-Category ordered by Priority (highest first)':LPI_cat_name})
# with st.expander('Details LPI-categories'):
#     st.write('Identified policy areas to reduce Food Loss')
#     st.dataframe(LPI_cat_order_country['LPI-Category ordered by Priority (highest first)'])







# ################
# #Factor Analysis
# ################
# data = pd.read_csv('data/CFA_Policies.csv',sep=';')
# LPI_categories=data['LPI-category'].unique()

# #Functions to avoid singular matrix error
# mu=0.0
# std = 0.1
# np.random.seed(21389712) #repeatable random number
# def gaussian_noise(x):
#     noise = np.random.normal(mu, std, size = x.shape)
#     x_noisy = x + noise
#     return x_noisy 

# def make_nonsingular(matrix):
#   n = matrix.shape[0]
#   m = matrix.shape[1]
#   for i in range(n):
#     for j in range(m):
#       matrix.iloc[i, j] = gaussian_noise(matrix.iloc[i, j])
#   return matrix

# #function to perform Faktor Analysis

# # import semopy
# # from semopy import Model

# def factor_analysis(LPI_cat):
#     LPI_category_data=data[data['LPI-category']==LPI_cat]
#     X=LPI_category_data[['survey_1','survey_2','survey_3','survey_4','survey_5']]
#     # Create factor analysis object and perform factor analysis
#     fa = FactorAnalyzer(rotation=None, n_factors=1)

#     #add noise to each value to avoid a singular matrix
#     X=make_nonsingular(X)
#     fa.fit(X)
#     with st.expander(f'See the factoranalysis insights of {LPI_cat}'):
#         st.write(f"{LPI_cat}: Chi-value, p-value ", calculate_bartlett_sphericity(X))
#         st.write(f"{LPI_cat}: Kaiser-Meyer-Olkin criterion per variable, in total ", calculate_kmo(X))

#     # model = Model("example_model", manifest_variables=['survey_1','survey_2','survey_3','survey_4','survey_5'], latent_variables=['factor1'], paths=[('factor1', 'survey_1'), ('factor1', 'survey_2'), ('factor1', 'survey_3'),('factor1', 'survey_4'),('factor1', 'survey_5')])
#     # fit = model.fit(X)
#     # stats = semopy.calc_stats(model)
#     # st.write(stats.T)
#     #returned are Factor loadings
#     return fa.transform(X)

#     # Get results
#     # fa.loadings_
#     # fa.get_communalities()
#     # fa.get_eigenvalues()
#     # fa.get_factor_variance()
#     # fa.get_uniquenesses()





# #run function for each LPI-categorie
# Factor_Loadings=pd.DataFrame()
# for i in range(len(LPI_categories)-1):
#     Factor_Loadings[LPI_categories[i]]=pd.DataFrame(factor_analysis(LPI_categories[i]))

# #get tidy data    
# Factor_Loadings=Factor_Loadings.melt(ignore_index=False).dropna()
# Factor_Loadings.columns=['LPI-category', 'factor_loading']

# #factor loading for Timeliness is set to 1
# Factor_Loadings=Factor_Loadings.append({'LPI-category': 'Timeliness', 'factor_loading':1}, ignore_index=True)
# data['factor_loading']=Factor_Loadings['factor_loading']

# st.subheader(f'List of policy-measures to reduce Food Loss in {selected_country}')
# for i in range(len(LPI_cat_order_country['LPI-Category ordered by Priority (highest first)'])):
#     LPI_cat = LPI_cat_order_country['LPI-Category ordered by Priority (highest first)'][i]
#     data_LPI_cat = data[data['LPI-category']==LPI_cat]
#     data_LPI_cat = pd.DataFrame(data_LPI_cat).sort_values(by=['factor_loading'],ascending=False)
#     data_LPI_cat = data_LPI_cat.fillna('not found in literature')
#     if i ==0: 
#         st.write(f'1st Priority: **{LPI_cat}**.   See its policy-measures (highest priority first) below in order to improve in this area.')
#     if i ==1: 
#         st.write(f'2nd Priority: **{LPI_cat}**.   See its policy-measures (highest priority first) below in order to improve in this area.')
#     if i ==2: 
#         st.write(f'3rd Priority: **{LPI_cat}**.   See its policy-measures (highest priority first) below in order to improve in this area.')
#     if i ==3: 
#         st.write(f'4th Priority: **{LPI_cat}**.   See its policy-measures (highest priority first) below in order to improve in this area.')
#     if i ==4: 
#         st.write(f'5th Priority: **{LPI_cat}**.   See its policy-measures (highest priority first) below in order to improve in this area.')
#     if i ==5: 
#         st.write(f'6th Priority: **{LPI_cat}**.   See its policy-measures (highest priority first) below in order to improve in this area.')
#     st.write(data_LPI_cat[['Policy','Benefits','Risks']].set_index('Policy'))

























# LPI_cat_order=df_reg_income_total[df_reg_income_total[Income_group_actual]<0][Income_group_actual]
# LPI_cat_order=LPI_cat_order.sort_values(ascending=False)
# LPI_cat_order=pd.DataFrame(LPI_cat_order)
# LPI_cat_order['Weights']=np.arange(len(LPI_cat_order))+1
# LPI_cat_order['Weights']=pd.to_numeric(LPI_cat_order['Weights'])
# LPI_cat_order=LPI_cat_order.reset_index()
# LPI_cat_order.columns=['category','Regression_Coef', 'weight']
# selected_Commodity_Group = df_selected_country['Commodity Groups'].value_counts().idxmax()
# selected_supply_stage = df_selected_country['food_supply_stage'].value_counts().idxmax()
# df_cat_weights=LPI_cat_order[['category','weight']]
# df_cat_weights=df_cat_weights.append(LPI_cat_order_country[['category','weight']])
# df_cat_weights.loc[len(df_cat_weights.index)]=[selected_region, 1]
# df_cat_weights.loc[len(df_cat_weights.index)]=[selected_supply_stage, 1]
# df_cat_weights.loc[len(df_cat_weights.index)]=[selected_Commodity_Group, 1]
# df_cat_weights=df_cat_weights.groupby(['category']).sum()
# df_cat_weights=df_cat_weights.reset_index()
# st.dataframe(df_cat_weights)
# #st.dataframe(df_policies)

# df_policies_test=df_policies
# for i in [0,4,6-38]:
#     df_policies_test.iloc[:,i]=pd.to_numeric(df_policies_test.iloc[:,i])

# for i in range(len(df_cat_weights)):
#     df_policies_test[df_cat_weights.iloc[i,0]]+=df_cat_weights.iloc[i,1]

# df_policies_test=df_policies_test.replace(1,0)
# df_policies_test['Total weight']=df_policies_test.iloc[:,4:38].sum(axis = 1)
# df_policies_test=df_policies_test.sort_values('Total weight', ascending=False)


# df_policy_recommendation=df_policies_test[['Policy Short', 'Policy Component', 'Policy Area', 'Total weight']]
# st.dataframe(df_policy_recommendation)





