########################################
# Import libraries
########################################
import streamlit as st
import pandas as pd
from scipy import stats
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression


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
df_LPI= pd.read_csv("data\LPI_Data.csv", encoding='latin-1')
# FAO Data
df_FAO = pd.read_csv("data\Data.csv")
# Country Categories by Worldbank 
country_categories_tidy= pd.read_csv('data/country_categories_tidy.csv', delimiter=';')
# Country Categories by Worldbank 
commodity_groups_tidy= pd.read_csv('data/commodity_groups.csv', delimiter=';')
# LPI Country Name Changes
df_LPI_new_country_names= pd.read_csv("data\LPI_Countries_name_changes.csv", delimiter=';')
# FAO Country Name Changes
df_FAO_new_country_names= pd.read_csv("data\FAO_Countries_name_changes.csv", delimiter=';')
# FAO Country Name Changes
df_policies= pd.read_csv("data\Policies.csv", delimiter=';')

########################################
# Clean Data 
# Change Data Types
########################################
#align country names for later merge
def Change_Country_Names( df_new_names,column_new_names ,df_to_change,column_name):
    name_dict=df_new_names.set_index(column_new_names).T.to_dict('list')
    df_to_change[column_name]=df_to_change[column_name].map(name_dict)
    df_to_change[column_name]=df_to_change[column_name].str.get(0)

Change_Country_Names(df_LPI_new_country_names,'LPI Countries',df_LPI, 'Country Name' )
Change_Country_Names(df_FAO_new_country_names,'FAO Countries',df_FAO, 'country' )
df_LPI=df_LPI[df_LPI['Country Name']!='Guinea']

# select columns from LPI-data
df_LPI_reg=df_LPI[["Country Name","Series Name","2007 [YR2007]","2010 [YR2010]","2012 [YR2012]","2014 [YR2014]","2016 [YR2016]","2018 [YR2018]"]]

# rename columns
df_LPI_reg.columns=["Country Name","Series Name","2007","2010","2012","2014","2016","2018"]

# select only score data from LPI --> Adjustable in Streamlit
df_LPI_reg=df_LPI_reg[df_LPI_reg["Series Name"].str.contains('score',na=False)]

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
## adding FAO Food Loss Percentage to the Dataframe
# selectable food supply-stage --> Adjustable in Streamlit
#food_supply_stage=['Whole supply chain']#,'Harvest']

# filter orginial FAO Data for selected supply-stage
#df_FAO = df_FAO[df_FAO['food_supply_stage'].isin(food_supply_stage)]

# prepare FAO Data for merge
df_FAO_pivot=pd.pivot_table(df_FAO,index=['country','year','commodity','food_supply_stage'], values='loss_percentage',aggfunc=np.mean, fill_value=None)
df_FAO_pivot=df_FAO_pivot.reset_index()

# merge filtered data of both data source
df_reg=pd.merge(y,df_FAO_pivot,how='inner', left_on=['Country Name','year'],right_on=['country','year'])
df_reg=df_reg.replace("..",np.NaN)

## Add Country Categories to the Dataframe
df_reg=pd.merge(df_reg,country_categories_tidy, how='inner', left_on='country', right_on='Country')
df_reg=df_reg.drop(columns=['Country'])
df_reg=df_reg.ffill(axis = 0)

## Add Country Categories to the Dataframe
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
st.dataframe(df_reg)




# test=df_reg.groupby(['year','country','Commodity Groups']).agg({'loss_percentage': ['mean']})
# commodity_country_ranking=df_reg.groupby(['country','commodity']).agg({'loss_percentage': ['mean', 'max', 'min','count']})
########################################
# Test Statistic
########################################

dict_income_group={
    'UPPER-MIDDLE-INCOME ECONOMIES ($4,256 TO $13,205)':'Upper-Middle-Income Economies',
    'LOW-INCOME ECONOMIES ($1,085 OR LESS) ':'Low-Income Economies',
    'LOWER-MIDDLE INCOME ECONOMIES ($1,086 TO $4,255)':'Lower-Middle Income Economies',
    'HIGH-INCOME ECONOMIES ($13,205 OR MORE) ':'High-Income Economies'
}

df_reg['Income']=df_reg['Income'].replace(dict_income_group)

# select Country
selected_country=st.selectbox('Select country',options=df_reg['country'].unique())

# filter df for dependent Income-Group
selected_income_group=df_reg[df_reg['country']==selected_country]['Income'].unique()
selected_income_group=str(selected_income_group)[2:-2]
# st.write('Dataframe of Income Group excluding selected Country')
# st.dataframe(df_reg[(df_reg['Income']==f'{selected_income_group}')& (df_reg['country']!=selected_country)])

# filter df for dependent Region
selected_region=df_reg[df_reg['country']==selected_country]['Region'].unique()
selected_region=str(selected_region)[2:-2]
# st.write('Dataframe of Region excluding selected Country')
# st.dataframe(df_reg[(df_reg['Region']==f'{selected_region}')& (df_reg['country']!=selected_country)])
st.markdown(f'**Income Group:** {selected_income_group}')
st.markdown(f'**Region:** {selected_region}')
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


st.subheader('Test Statistic')

selected_series=st.selectbox('Select series',options=dict_series.keys())



test_df_1=df_reg[df_reg['country']==selected_country]
test_df_2=df_reg[(df_reg['Income']==f'{selected_income_group}')]#& (df_reg['country']!=selected_country)]
test_df_3=df_reg[(df_reg['Region']==f'{selected_region}')]#& (df_reg['country']!=selected_country)]

test_data_1=test_df_1.iloc[:,dict_series[selected_series]]
test_data_1=test_data_1.astype('float')

test_data_2=test_df_2.iloc[:,dict_series[selected_series]]
test_data_2=test_data_2.astype('float')

test_data_3=test_df_3.iloc[:,dict_series[selected_series]]
test_data_3=test_data_3.astype('float')


#describe Dataframes
df_describe=pd.DataFrame(test_data_1.describe())
df_describe=df_describe.join(pd.DataFrame(test_data_2.describe()),lsuffix='_Country', rsuffix='_Income_group')
df_describe=df_describe.join(pd.DataFrame(test_data_3.describe()),lsuffix='_Country', rsuffix='_Region_group')
df_describe.columns=['Country','Income Group','Region']
with st.expander('See Description of Selected Data'):
    col1, col2 = st.columns(2)
    with col1: 
        st.write('Data Description')
        st.dataframe(df_describe)
    with col2: 
        st.write('Histograms')
        tab1, tab2, tab3 = st.tabs(["Country", "Income Group", "Region"])
        with tab1:
            fig1 = px.histogram(test_data_1,nbins=20, x=selected_series, labels={selected_series:'Country Data Distribution', 'y':'count'})
            fig1.update_layout({
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'paper_bgcolor': 'rgba(0,0,0,0)'
                })
            st.plotly_chart(fig1, use_container_width=True)
        with tab2:
            fig2 = px.histogram(test_data_2,nbins=20, x=selected_series, labels={selected_series:'Income Group Data Distribution', 'y':'count'})
            fig2.update_layout({
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'paper_bgcolor': 'rgba(0,0,0,0)'
                })
            st.plotly_chart(fig2, use_container_width=True)
        with tab3:
            fig3 = px.histogram(test_data_3,nbins=20, x=selected_series, labels={selected_series:'Region Data Distribution', 'y':'count'})
            fig3.update_layout({
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'paper_bgcolor': 'rgba(0,0,0,0)'
                })
            st.plotly_chart(fig3, use_container_width=True)


#Normality Test results
normality_test_results=pd.DataFrame(stats.shapiro(test_data_1))
normality_test_results=normality_test_results.join(pd.DataFrame(stats.shapiro(test_data_2)),lsuffix='_Country', rsuffix='_Income_group')
normality_test_results=normality_test_results.join(pd.DataFrame(stats.shapiro(test_data_3)),lsuffix='_Country', rsuffix='_Region_group')
normality_test_results.index=['statistic','pvalue']
normality_test_results=normality_test_results.transpose()
normality_test_results.index=['Country','Income Group','Region']
normality_test_results['Normality condition']=(normality_test_results['pvalue'] >= 0.000005)
# print(normality_test_results)
col3, col4 = st.columns(2)
with col3:
    st.markdown('**Normality Test Results**')
    st.dataframe(normality_test_results)
    with st.expander('See Test Hypothesis'):
        st.markdown(f'''
            **H0**: Die Grundgesamtheit von Land/Income-Group/Region sind normalverteilt.

            **H1**: Die Grundgesamtheit von Land/Income-Group/Region sind nicht normalverteilt. 
        ''')


#T Test results
Ttest_results=pd.DataFrame(stats.ttest_ind(test_data_1, test_data_2, equal_var=False))
Ttest_results=Ttest_results.join(pd.DataFrame(stats.ttest_ind(test_data_1, test_data_3, equal_var=False)),lsuffix='_Country/Region', rsuffix='_Country/Income')
Ttest_results.index=['statistic','pvalue']
Ttest_results=Ttest_results.transpose()
Ttest_results.index=['Country/Region','Country/Income Group']
Ttest_results['Ttest Check (pval<0.05)']=(Ttest_results['pvalue']<= 0.05)
# print(Ttest_results)
with col4:
    st.markdown('**T-Test Results**')
    st.dataframe(Ttest_results)
    st.write('')
    st.write('')
    with st.expander('See Test Hypothesis'):
        st.markdown(f'''
        **H0**: Es gibt **keinen** Mittelwertsunterschied zwischen gewähltem Land und dessen zugehöriger Income-Gruppe/Region.

        **H1**: Es gibt **einen** Mittelwertsunterschied zwischen gewähltem Land und dessen zugehöriger Income-Gruppe/Region. 
        ''')
st.markdown(f'''
        Sind alle Vorraussetzungen des T-Tests erfüllt (inkl. *Normalverteilung der Stichproben*) kann folgende Aussage getroffen werden.

        Ist der p-Wert (pvalue) kleiner 0,05 , dem festgelegten Signifikanzlevel, so kann die Null-Hypothese abgelehnt werden.
        
        In diesem Fall existiert ein Mittelwertsunterschied zwischen dem gewählten Land (*{selected_country}*) und seiner Income-Group(*{selected_income_group}*) /Region(*{selected_region}*)''')


########################################
# Create Model
########################################
def build_model_visualisation(model_data):
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
    #X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=21, stratify=y)
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
    bar_chart_df=pd.DataFrame(list(Regression_model.coef_))
    bar_chart_df.index=names
    bar_chart_df.columns=['Coefficient Value']
    st.bar_chart(bar_chart_df)

st.subheader('Linear Regression')
col5,col6,col7=st.columns(3)
with col5:
    st.write(f'''
    **{selected_country}** 
    specific models coefficients
    ''')
    build_model_visualisation(test_df_1)
with col6:
    st.write(f'''
    **{selected_income_group}** 
    specific models coefficients
    ''')

    build_model_visualisation(test_df_2)
with col7:
    st.write(f'''
    **{selected_region}** 
    specific models coefficients
    ''')
    build_model_visualisation(test_df_3)

st.subheader('Income Groups Linear Regression')

#### Prepare Data for Income Goup Regressions
def Income_group_data(Income_Group):
    df_income_group=df_reg[df_reg['Income']==f'{Income_Group}']
    return df_income_group


col8,col9,col10,col11=st.columns(4)
with col8:
    st.write(f'''
    **Low Income** 
    specific models coefficients
    ''')
    Income_Group='Low-Income Economies'
    reg_data=Income_group_data(Income_Group)
    build_model_visualisation(reg_data)

with col9:
    st.write(f'''
    **Lower-Middle Income** 
    specific models coefficients
    ''')
    Income_Group='Lower-Middle Income Economies'
    reg_data=Income_group_data(Income_Group)
    build_model_visualisation(reg_data)

with col10:
    st.write(f'''
    **Upper-Middle Income** 
    specific models coefficients
    ''')
    Income_Group='Upper-Middle-Income Economies'
    reg_data=Income_group_data(Income_Group)
    build_model_visualisation(reg_data)

with col11:
    st.write(f'''
    **High Income** 
    specific models coefficients
    ''')
    Income_Group='High-Income Economies'
    reg_data=Income_group_data(Income_Group)
    build_model_visualisation(reg_data)


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
    #X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=21, stratify=y)
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

df_reg_income_total=pd.DataFrame()
df_reg_income_total.index=LPI_cat_names
for i in range(len(Income_Groups)):
    reg_data=Income_group_data(Income_Groups[i])
    build_model(reg_data)
    df_reg_values.columns=[Income_Groups[i]]
    df_reg_income_total=df_reg_income_total.join(df_reg_values)
st.write('Datatable with Regression Coef. per Income Group:')
st.dataframe(df_reg_income_total)

#### Prepare Data for Commodity Group Regressions
def Commodity_group_data(Commodity_Group):
    df_Commodity_group=df_reg[df_reg['Commodity Groups']==f'{Commodity_Group}']
    return df_Commodity_group

commodity_groups=[
    'Cereals (excluding beer)',
    'Roots and Tubers',
    'Oilseeds and Pulses (including nuts)',
    'Fruit and Vegetables (including bananas)',
    'Meat',
    'Dairy products'
    ]

df_reg_commodity_total=pd.DataFrame()
df_reg_commodity_total.index=LPI_cat_names
for i in range(len(commodity_groups)):
    reg_data=Commodity_group_data(commodity_groups[i])
    build_model(reg_data)
    df_reg_values.columns=[commodity_groups[i]]
    df_reg_commodity_total=df_reg_commodity_total.join(df_reg_values)
st.write('Datatable with Regression Coef. per Commodity Group:')
st.dataframe(df_reg_commodity_total)

#### Prepare Data for Region  Regressions

def Region_group_data(Region):
    df_Region_group=df_reg[df_reg['Region']==f'{Region}']
    return df_Region_group

Region_groups=[
    'SOUTH ASIA',
    'SUB-SAHARAN AFRICA',
    'EUROPE AND CENTRAL ASIA',
    'NORTH AMERICA',
    'MIDDLE EAST AND NORTH AFRICA',
    'EAST ASIA AND PACIFIC',
    'LATIN AMERICA AND THE CARIBBEAN',
]

df_reg_Region_total=pd.DataFrame()
df_reg_Region_total.index=LPI_cat_names
for i in range(len(Region_groups)):
    reg_data=Region_group_data(Region_groups[i])
    build_model(reg_data)
    df_reg_values.columns=[Region_groups[i]]
    df_reg_Region_total=df_reg_Region_total.join(df_reg_values)
st.write('Datatable with Regression Coef. per Region:')
st.dataframe(df_reg_Region_total)




#### Prepare Data for Region  Regressions
def food_supply_stage_data(food_supply_stage):
    df_food_supply_stage=df_reg[df_reg['food_supply_stage']==f'{food_supply_stage}']
    return df_food_supply_stage

food_supply_stage= [
    'Whole supply chain',
    #'Pre-harvest',
    'Harvest',
    'Post-harvest',
    'Farm',
    #'Grading',
    #'Stacking',
    'Storage',
    'Transport',
    #'Distribution',
    'Processing',
    #'Packing',
    'Wholesale',
    #'Export',
    'Trader',
    #'Market',
    'Retail',
    'Food Services',
    'Households',    
]

df_reg_food_supply_stage_total=pd.DataFrame()
df_reg_food_supply_stage_total.index=LPI_cat_names
for i in range(len(food_supply_stage)):
    reg_data=food_supply_stage_data(food_supply_stage[i])
    build_model(reg_data)
    df_reg_values.columns=[food_supply_stage[i]]
    df_reg_food_supply_stage_total=df_reg_food_supply_stage_total.join(df_reg_values)
st.write('Datatable with Regression Coef. per Supply Chain stage:')
st.dataframe(df_reg_food_supply_stage_total)



########################################
# Create MEAN Comparisons
########################################

# Country: selected_country

# test_df_1=df_reg[df_reg['country']==selected_country]
# test_df_2=df_reg[(df_reg['Income']==f'{selected_income_group}')]#& (df_reg['country']!=selected_country)]
# test_df_3=df_reg[(df_reg['Region']==f'{selected_region}')]#& (df_reg['country']!=selected_country)]

test=df_reg#[df_reg.index==max(df_reg.index)]
values=[
    'Ability to track and trace consignments, score (1=low to 5=high)',
    'Competence and quality of logistics services, score (1=low to 5=high)',
    'Ease of arranging competitively priced international shipments, score (1=low to 5=high)',
    'Efficiency of the clearance process, score (1=low to 5=high)',
    'Frequency with which shipments reach consignee within scheduled or expected time, score (1=low to 5=high)',
    'Quality of trade- and transport-related infrastructure, score (1=low to 5=high)']

for i in [0,1,2,3,4,6,7,8,14]:
    test.iloc[:,i]=pd.to_numeric(test.iloc[:,i])
for i in [9,10,11,12,13]:
    test.iloc[:,i]=test.iloc[:,i].astype("category")

Income_mean=pd.pivot_table(df_reg,index=['Income'], values=values,aggfunc=np.mean, fill_value=None)
Region_mean=pd.pivot_table(df_reg,index=['Region'], values=values,aggfunc=np.mean, fill_value=None)
Commodity_group_mean=pd.pivot_table(df_reg,index=['Commodity Groups'], values=values,aggfunc=np.mean, fill_value=None)
Supply_stage_mean=pd.pivot_table(df_reg,index=['food_supply_stage'], values=values,aggfunc=np.mean, fill_value=None)

st.dataframe(Income_mean)
st.dataframe(Region_mean)
st.dataframe(Commodity_group_mean)
st.dataframe(Supply_stage_mean)
# test3=pd.concat([Income_mean,Region_mean,Commodity_group_mean,Supply_stage_mean])
# st.dataframe(test3)

df_selected_country=test[test['country']==selected_country]
Country_mean=pd.pivot_table(df_selected_country,index=['country'], values=values,aggfunc=np.mean, fill_value=None)
st.dataframe(Country_mean)

st.write('Diff. of Means')
Diff_Mean_Income= Income_mean.sub(Country_mean.iloc[0, :])
Diff_Mean_Income['Abs_Sum_Mean_Diff']= abs(Diff_Mean_Income.sum(axis = 1))
Diff_Mean_Income=Diff_Mean_Income.sort_values('Abs_Sum_Mean_Diff', ascending=True)
st.dataframe(Diff_Mean_Income)

Diff_Mean_Region= Region_mean.sub(Country_mean.iloc[0, :])
Diff_Mean_Region['Abs_Sum_Mean_Diff']= abs(Diff_Mean_Region.sum(axis = 1))
Diff_Mean_Region=Diff_Mean_Region.sort_values('Abs_Sum_Mean_Diff', ascending=True)
st.dataframe(Diff_Mean_Region)

Diff_Mean_Commodity_group= Commodity_group_mean.sub(Country_mean.iloc[0, :])
Diff_Mean_Commodity_group['Abs_Sum_Mean_Diff']= abs(Diff_Mean_Commodity_group.sum(axis = 1))
Diff_Mean_Commodity_group=Diff_Mean_Commodity_group.sort_values('Abs_Sum_Mean_Diff', ascending=True)
st.dataframe(Diff_Mean_Commodity_group)

Diff_Mean_Supply_stage= Supply_stage_mean.sub(Country_mean.iloc[0, :])
Diff_Mean_Supply_stage['Abs_Sum_Mean_Diff']= abs(Diff_Mean_Supply_stage.sum(axis = 1))
Diff_Mean_Supply_stage=Diff_Mean_Supply_stage.sort_values('Abs_Sum_Mean_Diff', ascending=True)
st.dataframe(Diff_Mean_Supply_stage)



test4={'Actual Group': [df_selected_country['Income'].value_counts().idxmax(),df_selected_country['Region'].value_counts().idxmax(),df_selected_country['Commodity Groups'].value_counts().idxmax(),df_selected_country['food_supply_stage'].value_counts().idxmax()], 'Nearest Group': [Diff_Mean_Income.index[0],Diff_Mean_Region.index[0],Diff_Mean_Commodity_group.index[0],Diff_Mean_Supply_stage.index[0]]}
df_comparison=pd.DataFrame(test4)
df_comparison.index=['Income Group', 'Region', 'Commodity Group', 'Food Supply Stage']
df_comparison['is_equal']=(df_comparison['Nearest Group']==df_comparison['Actual Group'])
st.dataframe(df_comparison)

if test4.iloc[0,2]==False:
    df_normality_test=pd.DataFrame(stats.shapiro(test[test['country']==selected_country]))

# df_normality_test=pd.DataFrame(stats.shapiro(test[test['country']==selected_country]))
# df_normality_test=df_normality_test.join(pd.DataFrame(stats.shapiro(test[test['Income']==df_comparison.iloc[0,0]])),lsuffix='_Country', rsuffix='_Income_group')
# df_normality_test=df_normality_test.join(pd.DataFrame(stats.shapiro(test[test['Region']==df_comparison.iloc[1,0]])),lsuffix='_Country', rsuffix='_Region_group')
# df_normality_test=df_normality_test.join(pd.DataFrame(stats.shapiro(test[test['Commodity Groups']==df_comparison.iloc[2,0]])),lsuffix='_Country', rsuffix='_Region_group')
# df_normality_test=df_normality_test.join(pd.DataFrame(stats.shapiro(test[test['food_supply_stage']==df_comparison.iloc[3,0]])),lsuffix='_Country', rsuffix='_Region_group')
# df_normality_test=df_normality_test.join(pd.DataFrame(stats.shapiro(test[test['Income']==df_comparison.iloc[0,1]])),lsuffix='_Country', rsuffix='_Income_group')
# df_normality_test=df_normality_test.join(pd.DataFrame(stats.shapiro(test[test['Region']==df_comparison.iloc[1,1]])),lsuffix='_Country', rsuffix='_Region_group')
# df_normality_test=df_normality_test.join(pd.DataFrame(stats.shapiro(test[test['Commodity Groups']==df_comparison.iloc[2,1]])),lsuffix='_Country', rsuffix='_Region_group')
# df_normality_test=df_normality_test.join(pd.DataFrame(stats.shapiro(test[test['food_supply_stage']==df_comparison.iloc[3,1]])),lsuffix='_Country', rsuffix='_Region_group')
# df_normality_test.index=['statistic','pvalue']
# df_normality_test=df_normality_test.transpose()
# df_normality_test.index=['Country','Income_actual','Region_actual','commodity_actual', 'Supply_stage_actual','Income_nearest','Region_nearest','commodity_nearest', 'Supply_stage_nearest' ]
#df_normality_test['Normality condition']=(df_normality_test['pvalue'] >= 0.05)
#st.dataframe(df_normality_test)



st.dataframe(df_policies)
Income_group_actual=df_comparison.iloc[0,0]
test5=df_reg_income_total[df_reg_income_total[Income_group_actual]<0][Income_group_actual]
test5=test5.sort_values(ascending=True)
test5=pd.DataFrame(test5)
test5['Order']=np.arange(len(test5))+1
test5['Order']=pd.to_numeric(test5['Order'])
test5=test5.reset_index()
st.dataframe(test5)

df_policies_test=df_policies
for i in [0,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]:
    df_policies_test.iloc[:,i]=pd.to_numeric(df_policies_test.iloc[:,i])


# for i in range(len(test5)):
#    df_policies_test[df_policies_test[test5['index'][i]]>=1]=df_policies_test[df_policies_test[test5['index'][i]]>=1]+test5.iloc[2,i]



df_policy_recommendation=pd.DataFrame(columns=['Policy Short', 'Policy Component ', 'Policy Area'])
for i in range(len(test5)):
    df_new_policy_recommendation=df_policies_test[df_policies_test[test5['index'][i]]>=1][['Policy Short', 'Policy Component ', 'Policy Area']]
    df_policy_recommendation=df_policy_recommendation.append(df_new_policy_recommendation)

st.dataframe(df_policy_recommendation)


