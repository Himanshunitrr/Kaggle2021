# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

pd.set_option('mode.chained_assignment', None)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
st.set_page_config(layout="wide")
header_container = st.container()
container = st.container()

with header_container:
    country = header_container.selectbox("Select Country", options=['India', 'Algeria', 'Argentina', 'Australia', 'Austria', 'Bangladesh', 'Belarus',
 'Belgium', 'Brazil', 'Canada', 'Chile', 'China', 'Colombia', 'Czech Republic',
 'Denmark', 'Ecuador', 'Egypt', 'Ethiopia', 'France', 'Germany', 'Ghana',
 'Greece', 'Hong Kong (S.A.R.)', 'I do not wish to disclose my location',
 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Japan',
 'Kazakhstan', 'Kenya', 'Malaysia', 'Mexico', 'Morocco', 'Nepal', 'Netherlands',
 'Nigeria', 'Norway', 'Other', 'Pakistan', 'Peru', 'Philippines', 'Poland',
 'Portugal', 'Romania', 'Russia', 'Saudi Arabia', 'Singapore', 'South Africa',
 'South Korea', 'Spain', 'Sri Lanka', 'Sweden', 'Switzerland', 'Taiwan',
 'Thailand', 'Tunisia', 'Turkey', 'UAE', 'UK', 'USA', 'Uganda', 'Ukraine',
 'Vietnam'], index = 0)

with container:

  left_column, right_column = st.columns(2)
  with left_column:

    df = pd.read_csv('./kaggle_survey_2021_responses.csv', skiprows=1)
    df[df.columns[3]].replace({'United Kingdom of Great Britain and Northern Ireland':'UK',
                                'Iran, Islamic Republic of...':'Iran',
                                'United Arab Emirates':'UAE',
                                'United States of America':'USA',
                                'Viet Nam':'Vietnam'}, inplace=True)

    #   st.write(f"List of countries:\n{np.sort(df[df.columns[3]].unique())}\n")

    

    if country not in df[df.columns[3]].unique():
        raise ValueError(f'{country} not found in the list')
    df['country_agg'] = np.where(df[df.columns[3]]==country,country,'Others')

    fig = px.pie(df, df.columns[3], 
            title=f"{len(df[df[df.columns[3]]==country])*100/len(df):.2f}% of all survey respondents are from {country}", 
            hole=0.6)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)
  

    overall_pct = len(df[(df[df.columns[1]].isin(['18-21','22-24','25-29']))])*100/len(df)

    fig = px.pie(df, df.columns[1], title=f"{overall_pct:.0f}% of all Kagglers are less than 30 years old", hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)

    age_pct = len(df[(df[df.columns[3]]==country) & (df[df.columns[1]].isin(['18-21','22-24','25-29']))])*100/len(df[df[df.columns[3]]==country])
    if age_pct < overall_pct:
        title = f"{country} is older, with {age_pct:.0f}% of Kagglers being under under 30"
    elif age_pct > overall_pct:
        title = f"{country} is younger, with {age_pct:.0f}% of Kagglers being under under 30"
    else:
        title = f"{age_pct:.0f}% of Kagglers from {country} are also under 30"
    fig = px.pie(df[df[df.columns[3]]==country], df.columns[1], title=title, hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)

    df['age1'] = df.iloc[:,1].str.split('-').str[0]
    df['age1'].replace('70+','70', inplace=True)
    df['age2'] = df.iloc[:,1].str.split('-').str[1]
    df['age1'] = df.age1.astype('int')
    df.age2.fillna(70, inplace=True)
    df['age2'] = df.age2.astype('int')
    df['age'] = (df.age1+df.age2)/2

    global_median = df.age.mean()
    country_median = df[df.country_agg==country].age.mean()

    if country_median <= global_median:
        title = f"With an average age of {country_median:.0f},<br>Kagglers from {country} are generally {global_median - country_median:.0f} years younger than the average Kaggler"
    else:
        title = f"With an average age of {country_median:.0f},<br>Kagglers from {country} are generally {country_median - global_median:.0f} years younger than the average Kaggler"

    loc = df.groupby(df.columns[3]).age.mean().sort_values(ascending=False).index.to_list().index(country)
    color = ['#636EFA']*len(df.groupby(df.columns[3]).age.mean().sort_values(ascending=False).index)
    color[loc] = 'orange'

    fig = go.Figure(data=[go.Bar(x=df.groupby(df.columns[3]).age.mean().sort_values(ascending=False).index
            , y=df.groupby(df.columns[3]).age.mean().sort_values(ascending=False)
                , marker_color=color)])

    fig.update_layout(
        shapes=[
        dict(
            type= 'line',
            yref= 'y', y0= global_median, y1= global_median,
            xref= 'x', x0= -0.5, x1= len(df.groupby(df.columns[3]).age)-0.5
        )],
        title=title,
        xaxis_title=None,
        yaxis_title='Age')

    fig.add_annotation(x=len(df.groupby(df.columns[3]).age)*0.95, y=global_median, xshift=-20, yshift=10,
                text="Global Average",
                showarrow=False)
    st.plotly_chart(fig)

    df_country = df[df.country_agg==country][df.columns[1]].value_counts(normalize=True).sort_index()
    df_others = df[df.country_agg=='Others'][df.columns[1]].value_counts(normalize=True).sort_index()
    for index in df_others.index:
        if index not in df_country.index:
            df_country[index] = 0
    df_country.sort_index(inplace=True)

    fig = go.Figure(data=[
        go.Bar(name=country, y=df_country.values*100),
        go.Bar(name='Others', y=df_others.values*100)
    ])

    # Change the bar mode
    fig.update_layout(
        barmode='group',
        title=f'Age distribution of Kagglers from {country} compared to others',
        xaxis_title='Age',
        yaxis_title='Percentage of respondents',
        xaxis = dict(
            tickmode = 'array',
            tickvals = [x for x in range(12)],
            ticktext = df[df.columns[1]].sort_values().unique()
        )
    )

    st.plotly_chart(fig)

    fig = px.pie(df, df.columns[2], title=f"Gender distribution of all Kagglers", hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)

    categories = sorted(df.iloc[:,2].unique())
    df_country_gender = df[df.country_agg==country].iloc[:,2].value_counts(normalize=True).sort_index()
    df_other_platform = df[df.country_agg=='Others'].iloc[:,2].value_counts(normalize=True).sort_index()

    fig = go.Figure(data=[
        go.Bar(name=country, x=categories, y=df_country_gender.values*100),
        go.Bar(name='Others', x=categories, y=df_other_platform.values*100)
    ])
    # Change the bar mode
    fig.update_layout(
        title=f'Gender of respondents from {country} compared to rest of the world',
        xaxis_title=None,
        yaxis_title='Percentage',
        xaxis={'categoryorder':'array',
                'categoryarray':categories}
    )
    st.plotly_chart(fig)

    genders = [x for x in df.iloc[:,2].unique() if x !='Prefer not to say']

    for gender in genders:
        df_all = df.groupby(df.columns[3])[df.columns[2]].value_counts().groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))[:,gender].sort_values(ascending=False)
        if country in df_all.index:
            country_avg = df_all[country]
            global_avg = len(df[df[df.columns[2]]==gender])*100/len(df)
            gndr_pct = len(df[df[df.columns[2]]==gender])*100/len(df)
            title=f"{country_avg:.1f}% of Kagglers from {country} identify as \"{gender}\",<br>compared to the global average of {global_avg:.1f}%"

            loc = df_all.index.to_list().index(country)
            color = ['#636EFA']*len(df_all.index)
            color[loc] = 'orange'

            fig = go.Figure(data=[go.Bar(x=df_all.index, y=df_all.values, marker_color=color)])
            fig.update_layout(
                shapes=[
                    dict(
                        type= 'line',
                        yref= 'y', y0= global_avg, y1= global_avg,
                        xref= 'x', x0= -0.5, x1= len(df_all.index)-0.5
                    )],
                title=title,
                xaxis_title=None,
                yaxis_title='Percentage')
            fig.add_annotation(x=len(df_all.index)*0.95, y=global_avg, xshift=-20, yshift=10,
                        text="Global Average",
                        showarrow=False)
            st.plotly_chart(fig)

        else:
            st.write(f"Nobody from {country} identified as {gender}")

    df_academic = df[(df[df.columns[4]]!='I prefer not to answer') & (~df[df.columns[4]].isna())]

    fig = px.pie(df_academic, df_academic.columns[4], 
                title="Academic qualification of Kagglers", hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide', showlegend=False)
    st.plotly_chart(fig)

    most_comm = df_academic[df_academic.iloc[:,3]==country].iloc[:,4].value_counts(normalize=True)
    most_comm_deg = most_comm.index[0]
    most_comm_pct = most_comm[[0]]
    title = f"{most_comm[0]*100:.0f}% respondents from {country} report having {most_comm_deg}"
    fig = px.pie(df_academic[df_academic.iloc[:,3]==country], df_academic.columns[4], title=title, hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide', showlegend=False)
    st.plotly_chart(fig)

    categories = ['No formal education past high school',
                'Some college/university study without earning a bachelor’s degree',
                'Professional degree',
                'Bachelor’s degree',
                'Master’s degree',
                'Doctoral degree']

    df_country_agg = df[df.country_agg==country].iloc[:,4].value_counts(normalize=True)
    df_country_agg.index = pd.Categorical(df_country_agg.index, categories)
    df_country_agg.sort_index(inplace=True)

    df_others_agg = df[df.country_agg=='Others'].iloc[:,4].value_counts(normalize=True)
    df_others_agg.index = pd.Categorical(df_others_agg.index, categories)
    df_others_agg.sort_index(inplace=True)

    fig = go.Figure(data=[
        go.Bar(name=country, x=categories, y=df_country_agg.values*100),
        go.Bar(name='Others', x=categories, y=df_others_agg.values*100)
    ])
    # Change the bar mode
    fig.update_layout(
        title=f'Academic Qualification of respondents from {country} compared to Other countries',
        xaxis_title=None,
        yaxis_title='Percentage'
    )
    st.plotly_chart(fig)

    qualifications = df_academic.iloc[:,4].unique()

    for qualification in qualifications:
        df_all = df.groupby(df.columns[3])[df.columns[4]].value_counts().groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))[:,qualification].sort_values(ascending=False)
        country_avg = df_all[country]
        global_avg = len(df[df[df.columns[4]]==qualification])*100/len(df)
        gndr_pct = len(df[df[df.columns[4]]==qualification])*100/len(df)
        title=f"{country_avg:.1f}% of Kagglers from {country} have {qualification},<br>compared to the global average of {global_avg:.1f}%"

        loc = df_all.index.to_list().index(country)
        color = ['#636EFA']*len(df_all.index)
        color[loc] = 'orange'

        fig = go.Figure(data=[go.Bar(x=df_all.index, y=df_all.values, marker_color=color)])
        fig.update_layout(
            shapes=[
                dict(
                    type= 'line',
                    yref= 'y', y0= global_avg, y1= global_avg,
                    xref= 'x', x0= -0.5, x1= len(df_all.index)-0.5
                )],
            title=title,
            xaxis_title=None,
            yaxis_title='Percentage')
        fig.add_annotation(x=len(df_all.index)*0.95, y=global_avg, xshift=-20, yshift=10,
                    text="Global Average",
                    showarrow=False)
        st.plotly_chart(fig)

    df_job = df[~df[df.columns[5]].isna()]

    fig = px.pie(df_job, df_job.columns[5], title=f'Globally, 26% Kagglers are students, followed by 14% being Data Scientists', hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide', showlegend=False)
    st.plotly_chart(fig)

    df_country_job = df_job[df_job.iloc[:,3]==country].iloc[:,5].value_counts(normalize=True)
    com_job = df_country_job[[0]].index[0]
    if com_job=='Student':
        title = f"For {country} too, most Kagglers are {com_job}s ({df_country_job[0]*100:.0f}%), followed by {df_country_job[[1]].index[0]}s ({df_country_job[1]*100:.0f}%)"
    else:
        title = f"However, for {country}, most Kagglers are {com_job}s ({df_country_job[0]*100:.0f}%), followed by {df_country_job[[1]].index[0]}s ({df_country_job[1]*100:.0f}%)"

        
    fig = px.pie(df_job[df_job.iloc[:,3]==country], df_job.columns[5], title=title, hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide', showlegend=False)
    st.plotly_chart(fig)

    jobs = df_job.iloc[:,5].unique()
    for job in jobs:
        df_all = df.groupby(df.columns[3])[df.columns[5]].value_counts().groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))[:,job].sort_values(ascending=False)
        country_avg = df_all[country]
        global_avg = len(df[df[df.columns[5]]==job])*100/len(df)

        loc = df_all.index.to_list().index(country)
        color = ['#636EFA']*len(df_all.index)
        color[loc] = 'orange'

        fig = go.Figure(data=[go.Bar(x=df_all.index, y=df_all.values, marker_color=color)])

        fig.update_layout(
            shapes=[
            dict(
                type= 'line',
                yref= 'y', y0= global_avg, y1= global_avg,
                xref= 'x', x0= -0.5, x1= len(df_all.index)-0.5
            )],
            title=f"{country_avg:.1f}% of Kagglers from {country} reported their job-title as \'{job}\',<br>compared to the global average of {global_avg:.1f}%",
            xaxis_title=None,
            yaxis_title='Percentage',)

        fig.add_annotation(x=len(df_all.index), y=global_avg, xshift=-60, yshift=10,
                    text="Global Average",
                    showarrow=False)

        st.plotly_chart(fig)

    df_comp = df[(~df.iloc[:,127].isna()) & (~df.iloc[:,5].isin(['Student','Currently not employed']))]

    df_comp['comp1'] = df_comp.iloc[:,127].str.split('-').str[0].apply(lambda x: x.replace(',','').replace('$','').replace('>','')).astype('int')
    df_comp['comp2'] = df_comp.iloc[:,127].str.split('-').str[1].fillna('500000').apply(lambda x: x.replace(',','')).astype('int').astype('int')
    df_comp['comp'] = (df_comp.comp1+df_comp.comp2)/2

    global_median = df_comp.comp.mean()
    country_median = df_comp[df_comp.country_agg==country].comp.mean()

    if country_median <= global_median:
        title = f"With an average annual compensation of {country_median:.0f} USD,<br>Kagglers from {country} generally earn less than the global average ({global_median:.0f} USD)"
    else:
        title = f"With an average annual compensation of {country_median:.0f} USD,<br>Kagglers from {country} generally earn more than the global average ({global_median:.0f} USD)"

    loc = df_comp.groupby(df_comp.columns[3]).comp.mean().sort_values(ascending=False).index.to_list().index(country)
    color = ['#636EFA']*len(df_comp.groupby(df_comp.columns[3]).comp.mean().sort_values(ascending=False).index)
    color[loc] = 'orange'

    fig = go.Figure(data=[go.Bar(x=df_comp.groupby(df_comp.columns[3]).comp.mean().sort_values(ascending=False).index
            , y=df_comp.groupby(df.columns[3]).comp.mean().sort_values(ascending=False)
                , marker_color=color)])

    fig.update_layout(
        shapes=[
        dict(
            type= 'line',
            yref= 'y', y0= global_median, y1= global_median,
            xref= 'x', x0= -0.5, x1= len(df_comp.groupby(df_comp.columns[3]).comp)-0.5
        )],
        title=title,
        xaxis_title=None,
        yaxis_title='Average Annual Compensation (in USD)')

    fig.add_annotation(x=len(df_comp.groupby(df.columns[3]).comp)*0.95, y=global_median, xshift=-20, yshift=10,
                text="Global Average",
                showarrow=False)
    st.plotly_chart(fig)

    most_common = df_comp[df_comp.country_agg==country].groupby(df_comp.columns[127]).size().sort_values(ascending=False)
    most_common_comp = most_common.index[0]
    most_common_comp_pct = most_common[0]*100/most_common.sum()

    fig = px.pie(df_comp[df_comp.country_agg==country], df_comp.columns[127], 
                title=f'{most_common_comp_pct:.0f}% of Kagglers from {country} reported an annual compensation between {most_common_comp}', 
                hole=0.6)
    fig.update_traces(textposition='inside',textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)

    df_coding = df[~df[df.columns[6]].isna()]

    fig = px.pie(df_coding, df_coding.columns[6], title='57% of all survey respondents have been coding for less than 5 years', hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)

    country_coding_exp = len(df_coding[(df_coding[df_coding.columns[3]]==country) & (df_coding[df_coding.columns[6]].isin(['3-5 years','< 1 years','1-3 years']))])*100/len(df_coding[df_coding[df_coding.columns[3]]==country])
    if country_coding_exp > 57:
        title = f"For {country}, this percentage increases to {country_coding_exp:.0f}%"
    elif country_coding_exp < 57:
        title = f"For {country}, this percentage decreases to {country_coding_exp:.0f}%" 
    else:
        title = f"At {country_coding_exp:.0f}%, it is the same for {country} too<br>The average Kaggler from {country} about the same coding experience as the global average"
    fig = px.pie(df_coding[df_coding[df_coding.columns[3]]==country], df_coding.columns[6], title=title, hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)

    df_coding[df_coding.columns[6]] = df_coding[df_coding.columns[6]].replace('< 1 years','0-1 years').replace('I have never written code','0-0').apply(lambda x: x.split()[0])
    df_coding['code1'] = df_coding.iloc[:,6].str.split('-').str[0].replace('20+','20').astype('int')
    df_coding['code2'] = df_coding.iloc[:,6].str.split('-').str[1].fillna('20').astype('int')
    df_coding.groupby([df_coding.columns[6]]+['code1','code2']).size()
    df_coding['code'] = (df_coding.code1+df_coding.code2)/2

    global_median = df_coding.code.mean()
    country_median = df_coding[df_coding.country_agg==country].code.mean()

    if country_median <= global_median:
        title = f"The average Kaggler from {country} has been coding for {country_median:.1f} years,<br>less than the global average of {global_median:.1f} years"
    else:
        title = f"The average Kaggler from {country} has been coding for {country_median:.1f} years,<br>more than the global average of {global_median:.1f} years"

    loc = df_coding.groupby(df_coding.columns[3]).code.mean().sort_values(ascending=False).index.to_list().index(country)
    color = ['#636EFA']*len(df_coding.groupby(df_coding.columns[3]).code.mean().sort_values(ascending=False).index)
    color[loc] = 'orange'

    fig = go.Figure(data=[go.Bar(x=df_coding.groupby(df_comp.columns[3]).code.mean().sort_values(ascending=False).index
            , y=df_coding.groupby(df_coding.columns[3]).code.mean().sort_values(ascending=False)
                , marker_color=color)])

    fig.update_layout(
        shapes=[
        dict(
            type= 'line',
            yref= 'y', y0= global_median, y1= global_median,
            xref= 'x', x0= -0.5, x1= len(df_coding.groupby(df_coding.columns[3]).code)-0.5
        )],
        title=title,
        xaxis_title=None,
        yaxis_title='Average coding experience')

    fig.add_annotation(x=len(df_coding.groupby(df.columns[3]).code)*0.95, y=global_median, xshift=-20, yshift=10,
                text="Global Average",
                showarrow=False)
    st.plotly_chart(fig)

    categories = ['I have never written code','< 1 years','1-3 years','3-5 years','5-10 years','10-20 years','20+ years']

    df_country_agg = df[df.country_agg==country].iloc[:,6].value_counts(normalize=True)
    df_country_agg.index = pd.Categorical(df_country_agg.index, categories)
    df_country_agg.sort_index(inplace=True)

    df_others_agg = df[df.country_agg=='Others'].iloc[:,6].value_counts(normalize=True)
    df_others_agg.index = pd.Categorical(df_others_agg.index, categories)
    df_others_agg.sort_index(inplace=True)

    for index in df_others_agg.index:
            if index not in df_country_agg.index:
                df_country_agg[index] = 0
    df_country_agg.sort_index(inplace=True)

    fig = go.Figure(data=[
        go.Bar(name=country, x=categories, y=df_country_agg.values*100),
        go.Bar(name='Others', x=categories, y=df_others_agg.values*100)
    ])
    # Change the bar mode
    fig.update_layout(
        title=f'Coding Experience of respondents from {country} compared to Other countries',
        xaxis_title=None,
        yaxis_title='Percentage'
    )
    st.plotly_chart(fig)

    programming_cols = [col for col in df.columns 
                        if 'What programming languages do you use on a regular basis?' in col]
    df_programming = df.copy()
    mapper = [col.split('-')[-1].lstrip() for col in programming_cols]
    mapping_dict = dict(zip(programming_cols,mapper))
    df_programming = df_programming[programming_cols + [df.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_programming.dropna(how='all', subset=mapper, inplace=True)
    df_programming.drop(columns=["None"], inplace=True)

    most_comm = df_programming[df_programming.columns[:-2]].count().sort_values(ascending=False)/len(df_programming)
    most_comm_lang = most_comm.index[0]
    most_comm_pct = most_comm[0]*100
    fig = px.bar(df_programming[df_programming.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f"{most_comm_pct:.0f}% of all respondents use {most_comm_lang} on a regular basis",
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_programming[df_programming.iloc[:,-1]==country][df_programming.columns[:-2]].count().sort_values(ascending=False)
    pop_lang = country_most_pop.index[0]
    most_pop_pct = country_most_pop[0]*100/len(df_programming[df_programming.iloc[:,-1]==country])
    if pop_lang==most_comm_lang:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_lang} is even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
        else:
            title = f'{pop_lang} remains the most popular programming language in {country} too,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
    else:
        title = f"However, in {country}, {pop_lang} is more popular,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis"

    fig = px.bar(df_programming[df_programming.iloc[:,-1]==country][df_programming.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    ide_cols = [col for col in df.columns 
                if "Which of the following integrated development environments (IDE's) do you use on a regular basis?" in col]
    df_ide = df.copy()
    mapper = [col.split('-')[-1].lstrip() for col in ide_cols]
    mapping_dict = dict(zip(ide_cols,mapper))
    df_ide = df_ide[ide_cols + [df.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_ide.dropna(how='all', subset=mapper, inplace=True)
    df_ide.rename(columns={'Jupyter (JupyterLab, Jupyter Notebooks, etc) ':'Jupyter'}, inplace=True)

    most_comm = df_ide[df_ide.columns[:-2]].count().sort_values(ascending=False)/len(df_ide)
    most_comm_ide = most_comm.index[0]
    most_comm_pct = most_comm[0]*100
    fig = px.bar(df_ide[df_ide.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use {most_comm_ide}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_ide[df_ide.iloc[:,-1]==country][df_ide.columns[:-2]].count().sort_values(ascending=False)
    pop_ide = country_most_pop.index[0]
    most_pop_pct = country_most_pop[0]*100/len(df_ide[df_ide.iloc[:,-1]==country])
    if pop_ide==most_comm_ide:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_ide} is even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
        else:
            title = f'{pop_ide} remains the most popular IDE in {country} too,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
    else:
        title = f"However, in {country}, {pop_ide} is more common,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis"

    fig = px.bar(df_ide[df_ide.iloc[:,-1]==country][df_ide.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    nb_cols = [col for col in df.columns 
                if "Which of the following hosted notebook products do you use on a regular basis?" in col]
    df_nb = df.copy()
    mapper = [col.split('-')[-1].lstrip() for col in nb_cols]
    mapping_dict = dict(zip(nb_cols,mapper))
    df_nb = df_nb[nb_cols + [df.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_nb.dropna(how='all', subset=mapper, inplace=True)
    df_nb.drop(columns=['None'], inplace=True)

    most_comm = df_nb[df_nb.columns[:-2]].count().sort_values(ascending=False)/len(df_nb)
    most_comm_nb = most_comm.index[0]
    most_comm_pct = most_comm[0]*100
    fig = px.bar(df_nb[df_nb.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use {most_comm_nb}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_nb[df_nb.iloc[:,-1]==country][df_nb.columns[:-2]].count().sort_values(ascending=False)
    pop_nb = country_most_pop.index[0]
    most_pop_pct = country_most_pop[0]*100/len(df_nb[df_nb.iloc[:,-1]==country])
    if pop_nb==most_comm_nb:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_nb} are even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using them on a regular basis'
        else:
            title = f'{pop_nb} remain the most popular hosted notebooks in {country} too,<br>with {most_pop_pct:.0f}% of respondents using them on a regular basis'
    else:
        title = f"However, in {country}, {pop_nb} are more popular,<br>with {most_pop_pct:.0f}% of respondents using them on a regular basis"

    fig = px.bar(df_nb[df_nb.iloc[:,-1]==country][df_nb.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    df_platform = df[(~df.iloc[:,51].isna()) & (df.iloc[:,51]!='None')]

    fig = px.pie(df_platform, df_platform.columns[51], 
                title=f"66% of all survey respondents use a Laptop<br>as their primary computing platform for DS projects", hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide', showlegend=False)
    st.plotly_chart(fig)

    df_country_platform = df_platform[df_platform.iloc[:,3]==country][df.columns[51]].value_counts(normalize=True)*100

    country_avg = df_country_platform.loc['A laptop']

    if country_avg > 66.4:
        title=f"In {country}, this percentage increases to {country_avg:.0f}%"
    else:
        title=f"In {country}, this percentage decreases to {country_avg:.0f}%"
        
    fig = px.pie(df_country_platform, df_country_platform.index, df_country_platform.values, title=title, hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide', showlegend=False)
    st.plotly_chart(fig)

    categories = ['A cloud computing platform', 'A deep learning workstation', 'A laptop', 'A personal computer / desktop', 'Other']
    df_country_platform = df_platform[df_platform.country_agg==country].iloc[:,51].value_counts(normalize=True).sort_index()
    df_other_platform = df_platform[df_platform.country_agg=='Others'].iloc[:,51].value_counts(normalize=True).sort_index()

    for index in df_other_platform.index:
            if index not in df_country_platform.index:
                df_country_platform[index] = 0
    df_country_platform.sort_index(inplace=True)

    fig = go.Figure(data=[
        go.Bar(name=country, x=categories, y=df_country_platform.values*100),
        go.Bar(name='Others', x=categories, y=df_other_platform.values*100)
    ])
    # Change the bar mode
    fig.update_layout(
        title=f'Coding Platform preference of Kagglers from {country} compared to others',
        xaxis_title=None,
        yaxis_title='Percentage',
        xaxis={'categoryorder':'array',
                'categoryarray':categories}
    )
    st.plotly_chart(fig)


    viz_cols = [col for col in df.columns 
                if "What data visualization libraries or tools do you use on a regular basis?" in col]
    df_viz = df.copy()
    mapper = [col.split('-')[-1].strip() for col in viz_cols]
    mapping_dict = dict(zip(viz_cols,mapper))
    df_viz = df_viz[viz_cols + [df.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_viz.dropna(how='all', subset=mapper, inplace=True)
    df_viz.drop(columns=['None'], inplace=True)

    most_comm = df_viz[df_viz.columns[:-2]].count().sort_values(ascending=False)/len(df_viz)
    most_comm_viz = most_comm.index[0]
    most_comm_pct = most_comm[0]*100
    fig = px.bar(df_viz[df_viz.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use {most_comm_viz}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_viz[df_viz.iloc[:,-1]==country][df_viz.columns[:-2]].count().sort_values(ascending=False)
    pop_viz = country_most_pop.index[0]
    most_pop_pct = country_most_pop[0]*100/len(df_viz[df_viz.iloc[:,-1]==country])
    if pop_viz==most_comm_viz:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_viz} is even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
        else:
            title = f'{pop_viz} remains the most popular in {country} too,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
    else:
        title = f"However, in {country}, {pop_viz} is popular,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis"

    fig = px.bar(df_viz[df_viz.iloc[:,-1]==country][df_viz.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    ml_cols = [col for col in df.columns 
                if "Which of the following machine learning frameworks do you use on a regular basis?" in col]
    df_ml = df.copy()
    mapper = [col.split('-')[-1].strip() for col in ml_cols]
    mapper[0]='Scikit-learn'
    mapping_dict = dict(zip(ml_cols,mapper))
    df_ml = df_ml[ml_cols + [df.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_ml.dropna(how='all', subset=mapper, inplace=True)
    df_ml.drop(columns=['None'], inplace=True)

    most_comm = df_ml[df_ml.columns[:-2]].count().sort_values(ascending=False)/len(df_ml)
    most_comm_item = most_comm.index[0]
    most_comm_pct = most_comm[0]*100
    fig = px.bar(df_ml[df_ml.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_ml[df_ml.iloc[:,-1]==country][df_ml.columns[:-2]].count().sort_values(ascending=False)
    pop_ml = country_most_pop.index[0]
    most_pop_pct = country_most_pop[0]*100/len(df_ml[df_ml.iloc[:,-1]==country])
    if pop_ml==most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_ml} is even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
        else:
            title = f'{pop_ml} remains the most popular in {country} too,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
    else:
        title = f"However, in {country}, {pop_ml} is popular,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis"

    fig = px.bar(df_ml[df_ml.iloc[:,-1]==country][df_ml.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    ml_cols = [col for col in df.columns 
                if "Which of the following ML algorithms do you use on a regular basis?" in col]
    df_ml = df.copy()
    mapper = [col.split('-')[-1].strip() for col in ml_cols]
    mapper[9]='Transformer Networks (BERT, gpt-3, etc)'
    mapping_dict = dict(zip(ml_cols,mapper))
    df_ml = df_ml[ml_cols + [df.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_ml.dropna(how='all', subset=mapper, inplace=True)
    df_ml.drop(columns=['None'], inplace=True)

    most_comm = df_ml[df_ml.columns[:-2]].count().sort_values(ascending=False)/len(df_ml)
    most_comm_item = most_comm.index[0]
    most_comm_pct = most_comm[0]*100
    fig = px.bar(df_ml[df_ml.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_ml[df_ml.iloc[:,-1]==country][df_ml.columns[:-2]].count().sort_values(ascending=False)
    pop_ml = country_most_pop.index[0]
    most_pop_pct = country_most_pop[0]*100/len(df_ml[df_ml.iloc[:,-1]==country])
    if pop_ml==most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_ml} is even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
        else:
            title = f'{pop_ml} remains the most popular in {country} too,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
    else:
        title = f"However, in {country}, {pop_ml} is popular,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis"

    fig = px.bar(df_ml[df_ml.iloc[:,-1]==country][df_ml.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    cv_cols = [col for col in df.columns 
                if "Which categories of computer vision methods do you use on a regular basis?" in col]
    df_cv = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in cv_cols]
    mapping_dict = dict(zip(cv_cols,mapper))
    df_cv = df_cv[cv_cols + [df.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_cv.dropna(how='all', subset=mapper, inplace=True)
    df_cv.drop(columns=['None'], inplace=True)

    most_comm = df_cv[df_cv.columns[:-2]].count().sort_values(ascending=False)/len(df_cv)
    most_comm_item = most_comm.index[0]
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_cv[df_cv.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_cv[df_cv.iloc[:,-1]==country][df_cv.columns[:-2]].count().sort_values(ascending=False)
    pop_cv = country_most_pop.index[0]
    if '(' in pop_cv:
        pop_cv_split = pop_cv.split('(')[0].strip()
    most_pop_pct = country_most_pop[0]*100/len(df_cv[df_cv.iloc[:,-1]==country])
    if pop_cv==most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_cv_split} are even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using them on a regular basis'
        else:
            title = f'{pop_cv_split} remain the most popular in {country} too,<br>with {most_pop_pct:.0f}% of respondents using them on a regular basis'
    else:
        title = f"However, in {country}, {pop_cv_split} are popular,<br>with {most_pop_pct:.0f}% of respondents using them on a regular basis"

    fig = px.bar(df_cv[df_cv.iloc[:,-1]==country][df_cv.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    nlp_cols = [col for col in df.columns 
                if "Which of the following natural language processing (NLP) methods do you use on a regular basis?" in col]
    df_nlp = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in nlp_cols]
    mapping_dict = dict(zip(nlp_cols,mapper))
    df_nlp = df_nlp[nlp_cols + [df.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_nlp.dropna(how='all', subset=mapper, inplace=True)
    df_nlp.drop(columns=['None'], inplace=True)

    most_comm = df_nlp[df_nlp.columns[:-2]].count().sort_values(ascending=False)/len(df_nlp)
    most_comm_item = most_comm.index[0]
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_nlp[df_nlp.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_nlp[df_nlp.iloc[:,-1]==country][df_nlp.columns[:-2]].count().sort_values(ascending=False)
    pop_nlp = country_most_pop.index[0]
    if '(' in pop_nlp:
        pop_nlp_split = pop_nlp.split('(')[0].strip()
    most_pop_pct = country_most_pop[0]*100/len(df_nlp[df_nlp.iloc[:,-1]==country])
    if pop_nlp.strip()==most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_nlp_split} are even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using them on a regular basis'
        else:
            title = f'{pop_nlp_split} remain the most popular in {country} too,<br>with {most_pop_pct:.0f}% of respondents using them on a regular basis'
    else:
        title = f"However, in {country}, {pop_nlp_split} are popular,<br>with {most_pop_pct:.0f}% of respondents using them on a regular basis"

    fig = px.bar(df_nlp[df_nlp.iloc[:,-1]==country][df_nlp.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    ccp_cols = [col for col in df.columns 
                if "Which of the following cloud computing platforms do you use on a regular basis?" in col]
    df_ccp = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in ccp_cols]
    mapping_dict = dict(zip(ccp_cols, mapper))
    df_ccp = df_ccp[ccp_cols + [df.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_ccp.dropna(how='all', subset=mapper, inplace=True)
    df_ccp.drop(columns=['None'], inplace=True)

    most_comm = df_ccp[df_ccp.columns[:-2]].count().sort_values(ascending=False)/len(df_ccp)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_ccp[df_ccp.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_ccp[df_ccp.iloc[:,-1]==country][df_ccp.columns[:-2]].count().sort_values(ascending=False)
    pop_ccp = country_most_pop.index[0].strip()
    if '(' in pop_ccp:
        pop_ccp_split = pop_ccp.split('(')[0].strip()
    most_pop_pct = country_most_pop[0]*100/len(df_ccp[df_ccp.iloc[:,-1]==country])
    if pop_ccp == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_ccp_split} is even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
        else:
            title = f'{pop_ccp_split} remains the most popular in {country} too,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
    else:
        title = f"However, in {country}, {pop_ccp_split} is popular,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis"

    fig = px.bar(df_ccp[df_ccp.iloc[:,-1]==country][df_ccp.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    ccp_cols = [col for col in df.columns 
                if "Do you use any of the following cloud computing products on a regular basis?" in col]
    df_ccp = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in ccp_cols]
    mapping_dict = dict(zip(ccp_cols, mapper))
    df_ccp = df_ccp[ccp_cols + [df.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_ccp.dropna(how='all', subset=mapper, inplace=True)
    df_ccp.drop(columns=['No / None'], inplace=True)

    most_comm = df_ccp[df_ccp.columns[:-2]].count().sort_values(ascending=False)/len(df_ccp)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_ccp[df_ccp.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_ccp[df_ccp.iloc[:,-1]==country][df_ccp.columns[:-2]].count().sort_values(ascending=False)
    pop_ccp = country_most_pop.index[0].strip()
    if '(' in pop_nlp:
        pop_ccp_split = pop_ccp.split('(')[0].strip()
    most_pop_pct = country_most_pop[0]*100/len(df_ccp[df_ccp.iloc[:,-1]==country])
    if pop_ccp == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_ccp_split} is even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
        else:
            title = f'{pop_ccp_split} remains the most popular in {country} too,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
    else:
        title = f"However, in {country}, {pop_ccp_split} is popular,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis"

    fig = px.bar(df_ccp[df_ccp.iloc[:,-1]==country][df_ccp.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    mlp_cols = [col for col in df.columns 
                if "Do you use any of the following managed machine learning products on a regular basis?" in col]
    df_mlp = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in mlp_cols]
    mapping_dict = dict(zip(mlp_cols, mapper))
    df_mlp = df_mlp[mlp_cols + [df.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_mlp.dropna(how='all', subset=mapper, inplace=True)
    df_mlp.drop(columns=['No / None'], inplace=True)

    most_comm = df_mlp[df_mlp.columns[:-2]].count().sort_values(ascending=False)/len(df_mlp)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_mlp[df_mlp.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_mlp[df_mlp.iloc[:,-1]==country][df_mlp.columns[:-2]].count().sort_values(ascending=False)
    pop_mlp = country_most_pop.index[0].strip()
    most_pop_pct = country_most_pop[0]*100/len(df_mlp[df_mlp.iloc[:,-1]==country])
    if pop_mlp == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_mlp} is even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
        else:
            title = f'{pop_mlp} remains the most popular in {country} too,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
    else:
        title = f"However, in {country}, {pop_mlp} is popular,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis"

    fig = px.bar(df_mlp[df_mlp.iloc[:,-1]==country][df_mlp.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    bdp_cols = [col for col in df.columns 
                if "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis?" in col]
    df_bdp = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in bdp_cols]
    mapping_dict = dict(zip(bdp_cols, mapper))
    df_bdp = df_bdp[bdp_cols + [df.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_bdp.dropna(how='all', subset=mapper, inplace=True)
    df_bdp.drop(columns=['None'], inplace=True)

    most_comm = df_bdp[df_bdp.columns[:-2]].count().sort_values(ascending=False)/len(df_bdp)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_bdp[df_bdp.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_bdp[df_bdp.iloc[:,-1]==country][df_bdp.columns[:-2]].count().sort_values(ascending=False)
    pop_bdp = country_most_pop.index[0].strip()
    pop_bdp_split = pop_bdp.split('(')[0].strip() if '(' in pop_bdp else pop_bdp
    most_pop_pct = country_most_pop[0]*100/len(df_bdp[df_bdp.iloc[:,-1]==country])
    if pop_bdp == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_bdp_split} is even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
        else:
            title = f'{pop_bdp_split} remains the most popular in {country} too,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
    else:
        title = f"However, in {country}, {pop_bdp_split} is popular,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis"

    fig = px.bar(df_bdp[df_bdp.iloc[:,-1]==country][df_bdp.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

  with right_column:

    df_bdp1 = df[(~df.iloc[:,186].isna()) & (df.iloc[:,186]!='None')]

    fig = px.pie(df_bdp1, df_bdp1.columns[186], 
                title=f"Globally, 22% of Kagglers use MySQL most often", hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)

    df_country_bd = df_bdp1[df_bdp1.iloc[:,3]==country][df_bdp1.columns[186]].value_counts(normalize=True)*100

    country_avg = df_country_bd.loc['MySQL ']

    if country_avg > 22:
        title=f"In {country}, this percentage increases to {country_avg:.0f}%"
    else:
        title=f"In {country}, this percentage decreases to {country_avg:.0f}%"
        
    fig = px.pie(df_country_bd, df_country_bd.index, df_country_bd.values, title=title, hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)

    categories = sorted(list(df_bdp1.iloc[:,186].unique()))
    df_country_bd = df_bdp1[df_bdp1.country_agg==country].iloc[:,186].value_counts(normalize=True).sort_index()
    df_other_bd = df_platform[df_platform.country_agg=='Others'].iloc[:,186].value_counts(normalize=True).sort_index()

    for index in df_other_bd.index:
            if index not in df_country_bd.index:
                df_country_bd[index] = 0
    df_country_bd.sort_index(inplace=True)

    fig = go.Figure(data=[
        go.Bar(name=country, x=categories, y=df_country_bd.values*100),
        go.Bar(name='Others', x=categories, y=df_other_bd.values*100)
    ])
    # Change the bar mode
    fig.update_layout(
        title=f'Big Data Product preference of Kagglers from {country} compared to others',
        xaxis_title=None,
        yaxis_title='Percentage',
        xaxis={'categoryorder':'array',
                'categoryarray':categories}
    )
    st.plotly_chart(fig)

    bit_cols = [col for col in df.columns 
                if "Which of the following business intelligence tools do you use on a regular basis?" in col]
    df_bit = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in bit_cols]
    mapping_dict = dict(zip(bit_cols, mapper))
    df_bit = df_bit[bit_cols + [df_bit.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_bit.dropna(how='all', subset=mapper, inplace=True)
    df_bit.drop(columns=['None'], inplace=True)

    most_comm = df_bit[df_bit.columns[:-2]].count().sort_values(ascending=False)/len(df_bit)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_bit[df_bit.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_bit[df_bit.iloc[:,-1]==country][df_bit.columns[:-2]].count().sort_values(ascending=False)
    pop_bit = country_most_pop.index[0].strip()
    pop_bit_split = pop_bit.split('(')[0].strip() if '(' in pop_bit else pop_bit
    most_pop_pct = country_most_pop[0]*100/len(df_bit[df_bit.iloc[:,-1]==country])
    if pop_bit == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_bit_split} is even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
        else:
            title = f'{pop_bit_split} remains the most popular in {country} too,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
    else:
        title = f"However, in {country}, {pop_bit_split} is popular,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis"

    fig = px.bar(df_bit[df_bit.iloc[:,-1]==country][df_bit.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    df_bit1 = df[(~df.iloc[:,204].isna()) & (df.iloc[:,204]!='None')]

    fig = px.pie(df_bit1, df_bit1.columns[204], 
                title=f"Globally, 36% of Kagglers use Tableau most often", hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)

    df_country_bi = df_bit1[df_bit1.iloc[:,3]==country][df_bit1.columns[204]].value_counts(normalize=True)*100

    country_avg = df_country_bi.loc['Tableau']

    if country_avg > 36:
        title=f"In {country}, this percentage increases to {country_avg:.0f}%"
    else:
        title=f"In {country}, this percentage decreases to {country_avg:.0f}%"
        
    fig = px.pie(df_country_bi, df_country_bi.index, df_country_bi.values, title=title, hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)

    categories = sorted(list(df_bit1.iloc[:,204].unique()))
    df_country_bi = df_bit1[df_bit1.country_agg==country].iloc[:,204].value_counts(normalize=True).sort_index()
    df_other_bi = df_bit1[df_bit1.country_agg=='Others'].iloc[:,204].value_counts(normalize=True).sort_index()

    for index in df_other_bi.index:
        if index not in df_country_bi.index:
            df_country_bi[index] = 0
    df_country_bi.sort_index(inplace=True)

    fig = go.Figure(data=[
        go.Bar(name=country, x=categories, y=df_country_bi.values*100),
        go.Bar(name='Others', x=categories, y=df_other_bi.values*100)
    ])
    # Change the bar mode
    fig.update_layout(
        title=f'Business Intelligence Product preference of Kagglers from {country} compared to others',
        xaxis_title=None,
        yaxis_title='Percentage',
        xaxis={'categoryorder':'array',
                'categoryarray':categories}
    )
    st.plotly_chart(fig)

    cols = [col for col in df.columns 
                if "Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?" in col]
    df_filtered = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in cols]
    mapping_dict = dict(zip(cols, mapper))
    df_filtered = df_filtered[cols + [df_filtered.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_filtered.dropna(how='all', subset=mapper, inplace=True)
    df_filtered.drop(columns=['No / None'], inplace=True)

    full_automation_pct = len(df_filtered[(~df_filtered['Automation of full ML pipelines (e.g. Google AutoML, H2O Driverless AI)'].isna())])*100/len(df_filtered)

    fig = px.bar(df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{full_automation_pct:.0f}% of all respondents use Automation of full ML pipelines',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False)
    most_pop_pct = country_most_pop.loc['Automation of full ML pipelines (e.g. Google AutoML, H2O Driverless AI)']*100/len(df_filtered[df_filtered.iloc[:,-1]==country])
    if most_pop_pct > full_automation_pct:
        title = f'Automation of full ML pipelines is even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
    else:
        title = f'Automation of full ML pipelines remains the most popular in {country} too,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'

    fig = px.bar(df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    cols = [col for col in df.columns 
                if "Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?" in col]
    df_filtered = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in cols]
    mapping_dict = dict(zip(cols, mapper))
    df_filtered = df_filtered[cols + [df_filtered.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_filtered.dropna(how='all', subset=mapper, inplace=True)
    df_filtered.drop(columns=['No / None'], inplace=True)


    most_comm = df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False)/len(df_filtered)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False)
    pop = country_most_pop.index[0].strip()
    pop_split = pop.split('(')[0].strip() if '(' in pop else pop
    most_pop_pct = country_most_pop[0]*100/len(df_filtered[df_filtered.iloc[:,-1]==country])
    if pop == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_split} is even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
        else:
            title = f'{pop_split} remains the most popular in {country} too,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
    else:
        title = f"However, in {country}, {pop_split} is popular,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis"

    fig = px.bar(df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    cols = [col for col in df.columns 
                if "Do you use any tools to help manage machine learning experiments?" in col]
    df_filtered = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in cols]
    mapping_dict = dict(zip(cols, mapper))
    df_filtered = df_filtered[cols + [df_filtered.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_filtered.dropna(how='all', subset=mapper, inplace=True)
    df_filtered.drop(columns=['No / None'], inplace=True)

    most_comm = df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False)/len(df_filtered)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False)
    pop = country_most_pop.index[0].strip()
    pop_split = pop.split('(')[0].strip() if '(' in pop else pop
    most_pop_pct = country_most_pop[0]*100/len(df_filtered[df_filtered.iloc[:,-1]==country])
    if pop ==most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_split} is even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
        else:
            title = f'{pop_split} remains the most popular in {country} too,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
    else:
        title = f"However, in {country}, {pop_split} is popular,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis"

    fig = px.bar(df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    cols = [col for col in df.columns 
                if "Where do you publicly share your data analysis or machine learning applications?" in col]
    df_filtered = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in cols]
    mapping_dict = dict(zip(cols, mapper))
    df_filtered = df_filtered[cols + [df_filtered.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_filtered.dropna(how='all', subset=mapper, inplace=True)
    df_filtered.drop(columns=['I do not share my work publicly'], inplace=True)

    most_comm = df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False)/len(df_filtered)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use share their work on {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False)
    pop = country_most_pop.index[0].strip()
    pop_split = pop.split('(')[0].strip() if '(' in pop else pop
    most_pop_pct = country_most_pop[0]*100/len(df_filtered[df_filtered.iloc[:,-1]==country])
    if pop == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_split} is even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
        else:
            title = f'{pop_split} remains the most popular in {country} too,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
    else:
        title = f"However, in {country}, {pop_split} is popular,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis"

    fig = px.bar(df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    cols = [col for col in df.columns 
                if "On which platforms have you begun or completed data science courses?" in col]
    df_filtered = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in cols]
    mapping_dict = dict(zip(cols, mapper))
    df_filtered = df_filtered[cols + [df_filtered.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_filtered.dropna(how='all', subset=mapper, inplace=True)
    df_filtered.drop(columns=['None'], inplace=True)

    most_comm = df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False)/len(df_filtered)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use {most_comm_item} for learning',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False)
    pop = country_most_pop.index[0].strip()
    pop_split = pop.split('(')[0].strip() if '(' in pop else pop
    most_pop_pct = country_most_pop[0]*100/len(df_filtered[df_filtered.iloc[:,-1]==country])
    if pop == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_split} is even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
        else:
            title = f'{pop_split} remains the most popular in {country} too,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
    else:
        title = f"However, in {country}, {pop_split} is popular,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis"

    fig = px.bar(df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    cols = [col for col in df.columns 
                if "Who/what are your favorite media sources that report on data science topics?" in col]
    df_filtered = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in cols]
    mapping_dict = dict(zip(cols, mapper))
    df_filtered = df_filtered[cols + [df_filtered.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_filtered.dropna(how='all', subset=mapper, inplace=True)
    df_filtered.drop(columns=['None'], inplace=True)

    most_comm = df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False)/len(df_filtered)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents use {most_comm_item} to stay updated on Data Science',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False)
    pop = country_most_pop.index[0].strip()
    pop_split = pop.split('(')[0].strip() if '(' in pop else pop
    most_pop_pct = country_most_pop[0]*100/len(df_filtered[df_filtered.iloc[:,-1]==country])
    if pop == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'{pop_split} is even more popular in {country},<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
        else:
            title = f'{pop_split} remains the most popular in {country} too,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis'
    else:
        title = f"However, in {country}, {pop_split} is popular,<br>with {most_pop_pct:.0f}% of respondents using it on a regular basis"

    fig = px.bar(df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    cols = [col for col in df.columns 
                if "Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years?" in col]
    df_filtered = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in cols]
    mapping_dict = dict(zip(cols, mapper))
    df_filtered = df_filtered[cols + [df_filtered.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_filtered.dropna(how='all', subset=mapper, inplace=True)
    df_filtered.drop(columns=['None'], inplace=True)

    most_comm = df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False)/len(df_filtered)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents hope to become familiar with {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False)
    pop = country_most_pop.index[0].strip()
    pop_split = pop.split('(')[0].strip() if '(' in pop else pop
    most_pop_pct = country_most_pop[0]*100/len(df_filtered[df_filtered.iloc[:,-1]==country])
    if pop == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'In {country}, at {most_pop_pct:.0f}%,<br>even more respondents hope to become familiar with {pop_split}'
        else:
            title = f'In {country}, {pop_split} is still highest in demand,<br> although slightly less popular at {most_pop_pct:.0f}%'
    else:
        title = f"However, in {country}, more respondents hope to become familiar with {pop_split} ({most_pop_pct:.0f}%)"

    fig = px.bar(df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    cols = [col for col in df.columns 
                if "In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products?" in col]
    df_filtered = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in cols]
    mapping_dict = dict(zip(cols, mapper))
    df_filtered = df_filtered[cols + [df_filtered.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_filtered.dropna(how='all', subset=mapper, inplace=True)
    df_filtered.drop(columns=['None'], inplace=True)

    most_comm = df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False)/len(df_filtered)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'Inspite of AWS\'s popularity, more respondents globally want to become more familiar with<br>{most_comm_item} ({most_comm_pct:.1f}%)',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False)
    pop = country_most_pop.index[0].strip()
    pop_split = pop.split('(')[0].strip() if '(' in pop else pop
    most_pop_pct = country_most_pop[0]*100/len(df_filtered[df_filtered.iloc[:,-1]==country])
    if pop == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'In {country}, at {most_pop_pct:.1f}%,<br>even more respondents hope to become familiar with {pop_split}'
        else:
            title = f'In {country}, {pop_split} is still highest in demand,<br> although slightly less popular at {most_pop_pct:.0f}%'
    else:
        title = f"However, in {country}, more respondents hope to become familiar with {pop_split} ({most_pop_pct:.0f}%)"

    fig = px.bar(df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    cols = [col for col in df.columns 
                if "In the next 2 years, do you hope to become more familiar with any of these managed machine learning products?" in col]
    df_filtered = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in cols]
    mapping_dict = dict(zip(cols, mapper))
    df_filtered = df_filtered[cols + [df_filtered.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_filtered.dropna(how='all', subset=mapper, inplace=True)
    df_filtered.drop(columns=['None'], inplace=True)

    most_comm = df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False)/len(df_filtered)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'Google maintains the lead here too, with {most_comm_pct:.0f}% respondents globally<br>wanting to become more familiar with {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False)
    pop = country_most_pop.index[0].strip()
    pop_split = pop.split('(')[0].strip() if '(' in pop else pop
    most_pop_pct = country_most_pop[0]*100/len(df_filtered[df_filtered.iloc[:,-1]==country])
    if pop == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'In {country}, at {most_pop_pct:.0f}%,<br>even more respondents hope to become familiar with {pop_split}'
        else:
            title = f'In {country}, {pop_split} is still highest in demand,<br> although slightly less popular at {most_pop_pct:.0f}%'
    else:
        title = f"However, in {country}, more respondents hope to become familiar with {pop_split} ({most_pop_pct:.0f}%)"

    fig = px.bar(df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    cols = [col for col in df.columns 
                if "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years?" in col]
    df_filtered = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in cols]
    mapping_dict = dict(zip(cols, mapper))
    df_filtered = df_filtered[cols + [df_filtered.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_filtered.dropna(how='all', subset=mapper, inplace=True)
    df_filtered.drop(columns=['None'], inplace=True)

    most_comm = df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False)/len(df_filtered)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'At {most_comm_pct:.0f}%, {most_comm_item} is the most desired Big Data Product',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False)
    pop = country_most_pop.index[0].strip()
    pop_split = pop.split('(')[0].strip() if '(' in pop else pop
    most_pop_pct = country_most_pop[0]*100/len(df_filtered[df_filtered.iloc[:,-1]==country])
    if pop == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'In {country}, at {most_pop_pct:.0f}%,<br>even more respondents hope to become familiar with {pop_split}'
        else:
            title = f'In {country}, {pop_split} is still highest in demand,<br> although slightly less popular at {most_pop_pct:.0f}%'
    else:
        title = f"However, in {country}, more respondents hope to become familiar with {pop_split} ({most_pop_pct:.0f}%)"

    fig = px.bar(df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)


    cols = [col for col in df.columns 
                if "Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years?" in col]
    df_filtered = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in cols]
    mapping_dict = dict(zip(cols, mapper))
    df_filtered = df_filtered[cols + [df_filtered.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_filtered.dropna(how='all', subset=mapper, inplace=True)
    df_filtered.drop(columns=['None'], inplace=True)

    most_comm = df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False)/len(df_filtered)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents hope to become more familiar with {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False)
    pop = country_most_pop.index[0].strip()
    pop_split = pop.split('(')[0].strip() if '(' in pop else pop
    most_pop_pct = country_most_pop[0]*100/len(df_filtered[df_filtered.iloc[:,-1]==country])
    if pop == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'In {country}, at {most_pop_pct:.0f}%,<br>even more respondents hope to become familiar with {pop_split}'
        else:
            title = f'In {country}, {pop_split} is still highest in demand,<br> although slightly less popular at {most_pop_pct:.0f}%'
    else:
        title = f"However, in {country}, more respondents hope to become familiar with {pop_split} ({most_pop_pct:.0f}%)"

    fig = px.bar(df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    cols = [col for col in df.columns 
                if "Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?" in col]
    df_filtered = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in cols]
    mapping_dict = dict(zip(cols, mapper))
    df_filtered = df_filtered[cols + [df_filtered.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_filtered.dropna(how='all', subset=mapper, inplace=True)
    df_filtered.drop(columns=['None'], inplace=True)

    most_comm = df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False)/len(df_filtered)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents hope to become more familiar with {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False)
    pop = country_most_pop.index[0].strip()
    pop_split = pop.split('(')[0].strip() if '(' in pop else pop
    most_pop_pct = country_most_pop[0]*100/len(df_filtered[df_filtered.iloc[:,-1]==country])
    if pop == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'In {country}, at {most_pop_pct:.0f}%,<br>even more respondents hope to become familiar with {pop_split}'
        else:
            title = f'In {country}, {pop_split} is still highest in demand,<br>although slightly less popular at {most_pop_pct:.0f}%'
    else:
        title = f"However, in {country}, more respondents hope to become familiar with {pop_split} ({most_pop_pct:.0f}%)"

    fig = px.bar(df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    cols = [col for col in df.columns 
                if "Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?" in col]
    df_filtered = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in cols]
    mapping_dict = dict(zip(cols, mapper))
    df_filtered = df_filtered[cols + [df_filtered.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_filtered.dropna(how='all', subset=mapper, inplace=True)
    df_filtered.drop(columns=['None'], inplace=True)

    most_comm = df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False)/len(df_filtered)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents hope to become more familiar with {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False)
    pop = country_most_pop.index[0].strip()
    pop_split = pop.split('(')[0].strip() if '(' in pop else pop
    most_pop_pct = country_most_pop[0]*100/len(df_filtered[df_filtered.iloc[:,-1]==country])
    if pop == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'In {country}, at {most_pop_pct:.0f}%,<br>even more respondents hope to become familiar with {pop_split}'
        else:
            title = f'In {country}, {pop_split} is still highest in demand,<br> although slightly less popular at {most_pop_pct:.0f}%'
    else:
        title = f"However, in {country}, more respondents hope to become familiar with {pop_split} ({most_pop_pct:.0f}%)"

    fig = px.bar(df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    cols = [col for col in df.columns 
                if "In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments?" in col]
    df_filtered = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in cols]
    mapping_dict = dict(zip(cols, mapper))
    df_filtered = df_filtered[cols + [df_filtered.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_filtered.dropna(how='all', subset=mapper, inplace=True)
    df_filtered.drop(columns=['None'], inplace=True)

    most_comm = df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False)/len(df_filtered)
    most_comm_item = most_comm.index[0].strip()
    most_comm_pct = most_comm[0]*100

    fig = px.bar(df_filtered[df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{most_comm_pct:.0f}% of all respondents hope to become more familiar with {most_comm_item}',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False)
    pop = country_most_pop.index[0].strip()
    pop_split = pop.split('(')[0].strip() if '(' in pop else pop
    most_pop_pct = country_most_pop[0]*100/len(df_filtered[df_filtered.iloc[:,-1]==country])
    if pop == most_comm_item:
        if most_pop_pct > most_comm_pct:
            title = f'In {country}, at {most_pop_pct:.0f}%,<br>even more respondents hope to become familiar with {pop_split}'
        else:
            title = f'In {country}, {pop_split} is still highest in demand,<br> although slightly less popular at {most_pop_pct:.0f}%'
    else:
        title = f"However, in {country}, more respondents hope to become familiar with {pop_split} ({most_pop_pct:.0f}%)"

    fig = px.bar(df_filtered[df_filtered.iloc[:,-1]==country][df_filtered.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    df_wp_size = df[~df.iloc[:,116].isna()]

    fig = px.pie(df_wp_size, df_wp_size.columns[116], title='47% of all Kagglers work in companies with less than 250 employees', hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)

    size_pct = len(df_wp_size[(df_wp_size[df_wp_size.columns[3]]==country) & (df_wp_size[df_wp_size.columns[116]].isin(['0-49 employees','50-249 employees']))])*100 \
    /len(df_wp_size[df_wp_size[df_wp_size.columns[3]]==country])
    if size_pct > 47:
        title = f"For {country}, this percentage increases to {size_pct:.0f}%<br>Kagglers from {country} tend to work in smaller companies"
    elif size_pct < 47:
        title = f"For {country}, this percentage decreases to {size_pct:.0f}%<br>Kagglers from {country} tend to work in larger companies"
    else:
        title = f"At {size_pct:.0f}%, it is the same for {country} too"
    fig = px.pie(df_wp_size[df_wp_size[df_wp_size.columns[3]]==country], df_wp_size.columns[116], title=title, hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)

    df_country = df_wp_size[df_wp_size.country_agg==country][df_wp_size.columns[116]].value_counts(normalize=True).sort_index()
    df_others = df_wp_size[df_wp_size.country_agg=='Others'][df_wp_size.columns[116]].value_counts(normalize=True).sort_index()

    for index in df_others.index:
        if index not in df_country.index:
            df_country[index] = 0
    df_country.sort_index(inplace=True)

    fig = go.Figure(data=[
        go.Bar(name=country, y=df_country.values*100),
        go.Bar(name='Others', y=df_others.values*100)
    ])

    # Change the bar mode
    fig.update_layout(
        barmode='group',
        title=f'Workplace size of Kagglers from {country} compared to others',
        xaxis_title=None,
        yaxis_title='Percentage of respondents',
        xaxis = dict(
            tickmode = 'array',
            tickvals = [x for x in range(5)],
            ticktext = df_wp_size[df_wp_size.columns[116]].sort_values().unique()
        )
    )

    st.plotly_chart(fig)

    df_dsts_size = df[~df.iloc[:,117].isna()]

    fig = px.pie(df_dsts_size, df_dsts_size.columns[117], 
                title='57.5% of all Kagglers work in companies with less than 5 individuals handling Data Science workloads', hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)


    size_pct = len(df_dsts_size[(df_dsts_size[df_dsts_size.columns[3]]==country) & (df_dsts_size[df_wp_size.columns[117]].isin(['0','1-2', '3-4']))])*100 \
    /len(df_dsts_size[df_dsts_size[df_dsts_size.columns[3]]==country])
    if size_pct > 57.5:
        title = f"For {country}, this percentage increases to {size_pct:.0f}%<br>Kagglers from {country} work in companies with smaller Data Science teams"
    elif size_pct < 57.5:
        title = f"For {country}, this percentage decreases to {size_pct:.0f}%<br>Kagglers from {country} work in companies with larger Data Science teams"
    else:
        title = f"At {size_pct:.0f}%, it is the same for {country} too"
    fig = px.pie(df_dsts_size[df_dsts_size[df_dsts_size.columns[3]]==country], df_dsts_size.columns[117], title=title, hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)

    df_country = df_dsts_size[df_dsts_size.country_agg==country][df_dsts_size.columns[117]].value_counts(normalize=True).sort_index()
    df_others = df_dsts_size[df_dsts_size.country_agg=='Others'][df_dsts_size.columns[117]].value_counts(normalize=True).sort_index()

    for index in df_others.index:
            if index not in df_country.index:
                df_country[index] = 0
    df_country.sort_index(inplace=True)

    fig = go.Figure(data=[
        go.Bar(name=country, y=df_country.values*100),
        go.Bar(name='Others', y=df_others.values*100)
    ])

    # Change the bar mode
    fig.update_layout(
        barmode='group',
        title=f'Company Data Science Team size of Kagglers from {country} compared to others',
        xaxis_title=None,
        yaxis_title='Percentage of respondents',
        xaxis = dict(
            tickmode = 'array',
            tickvals = [x for x in range(7)],
            ticktext = df_dsts_size[df_dsts_size.columns[117]].sort_values().unique()
        )
    )

    st.plotly_chart(fig)

    df_mla = df[~df.iloc[:,118].isna()]

    fig = px.pie(df_mla, df_mla.columns[118], 
                title='20.5% of all Kagglers work in companies which don\'t use ML methods', hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide', showlegend=False)
    st.plotly_chart(fig)

    size_pct = len(df_mla[(df_mla[df_mla.columns[3]]==country) & 
                            (df_mla[df_mla.columns[118]].isin(['No (we do not use ML methods)']))])*100 \
    /len(df_mla[df_mla[df_mla.columns[3]]==country])
    if size_pct > 20.5:
        title = f"For {country}, this percentage increases to {size_pct:.0f}%<br>Less Kagglers from {country} work in a company using ML methods"
    elif size_pct < 20.5:
        title = f"For {country}, this percentage decreases to {size_pct:.0f}%<br>More Kagglers from {country} work in a company using ML methods"
    else:
        title = f"At {size_pct:.0f}%, it is the same for {country} too"
    fig = px.pie(df_mla[df_mla[df_mla.columns[3]]==country], df_mla.columns[118], title=title, hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide', showlegend=False)
    st.plotly_chart(fig)

    df_country = df_mla[df_mla.country_agg==country][df_mla.columns[118]].value_counts(normalize=True).sort_index()
    df_others = df_mla[df_mla.country_agg=='Others'][df_mla.columns[118]].value_counts(normalize=True).sort_index()

    for index in df_others.index:
            if index not in df_country.index:
                df_country[index] = 0
    df_country.sort_index(inplace=True)

    fig = go.Figure(data=[
        go.Bar(name=country, y=df_country.values*100),
        go.Bar(name='Others', y=df_others.values*100)
    ])

    # Change the bar mode
    fig.update_layout(
        barmode='group',
        title=f'ML adoption in workplaces of Kagglers from {country} compared to others',
        xaxis_title=None,
        yaxis_title='Percentage of respondents',
        xaxis = dict(
            tickmode = 'array',
            tickvals = [x for x in range(7)],
            ticktext = df_mla[df_mla.columns[118]].sort_values().unique()
        )
    )

    st.plotly_chart(fig)

    df_mla = df[~df.iloc[:,118].isna()]

    fig = px.pie(df_mla, df_mla.columns[118], 
                title='20.5% of all Kagglers work in companies which don\'t use ML methods', hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide', showlegend=False)
    st.plotly_chart(fig)

    size_pct = len(df_mla[(df_mla[df_mla.columns[3]]==country) & 
                            (df_mla[df_mla.columns[118]].isin(['No (we do not use ML methods)']))])*100 \
    /len(df_mla[df_mla[df_mla.columns[3]]==country])
    if size_pct > 20.5:
        title = f"For {country}, this percentage increases to {size_pct:.0f}%<br>Less Kagglers from {country} work in a company using ML methods"
    elif size_pct < 20.5:
        title = f"For {country}, this percentage decreases to {size_pct:.0f}%<br>More Kagglers from {country} work in a company using ML methods"
    else:
        title = f"At {size_pct:.0f}%, it is the same for {country} too"
    fig = px.pie(df_mla[df_mla[df_mla.columns[3]]==country], df_mla.columns[118], title=title, hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide', showlegend=False)
    st.plotly_chart(fig)

    df_country = df_mla[df_mla.country_agg==country][df_mla.columns[118]].value_counts(normalize=True).sort_index()
    df_others = df_mla[df_mla.country_agg=='Others'][df_mla.columns[118]].value_counts(normalize=True).sort_index()

    for index in df_others.index:
            if index not in df_country.index:
                df_country[index] = 0
    df_country.sort_index(inplace=True)

    fig = go.Figure(data=[
        go.Bar(name=country, y=df_country.values*100),
        go.Bar(name='Others', y=df_others.values*100)
    ])

    # Change the bar mode
    fig.update_layout(
        barmode='group',
        title=f'ML adoption in workplaces of Kagglers from {country} compared to others',
        xaxis_title=None,
        yaxis_title='Percentage of respondents',
        xaxis = dict(
            tickmode = 'array',
            tickvals = [x for x in range(7)],
            ticktext = df_mla[df_mla.columns[118]].sort_values().unique()
        )
    )

    st.plotly_chart(fig)

    role_cols = [col for col in df.columns 
                if "Select any activities that make up an important part of your role at work" in col]
    df_role = df.copy()
    mapper = [col.split('- ',maxsplit=2)[2] for col in role_cols]
    mapping_dict = dict(zip(role_cols,mapper))
    df_role = df_role[role_cols + [df.columns[3]] + ['country_agg']].rename(columns=mapping_dict)
    df_role.dropna(how='all', subset=mapper, inplace=True)

    pct = len(df_role[~df_role['Do research that advances the state of the art of machine learning'].isna()])*100/len(df_role)

    fig = px.bar(df_role[df_role.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=f'{pct:.0f}% of all respondents are involved in ML research',
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    country_most_pop = df_role[df_role.iloc[:,-1]==country][df_role.columns[:-2]].count().sort_values(ascending=False)
    research_cnt = country_most_pop.loc['Do research that advances the state of the art of machine learning']
    research_pct = research_cnt*100/len(df_role[df_role.iloc[:,-1]==country])
    if research_pct > pct:
        title = f'At {research_pct:.0f}%, more Kagglers from {country} are involved in ML research than the global average'
    elif research_pct < pct:
        title = f'At {research_pct:.0f}%, less Kagglers from {country} are involved in ML research than the global average'
    else:
        title = f'At {research_pct:.0f}%, the percentage of Kagglers {country} involved in ML research is the same as the global average'

    fig = px.bar(df_role[df_role.iloc[:,-1]==country][df_role.columns[:-2]].count().sort_values(ascending=False))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title='Number of respondents',
        showlegend=False
    )
    st.plotly_chart(fig)

    df_ml_spend = df[~df.iloc[:,128].isna()]
    df_ml_spend[df_ml_spend.columns[128]] = df_ml_spend[df_ml_spend.columns[128]].apply(lambda x: x.replace('$','').replace(',',''))

    fig = px.pie(df_ml_spend, df_ml_spend.columns[128], 
                title='39% of all Kagglers work in companies which don\'t spend on ML/Cloud', hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)

    size_pct = len(df_ml_spend[(df_ml_spend.iloc[:,3]==country) & (df_ml_spend.iloc[:,128]=='0 (USD)')])*100/sum(df_ml_spend.iloc[:,3]==country)

    if size_pct > 38.9:
        title = f"For {country}, this percentage increases to {size_pct:.0f}%<br>More Kagglers from {country} work in a company not spending on ML/Cloud"
    elif size_pct < 38.9:
        title = f"For {country}, this percentage decreases to {size_pct:.0f}%<br>More Kagglers from {country} work in a company spending on ML/Cloud"
    else:
        title = f"At {size_pct:.0f}%, it is the same for {country} too"
        
    fig = px.pie(df_ml_spend[df_ml_spend[df_ml_spend.columns[3]]==country], df_ml_spend.columns[128], title=title, hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig)

    df_country = df_ml_spend[df_ml_spend.country_agg==country][df_ml_spend.columns[128]].value_counts(normalize=True).sort_index()
    df_others = df_ml_spend[df_ml_spend.country_agg=='Others'][df_ml_spend.columns[128]].value_counts(normalize=True).sort_index()
    for index in df_others.index:
            if index not in df_country.index:
                df_country[index] = 0
    df_country.sort_index(inplace=True)

    fig = go.Figure(data=[
        go.Bar(name=country, y=df_country.values*100),
        go.Bar(name='Others', y=df_others.values*100)
    ])

    # Change the bar mode
    fig.update_layout(
        barmode='group',
        title=f'Spend on ML/Cloud in workplaces of Kagglers from {country} compared to others',
        xaxis_title=None,
        yaxis_title='Percentage of respondents',
        xaxis = dict(
            tickmode = 'array',
            tickvals = [x for x in range(6)],
            ticktext = df_ml_spend[df_ml_spend.columns[128]].sort_values().unique()
        )
    )

    st.plotly_chart(fig)

    df_mla = df[~df.iloc[:,255].isna()]

    fig = px.pie(df_mla, df_mla.columns[255], 
                title='39% of all Kagglers primarily use their local development environment', hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide', showlegend=False)
    st.plotly_chart(fig)

    size_pct = len(df_mla[(df_mla[df_mla.columns[3]]==country) & 
                            (df_mla[df_mla.columns[255]].isin(['Local development environments (RStudio, JupyterLab, etc.)']))])*100 \
    /len(df_mla[df_mla[df_mla.columns[3]]==country])
    if size_pct > 38.8:
        title = f"For {country}, this percentage increases to {size_pct:.0f}%"
    elif size_pct < 38.8:
        title = f"For {country}, this percentage decreases to {size_pct:.0f}%"
    else:
        title = f"At {size_pct:.0f}%, it is the same for {country} too"
    fig = px.pie(df_mla[df_mla[df_mla.columns[3]]==country], df_mla.columns[255], title=title, hole=0.6)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide', showlegend=False)
    st.plotly_chart(fig)

    df_country = df_mla[df_mla.country_agg==country][df_mla.columns[255]].value_counts(normalize=True).sort_index()
    df_others = df_mla[df_mla.country_agg=='Others'][df_mla.columns[255]].value_counts(normalize=True).sort_index()

    for index in df_others.index:
            if index not in df_country.index:
                df_country[index] = 0
    df_country.sort_index(inplace=True)

    fig = go.Figure(data=[
        go.Bar(name=country, y=df_country.values*100),
        go.Bar(name='Others', y=df_others.values*100)
    ])

    # Change the bar mode
    fig.update_layout(
        barmode='group',
        title=f'Primary Tool used at Work in workplaces of Kagglers from {country} compared to others',
        xaxis_title=None,
        yaxis_title='Percentage of respondents',
        xaxis = dict(
            tickmode = 'array',
            tickvals = [x for x in range(7)],
            ticktext = df_mla[df_mla.columns[255]].sort_values().unique()
        )
    )

    st.plotly_chart(fig)