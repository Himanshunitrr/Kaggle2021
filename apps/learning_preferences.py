import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os


def app():
  header_container = st.container()
  left_column, right_column = st.columns(2)
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

  df = pd.read_csv('./apps/kaggle_survey_2021_responses.csv', skiprows=1)
  df[df.columns[3]].replace({'United Kingdom of Great Britain and Northern Ireland':'UK',
                              'Iran, Islamic Republic of...':'Iran',
                              'United Arab Emirates':'UAE',
                              'United States of America':'USA',
                              'Viet Nam':'Vietnam'}, inplace=True)

  #   st.write(f"List of countries:\n{np.sort(df[df.columns[3]].unique())}\n")

  

  if country not in df[df.columns[3]].unique():
    raise ValueError(f'{country} not found in the list')
  df['country_agg'] = np.where(df[df.columns[3]]==country,country,'Others')

  with left_column:
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
  with right_column:
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