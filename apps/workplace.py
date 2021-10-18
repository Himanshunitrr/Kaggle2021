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



  df_wp_size = df[~df.iloc[:,116].isna()]

  with left_column:

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

  with right_column:

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