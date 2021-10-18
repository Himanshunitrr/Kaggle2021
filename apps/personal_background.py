import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os


def app():
  header_container = st.container()
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