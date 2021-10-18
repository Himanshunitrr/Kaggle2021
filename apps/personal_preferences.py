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
