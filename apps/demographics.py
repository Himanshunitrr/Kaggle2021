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

