import streamlit as st
from multiapp import MultiApp
from apps import demographics, personal_background, personal_preferences, learning_preferences, workplace


st.set_page_config(layout='wide')
st.title('2021 Kaggle Machine Learning & Data Science Survey')
st.markdown("This is visualization of data collected by Kaggle which can be found at [Kaggle Dataset](https://www.kaggle.com/c/kaggle-survey-2021/data). Get interesting insights about ML & DS and you can even compare any country with the world. These visualizations are copied from the [notebook](https://www.kaggle.com/siddhantsadangi/kaggle-2021-your-country-vs-the-world/notebook) of [Siddhant Sadangi](https://www.kaggle.com/siddhantsadangi). Currently the website is not responsive. Website created by [Himanshu Maurya](https://www.kaggle.com/himanshunitrr).")
app = MultiApp()
app.add_app("Demographics", demographics.app)
app.add_app("PersonalBackground", personal_background.app)
app.add_app("PersonalPreferences", personal_preferences.app)
app.add_app("LearningPreferences", learning_preferences.app)
app.add_app("Workplace", workplace.app)

app.run()