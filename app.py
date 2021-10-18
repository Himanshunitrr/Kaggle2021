import streamlit as st
from multiapp import MultiApp
from apps import demographics, personal_background, personal_preferences, learning_preferences, workplace



app = MultiApp()
app.add_app("Demographics", demographics.app)
app.add_app("PersonalBackground", personal_background.app)
app.add_app("PersonalPreferences", personal_preferences.app)
app.add_app("LearningPreferences", learning_preferences.app)
app.add_app("Workplace", workplace.app)

app.run()