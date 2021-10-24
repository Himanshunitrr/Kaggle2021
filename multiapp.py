"""Frameworks for running multiple Streamlit applications as a single app.
"""
import streamlit as st

class MultiApp:
  """Framework for combining multiple streamlit applications.
  Usage:
      def foo():
          st.title("Hello Foo")
      def bar():
          st.title("Hello Bar")
      app = MultiApp()
      app.add_app("Foo", foo)
      app.add_app("Bar", bar)
      app.run()
  It is also possible keep each application in a separate file.
      import foo
      import bar
      app = MultiApp()
      app.add_app("Foo", foo.app)
      app.add_app("Bar", bar.app)
      app.run()
  """
  def __init__(self):
      self.apps = []


  def add_app(self, title, func):
    """Adds a new application.
    Parameters
    ----------
    func:
        the python function to render this app.
    title:
        title of the app. Appears in the dropdown in the sidebar.
    """
    
    self.apps.append({
        "title": title,
        "function": func
    })

  def run(self):
    st.sidebar.title('2021 Kaggle Machine Learning & Data Science Survey')
    st.sidebar.markdown("This is visualization of data collected by Kaggle which can be found at [Kaggle Dataset](https://www.kaggle.com/c/kaggle-survey-2021/data). Get interesting insights about ML & DS and you can even compare any country with the world. These visualizations are from the [notebook](https://www.kaggle.com/siddhantsadangi/kaggle-2021-your-country-vs-the-world/notebook) of [Siddhant Sadangi](https://www.kaggle.com/siddhantsadangi). Currently the website is not responsive. Website created by [Himanshu Maurya](https://www.kaggle.com/himanshunitrr).")
    app = st.sidebar.selectbox(
        'Explore',
        self.apps,
        format_func=lambda app: app['title'])

    app['function']()