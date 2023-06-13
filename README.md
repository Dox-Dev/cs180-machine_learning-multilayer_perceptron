# cs180-group19
## Utilizing Multilayer Perceptron for Physical Activity Level

The web application is powered by Streamlit, which is a convenient app builder made specifically to create shareable web apps from pure Python.

First, install Anaconda Navigator and Streamlit via the guide given in the documentations:
https://docs.streamlit.io/library/get-started/installation

The guide also gives insight on how to run our web application.

Next, make sure that all the used libraries aside from Streamlit are installed and updated.
```
pip install pandas
pip install -U scikit-learn
```

Now run Anaconda Navigator, select a new environment, and run the following line of code in the pop-up terminal:
```
streamlit run sleep_predictor.py
```

Make sure that ``sleep_predictor.py`` is within the directory selected by the terminal. Use the ``cd`` command to redirect to the directory ``sleep_predictor.py`` is located in.

Now the web application is up and running! 
