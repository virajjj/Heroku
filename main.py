#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
SFDR = pd.read_csv('EU_SFDR.csv')
TCFD = pd.read_csv('TCFD.csv')
Global = pd.read_csv('UN_globalcompact.csv')
SDG = pd.read_csv('UN_SDG.csv')


# In[2]:


EU_tax = SDG.loc[:, ~SDG.columns.str.contains('^Unnamed')]
EU_SFDR = SFDR.loc[:, ~SFDR.columns.str.contains('^Unnamed')]
TCFD = TCFD.loc[:, ~TCFD.columns.str.contains('^Unnamed')]
UN_global = Global.loc[:, ~Global.columns.str.contains('^Unnamed')]


# In[3]:


clnd_EU_tax = SDG.dropna()
clnd_EU_SFDR = SFDR.dropna()
clnd_TCFD = TCFD.dropna()
clnd_UN_global = Global.dropna()
#clnd_IFRS = IFRS.dropna()


# In[4]:


clnd_EU_tax


# In[5]:


clnd_EU_SFDR


# In[6]:


clnd_TCFD


# In[7]:


clnd_UN_global


# In[ ]:





# In[8]:


from fuzzywuzzy import fuzz
from thefuzz import process

a = len(clnd_EU_tax)
b = len(clnd_EU_SFDR)
similarity = np.empty((a,b), dtype= float)
similarity.round(decimals=2, out=None)

for i, ac in enumerate(clnd_EU_tax['Related Keywords from topic']):
    for j, bc in enumerate(clnd_EU_SFDR['Related Keywords from topic']):
        if i > j:
            continue
        if i == j:
            sim = 100
        else:
            sim = fuzz.ratio(ac, bc) # Use whatever metric you want here
                                     # for comparison of 2 strings.

        similarity[i, j] = sim
        similarity[j, i] = sim

EU_SDG_SFDR = pd.DataFrame(similarity, index=clnd_EU_tax['ID Column'], columns=clnd_EU_SFDR['ID Column'])
EU_SDG_SFDR = EU_SDG_SFDR.round(10)


# In[9]:


EU_SDG_SFDR


# In[10]:


from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
#fig = px.density_heatmap(EU_SDG_SFDR, x=clnd_EU_tax['ID Column'], clnd_EU_SFDR['ID Column'],
 #                        marginal_x="rug",marginal_y="histogram")
fig1 = px.imshow(EU_SDG_SFDR, text_auto=True, aspect="auto")    
fig1.show()


# In[11]:


from fuzzywuzzy import fuzz

a = len(clnd_EU_tax)
b = len(clnd_UN_global)
similarity = np.empty((a,b), dtype=int)
#similarity.round(decimals=20, out=None)

for i, ac in enumerate(clnd_EU_tax['Related Keywords from topic']):
    for j, bc in enumerate(clnd_UN_global['Related Keywords from topic']):
        if i > j:
            continue
        if i == j:
            sim = 100
        else:
            sim = fuzz.ratio(ac, bc) # Use whatever metric you want here
                                     # for comparison of 2 strings.

        similarity[i, j] = sim
        similarity[j, i] = sim

EU_SDG_UN = pd.DataFrame(similarity, index=clnd_EU_tax['ID Column'], columns=clnd_UN_global['ID Column'])
EU_SDG_UN = EU_SDG_UN.round(10)


# In[12]:


EU_SDG_UN


# In[13]:


from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
#fig = px.density_heatmap(EU_SDG_SFDR, x=clnd_EU_tax['ID Column'], clnd_EU_SFDR['ID Column'],
 #                        marginal_x="rug",marginal_y="histogram")
fig2 = px.imshow(EU_SDG_UN, text_auto=True, aspect="auto")    
fig2.show()


# In[14]:


from fuzzywuzzy import fuzz
from thefuzz import process

a = len(clnd_EU_tax)
b = len(clnd_TCFD)
similarity = np.empty((a,b), dtype=float)
similarity.round(decimals=2, out=None)

for i, ac in enumerate(clnd_EU_tax['Related Keywords from topic']):
    for j, bc in enumerate(clnd_TCFD['Keywords']):
        if i > j:
            continue
        if i == j:
            sim = 100
        else:
            sim = fuzz.ratio(ac, bc) # Use whatever metric you want here
                                     # for comparison of 2 strings.

        similarity[i, j] = sim
        similarity[j, i] = sim

EU_SDG_TCFD = pd.DataFrame(similarity, index=clnd_EU_tax['ID Column'], columns=clnd_TCFD['ID Column'])
#EU_SDG_TCFD = EU_SDG_SFDR.round(4)


# In[15]:


EU_SDG_TCFD


# In[16]:


from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
#fig = px.density_heatmap(EU_SDG_SFDR, x=clnd_EU_tax['ID Column'], clnd_EU_SFDR['ID Column'],
 #                        marginal_x="rug",marginal_y="histogram")
fig3 = px.imshow(EU_SDG_TCFD, text_auto=True, aspect="auto")    
fig3.show()


# In[ ]:





# In[ ]:


import dash
from dash import dcc, html
from flask import Flask
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly import graph_objs as go

server = Flask(__name__)
app = dash.Dash(__name__, server = server, external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
Header_component = html.H1("Taxonomy heatmaps", style = {'color': "darkcyan"})

app.layout = html.Div(
    [
        dbc.Row([
            Header_component
        ]),
        dbc.Row([dbc.Col([dcc.Graph(figure = fig1)])]),
        dbc.Row([dbc.Col([dcc.Graph(figure = fig3)])]),
        dbc.Row([dbc.Col([dcc.Graph(figure = fig2)])]),
    ]
)
app.run_server(port = 8090, dev_tools_ui= True, dev_tools_hot_reload = True, threaded=True, use_reloader=False, debug = True)


# In[ ]:




