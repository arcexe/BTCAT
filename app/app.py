import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import decimal
import dash_table
import dash_bootstrap_components as dbc
import flask
import ipywidgets as widgets
import time

from random import randint


#https://plotly.com/python/mapbox-county-chloropleth/
#https://medium.com/plotly/introducing-plotly-express-808df010143d

#interaction=select
#https://github.com/plotly/dash-px

#https://community.plotly.com/t/plotly-express-hover-selecting-event-only-partially-working/22136/9


#app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])
#server = flask.Flask(__name__)
#server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
#app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.DARKLY])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

with open('./data/wards.geojson') as f:
    wards = json.load(f)

#n_wards = len(wards_data['features'])
#wards = [wards_data['features'][k]['properties']['ward_no] for k in range(n_wards)]

#print(wards['features'][0])

od_data = pd.read_csv('./data/od_data.csv')
od_data = od_data[od_data['ward_no'] != od_data['origin_ward']]
df = od_data.groupby(['ward_no'])['route_counts'].sum().reset_index(name='route_counts')
df['ward_no'] = df['ward_no'].astype(str).str.zfill(3)
max_count = df['route_counts'].max()

ward_data = pd.read_csv('./data/ward_population.csv')
ward_nos = ward_data['ward_no']
population = ward_data['population']

route_data = pd.read_csv('./data/od_final_grouped.csv')
route_nums = route_data['route_num']
#print(route_data.head)


OD = pd.read_csv('./data/od_matrix.csv')
OD = OD.apply (pd.to_numeric, errors='coerce')
OD = OD.fillna(0)
   
OD = OD.drop(columns=['ward_no'])

OD = OD.to_numpy()

mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"

fig = px.choropleth_mapbox(df, geojson=wards, locations='ward_no', color='route_counts',
                            featureidkey="properties.WARD_NO", color_continuous_scale="Viridis",
                            range_color=(0, max_count),
                            template='plotly_dark',
                            mapbox_style='carto-darkmatter',   
                            zoom=10 , center = {"lat" : 12.972442, "lon" : 77.580643},
                            opacity=0.5,
                            height=425,
                            width=700,
                            labels={'routes counts':'mobility'}
                            )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

def get_wards(list_wards):
    dict_list = []
    for i in list_wards:
        dict_list.append({'label': i, 'value': i})

    return dict_list

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(
    id="root",
    children=[
        html.Div(
            id="header",
            children=[
                html.H2(
                    children="Bell The CAT (Covid-19 And Travel)",
                    style={
                        'textAlign': 'left',
                        'color': colors['text']
                    },
                ),
                html.P(
                    id="description",
                    children="This tool uses SIR models to analyze Covid-19 spread and suggests BMTC routes to operate",
                    style={
                        'textAlign': 'left',
                        'color': colors['text']
                    },
                ),
            ], style= {'width': '50%', 'display': 'inline-block','vertical-align': 'top'}
        ),
        html.Div(
            id="right-header",
            children=[
                html.Div(
                    id="sub-right",
                    children=[
                        html.Div(
                            id="sub-left",
                            children=[
                                html.Div(
                                    id="number-out",
                                    style={
                                        'textAlign': 'right',
                                        'color': colors['text']
                                    },
                                ),
                                html.Div(
                                    id="image-div",
                                    children=[
                                        html.A("Download Report", href="/download/routes_report", target="_blank"),
                                    ], style= {'horizontal-align': 'right', 'font-size': 24, 'vertical-align': 'bottom'}   
                                ),
                            ], style= {'width': '50%', 'display': 'inline-block','vertical-align': 'top'}
                        ),
                    ], style= {'width': '50%', 'display': 'inline-block','vertical-align': 'top'}
                ),   
            ], style= {'width': '50%', 'display': 'inline-block','vertical-align': 'top'}
        ),   
        html.Div(
            id="app-container",
            children=[
                html.Div(
                    id="left-column",
                    children=[
                        html.Div(
                            id="slider-container",
                            children=[
                                html.P(
                                    id="slider-text",
                                    children="Drag the slider to change the percentage of routes operated",
                                    style={
                                        'textAlign': 'left',
                                        'color': colors['text']
                                    },
                                ),
                                dcc.Slider(
                                    id="operation-slider",
                                    min=0,
                                    max=10,
                                    value=2,
                                    step=None,
                                    marks={
                                        0: '0%',
                                        1: '10%',
                                        2: '20%',
                                        3: '30%',
                                        4: '40%',
                                        5: '50%',
                                        6: '60%',
                                        7: '70%',
                                        8: '80%',
                                        9: '90%',
                                        10: '100%'
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            id="heatmap-container",
                            children=[
                                html.P(
                                    "Mobility intensity",
                                    id="heatmap-title",
                                    style={
                                        'textAlign': 'left',
                                        'color': colors['text']
                                    },
                                ),
                                dcc.Graph(figure=fig),
                            ],
                        ),
                    ], style= {'width': '40%', 'display': 'inline-block','vertical-align': 'top', 'margin-left': 10}
                ),
                html.Div(
                    id="graph-container",
                    children=[
                        html.Div(
                            id="inleft-column",
                            children=[
                                html.P(
                                    "Transmission Rate (beta)",
                                    id="r0-title",
                                    style={
                                        'textAlign': 'left',
                                        'color': colors['text']
                                    },
                                ),
                                dcc.Dropdown(
                                    id='transmission-dropdown',
                                    style={
                                        'textAlign': 'left',
                                        'color': 'black',
                                        'background': '#7FDBFF'
                                    },
                                    options=[
                                        {'label': '0.5', 'value': '0.5'},
                                        {'label': '0.6', 'value': '0.6'},
                                        {'label': '0.7', 'value': '0.7'},
                                        {'label': '0.8', 'value': '0.8'},
                                        {'label': '0.9', 'value': '0.9'},
                                        {'label': '1.0', 'value': '1.0'},
                                        {'label': '1.1', 'value': '1.1'},
                                        {'label': '1.2', 'value': '1.2'},
                                        {'label': '1.3', 'value': '1.3'},
                                        {'label': '1.4', 'value': '1.4'},
                                        {'label': '1.5', 'value': '1.5'},
                                        {'label': '1.6', 'value': '1.6'},
                                        {'label': '1.7', 'value': '1.7'},
                                        {'label': '1.8', 'value': '1.8'},
                                        {'label': '1.9', 'value': '1.9'},
                                        {'label': '2.0', 'value': '2.0'},
                                        {'label': '2.1', 'value': '2.1'},
                                        {'label': '2.2', 'value': '2.2'},
                                        {'label': '2.3', 'value': '2.3'},
                                        {'label': '2.4', 'value': '2.4'},
                                        {'label': '2.5', 'value': '2.5'},
                                        {'label': '2.6', 'value': '2.6'},
                                        {'label': '2.7', 'value': '2.7'},
                                        {'label': '2.8', 'value': '2.8'},
                                        {'label': '2.9', 'value': '2.9'},
                                        {'label': '3.0', 'value': '3.0'},
                                        {'label': '3.1', 'value': '3.1'},
                                        {'label': '3.2', 'value': '3.2'},
                                        {'label': '3.3', 'value': '3.3'},
                                        {'label': '3.4', 'value': '3.4'},
                                        {'label': '3.5', 'value': '3.5'},
                                        {'label': '3.6', 'value': '3.6'},
                                        {'label': '3.7', 'value': '3.7'},
                                        {'label': '3.8', 'value': '3.8'},
                                        {'label': '3.9', 'value': '3.9'},
                                        {'label': '4.0', 'value': '4.0'}
                                    ],
                                    value = '1.4'
                                ),
                            ], style= {'width': '50%', 'display': 'inline-block','vertical-align': 'top'}
                        ),
                        html.Div(
                            id="inright-column",
                            children=[
                                html.P(
                                    "Recovery Period (days)",
                                    id="transmission-label",
                                    style={
                                        'textAlign': 'left',
                                        'color': colors['text']
                                    },
                                ),
                                dcc.Dropdown(
                                    id='recovery-dropdown',
                                    style={
                                        'textAlign': 'left',
                                        'color': 'black',
                                        'background': '#7FDBFF'
                                    },    
                                    options=[
                                        {'label': '5', 'value': '5'},
                                        {'label': '6', 'value': '6'},
                                        {'label': '7', 'value': '7'},
                                        {'label': '8', 'value': '8'},
                                        {'label': '9', 'value': '9'},
                                        {'label': '10', 'value': '10'},
                                        {'label': '11', 'value': '11'},
                                        {'label': '12', 'value': '12'},
                                        {'label': '13', 'value': '13'},
                                        {'label': '14', 'value': '14'},
                                        {'label': '15', 'value': '15'},
                                        {'label': '16', 'value': '16'},
                                        {'label': '17', 'value': '17'},
                                        {'label': '18', 'value': '18'},
                                        {'label': '19', 'value': '19'},
                                        {'label': '20', 'value': '20'},
                                        {'label': '21', 'value': '21'},
                                        {'label': '22', 'value': '22'},
                                        {'label': '23', 'value': '23'},
                                        {'label': '24', 'value': '24'},
                                        {'label': '25', 'value': '25'},
                                        {'label': '26', 'value': '26'},
                                        {'label': '27', 'value': '27'},
                                        {'label': '28', 'value': '28'},
                                        {'label': '29', 'value': '29'},
                                        {'label': '30', 'value': '30'},                            
                                    ],                                    
                                    value = '15'
                                ), 
                            ], style= {'width': '50%', 'display': 'inline-block','vertical-align': 'top'}
                        ),   
                        dcc.Graph(id="overall-data", config={'displayModeBar': False}),
                        html.P(
                            "Peak Day Later Than ...",
                            id="table-title",
                            style={
                                'textAlign': 'left',
                                'color': colors['text']
                            },
                        ),
                        dcc.Dropdown(
                            id='peak-day-dropdown',
                            style={
                                'textAlign': 'left',
                                'color': 'black',
                                'background': '#7FDBFF'
                            },
                            options=[
                                {'label': '30', 'value': '30'},
                                {'label': '35', 'value': '35'},
                                {'label': '40', 'value': '40'},
                                {'label': '45', 'value': '45'},
                                {'label': '50', 'value': '50'},
                                {'label': '55', 'value': '55'},
                                {'label': '60', 'value': '60'},
                                {'label': '65', 'value': '65'},
                                {'label': '70', 'value': '70'},
                                {'label': '75', 'value': '75'},
                            ],
                            value = '30'
                        ), 
                    ], style= {'width': '27%', 'display': 'inline-block', 'margin-right': 20}
                ),
                html.Div(
                    id="most-right-column",
                    children=[
                        html.Div(
                            id="table-container",
                            children=[
                                html.P(
                                    "Ward Prediction",
                                    id="special-title",
                                    style={
                                        'textAlign': 'left',
                                        'color': colors['text']
                                    },
                                ),
                                dcc.Dropdown(
                                    id='ward-selector',
                                    style={
                                        'textAlign': 'left',
                                        'color': 'black',
                                        'background': '#7FDBFF'
                                    },
                                    options=get_wards(ward_data['ward_no']),
                                    multi=True,
                                    value=[ward_data['ward_no'].sort_values()[0]]
                                ),
                                dcc.Graph(id='timeseries', config={'displayModeBar': False}),
                                html.P(
                                    "Peak Infection [%] Less Than ...",
                                    id="table0-title",
                                    style={
                                        'textAlign': 'left',
                                        'color': colors['text']
                                    },
                                ),                        
                                dcc.Dropdown(
                                    id='peak-infection-dropdown',
                                    style={
                                        'textAlign': 'left',
                                        'color': 'black',
                                        'background': '#7FDBFF'
                                    },
                                    options=[
                                        {'label': '1', 'value': '1'},
                                        {'label': '2', 'value': '2'},
                                        {'label': '3', 'value': '3'},
                                        {'label': '4', 'value': '4'},
                                        {'label': '5', 'value': '5'},
                                        {'label': '6', 'value': '6'},
                                        {'label': '7', 'value': '7'},
                                        {'label': '8', 'value': '8'},
                                        {'label': '9', 'value': '9'},
                                        {'label': '10', 'value': '10'},
                                        {'label': '11', 'value': '11'},
                                        {'label': '12', 'value': '12'},
                                        {'label': '13', 'value': '13'},
                                        {'label': '14', 'value': '14'},
                                        {'label': '15', 'value': '15'},
                                        {'label': '16', 'value': '16'},
                                        {'label': '17', 'value': '17'},
                                        {'label': '18', 'value': '18'},
                                        {'label': '19', 'value': '19'},
                                        {'label': '20', 'value': '20'},
                                    ],
                                    value = '5'
                                ),                                
                            ],
                        ),
                    ], style= {'width': '27%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-lrft': 10}
                ),
            ],    
        ),
        html.Div(
            id="bottom-row",
            children=[
                dcc.Graph(id="content"),   
            ], style= {'width': '100%', 'display': 'inline-block','vertical-align': 'top', 'margin-top': 10}
        ),            
    ],
)

@app.server.route('/download/routes_report') 
def download_csv():
    timestr = time.strftime("%Y%m%d-%H%M%S")

    return flask.send_file('./output/output.csv',
            mimetype='text/csv',
            attachment_filename="output" + timestr + ".csv",
            as_attachment=True,
            cache_timeout=0
            )

@app.callback(Output('content', 'figure'),
              [Input('peak-infection-dropdown', 'value'), Input('peak-day-dropdown', 'value'), Input('operation-slider', 'value'), Input('transmission-dropdown','value'), Input('recovery-dropdown', 'value')])               
def update_routes(peak_infection, peak_day,  selected_slider_value, beta, recovery_period):

    alpha = selected_slider_value/10
    print(alpha)
    
    gamma = 1.0/float(recovery_period)

    payload = run_simulation(float(beta), gamma, alpha)

    overall_data = payload["overall_data"]
    all_days_data = payload["all_days_data"]

    #good one, but commented out due to missing day
    #max_infection_df = all_days_data.groupby(['WARD_NO'])['I'].max().reset_index(name='I') 

    #sort and fetch works to get the peaks in all wards
    max_infection_df = all_days_data.sort_values('I', ascending=False).drop_duplicates(['WARD_NO'])

    print("max_infection_df ", max_infection_df.shape, max_infection_df)
    
    x = float(peak_infection)/100
    ward_df = max_infection_df[(max_infection_df['I'].astype(float) <= float(x))]
    
    print("WARD_DF ", ward_df.shape, ward_df)

    #now lets remove records whose day is less than peak_day
    #this check also eliminates the scenarios where max is so low and day is equal around 1
    result_df = ward_df[(ward_df['DAY'].astype(float) >= float(peak_day))]
    print("result_df", result_df.shape, result_df)

    route_data.rename(columns={'ward_no':'WARD_NO'}, inplace=True)
    print(route_data)

    #https://stackoverflow.com/questions/28901683/pandas-get-rows-which-are-not-in-other-dataframe
    df_all = route_data.merge(result_df.drop_duplicates(), on=['WARD_NO'], how='left', indicator=True)
    remove_routes_df = df_all[df_all['_merge'] == 'left_only']

    print("remove_routes_df", remove_routes_df.shape)

    #now get the unique values of the routes in the remove df
    remove_routes = remove_routes_df.route_num.unique()
    print("remove_routes - ", remove_routes)

    filter_df = pd.DataFrame(remove_routes, columns=['route_num'])
    print("filter_df - ", filter_df)

    all_routes = route_data.route_num.unique()
    all_routes_df = pd.DataFrame(all_routes, columns=['route_num'])

    df_ply_routes = all_routes_df.merge(filter_df.drop_duplicates(), on=['route_num'], how='left', indicator=True)
    print("df_ply_routes", df_ply_routes.shape, df_ply_routes)

    df_ply_routes_final = df_ply_routes[df_ply_routes['_merge'] == 'left_only']
    print("df_ply_routes_final", df_ply_routes_final.shape, df_ply_routes_final)
    
    df_ply_routes_final = df_ply_routes_final.drop(columns=['_merge'])
    
    routes_df = route_data.merge(df_ply_routes_final.drop_duplicates(), on=['route_num'], how='left', indicator=True)
    routes_df_final = routes_df[routes_df['_merge'] == 'both']
    
    routes_df_final = routes_df_final[[ 'WARD_NO', 'origin_ward', 'route_num', 'route_counts']]
    print(routes_df_final)
    routes_df_final.to_csv('./output/output.csv', index=False)

    print("routes_df_final", routes_df_final.shape, routes_df_final)

    route_grouped_df = routes_df_final.groupby(['WARD_NO'])['route_counts'].sum().reset_index(name='total_count') 
    print("route_grouped_df", route_grouped_df.shape, route_grouped_df)

    print("------------------- ")
    all_wards = []
    all_wards_total_counts = []
    for i in range(0, 198):
        all_wards.append(float(i + 1))
        all_wards_total_counts.append(0.0)

    oo_numpy_array = np. array([all_wards, all_wards_total_counts])
    oo_transpose = oo_numpy_array.T
    oo_transpose_list = oo_transpose. tolist()

    temp_df = pd.DataFrame(oo_transpose_list, columns=['WARD_NO', 'total_count'])
    print("temp_df ", temp_df)
    
    fig_df = pd.concat([temp_df, route_grouped_df], ignore_index=True, sort=True)
    
    fig_df = fig_df.groupby(['WARD_NO'])['total_count'].sum().reset_index(name='total_count') 


    #fig_df = temp_df.merge(route_grouped_df.drop_duplicates(), on=['WARD_NO'], how='left', indicator=True)
    print(fig_df)

    trace = go.Bar(
        x=fig_df['WARD_NO'],
        y=fig_df['total_count']
    ) 
    return {
        'data': [trace],
        'layout': go.Layout(
            title='Routes [counts] / Ward',
            template='plotly_dark',
            xaxis=go.layout.XAxis(range=[0, 198]),
            hovermode='closest',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',            
            height=300
        )
    }
    
@app.callback(Output('timeseries', 'figure'),
              [Input('ward-selector', 'value'), Input('operation-slider', 'value'), Input('transmission-dropdown','value'), Input('recovery-dropdown', 'value')])               
def update_timeseries(selected_dropdown_value, selected_slider_value, beta, recovery_period):
    ''' Draw traces of the feature 'value' based one the currently selected wards '''
    alpha = selected_slider_value/10
    #print(alpha)

    gamma = 1.0/float(recovery_period)

    payload = run_simulation(float(beta), gamma, alpha)

    overall_data = payload["overall_data"]
    all_days_data = payload["all_days_data"]
    

    # STEP 1
    trace = []  
    
    # STEP 2
    # Draw and append traces for each stock
    for ward_num in selected_dropdown_value: 
        ward_df = all_days_data[(all_days_data['WARD_NO'].astype(float) == float(ward_num))]
        
        trace.append(go.Scatter(x=ward_df['DAY'],
                                 y=ward_df['S'],
                                 mode='lines',
                                 opacity=0.7,
                                 name='S',
                                 textposition='bottom center')) 
        
        trace.append(go.Scatter(x=ward_df['DAY'],
                                 y=ward_df['I'],
                                 mode='lines',
                                 opacity=0.7,
                                 name='I',
                                 textposition='bottom center'))  

        
        trace.append(go.Scatter(x=ward_df['DAY'],
                                 y=ward_df['R'],
                                 mode='lines',
                                 opacity=0.7,
                                 name='R',
                                 textposition='bottom center'))  

    # STEP 3
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # Define Figure
    # STEP 4
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  height=400,
                  xaxis={'title':'Days', 'range': [1, 100]},
                  yaxis={'title':'Fraction of Population'},                  
              ),
             }

    return figure

def run_simulation(beta, gamma, public_trans):
    payload = {}


    N_k = np.abs(population + OD.sum(axis=0) - OD.sum(axis=1))
    locs_len = len(N_k)
    #print(locs_len)
    SIR = np.zeros(shape=(locs_len, 3))
    #print(SIR)
    SIR[:,0] = N_k

    infection_data = pd.read_csv('./data/infected_ward_population.csv')
    first_infections = infection_data['may_27']

    SIR[:, 0] = SIR[:, 0] - first_infections
    SIR[:, 1] = SIR[:, 1] + first_infections  

    #self.beta = beta
    #self.gamma = gamma
    #self.public_trans = public_trans
    #beta = 1.6
    #gamma = 0.04
    #public_trans = 0.5                                 # alpha
    R0 = beta/gamma
    beta_vec = np.full(locs_len, beta)
    gamma_vec = np.full(locs_len, gamma)
    public_trans_vec = np.full(locs_len, public_trans)

    row_sums = SIR.sum(axis=1)
    SIR_n = SIR / row_sums[:, np.newaxis]
    # make copy of the SIR matrices 
    SIR_sim = SIR.copy()
    SIR_nsim = SIR_n.copy()
  
    # run model
    day_pop_norm = []
    infected_pop_norm = []
    susceptible_pop_norm = []
    recovered_pop_norm = []

    n = 1
    all_days_data = None

    for time_step in tqdm_notebook(range(100)):
        infected_mat = np.array([SIR_nsim[:,1],]*locs_len).transpose()
        
        OD_infected = OD*infected_mat
        
        inflow_infected = OD_infected.sum(axis=0)
        
        inflow_infected = inflow_infected*public_trans_vec
        
        new_infect = beta_vec*SIR_sim[:, 0]*inflow_infected/(N_k + OD.sum(axis=0))
        new_recovered = gamma_vec*SIR_sim[:, 1]
        
        new_infect = np.where(new_infect>SIR_sim[:, 0], SIR_sim[:, 0], new_infect)
        
        SIR_sim[:, 0] = SIR_sim[:, 0] - new_infect
        SIR_sim[:, 1] = SIR_sim[:, 1] + new_infect - new_recovered
        SIR_sim[:, 2] = SIR_sim[:, 2] + new_recovered
        
        SIR_sim = np.where(SIR_sim<0,0,SIR_sim)
        
        # recompute the normalized SIR matrix
        row_sums = SIR_sim.sum(axis=1)
        SIR_nsim = SIR_sim / row_sums[:, np.newaxis]
        
        S_j = SIR_sim[:,0]/N_k
        I_j = SIR_sim[:,1]/N_k
        R_j = SIR_sim[:,2]/N_k

        days = []
        for i in range(0, 198):
            days.append(n)

        #days_array = np.array(days)
        #days_array = days_array.T
        numpy_array = np. array([days, ward_nos, S_j, I_j, R_j])
        transpose = numpy_array.T
        transpose_list = transpose. tolist()

        day_data = pd.DataFrame(transpose_list, columns=['DAY', 'WARD_NO', 'S', 'I', 'R'])
        if n is 1:
            all_days_data = day_data
        else:
            all_days_data = pd.concat([all_days_data, day_data], ignore_index=True)

        S = SIR_sim[:,0].sum()/N_k.sum()
        I = SIR_sim[:,1].sum()/N_k.sum()
        R = SIR_sim[:,2].sum()/N_k.sum()
    
        day_pop_norm.append(n)
        susceptible_pop_norm.append(S)
        infected_pop_norm.append(I)
        recovered_pop_norm.append(R)

        n = n + 1

    o_numpy_array = np. array([day_pop_norm, susceptible_pop_norm, infected_pop_norm, recovered_pop_norm])
    o_transpose = o_numpy_array.T
    o_transpose_list = o_transpose. tolist()

    df = pd.DataFrame(o_transpose_list, columns=['Day', 'S', 'I', 'R'])

    payload["overall_data"] = df
    payload["all_days_data"] = all_days_data
    
    return payload


@app.callback(Output('overall-data', 'figure'), 
                [Input('operation-slider', 'value'), Input('transmission-dropdown','value'), Input('recovery-dropdown', 'value')])
def update_slider(selected_slider_value, beta, recovery_period):
    alpha = selected_slider_value/10
    #print(alpha)
    
    gamma = 1.0/float(recovery_period)

    payload = run_simulation(float(beta), gamma, alpha)

    overall_data = payload["overall_data"]
    all_days_data = payload["all_days_data"]

    #print(overall_data)
    #print(all_days_data)

    
    ''' Draw traces of the feature 'value' based one the currently selected wards '''
    # STEP 1
    trace = []  
        
    # STEP 2
    # Draw and append traces for each stock
    trace.append(go.Scatter(x=overall_data['Day'],
                                 y=overall_data['S'],
                                 mode='lines',
                                 opacity=0.7,
                                 name='S',
                                 textposition='bottom center')) 
        
    trace.append(go.Scatter(x=overall_data['Day'],
                                 y=overall_data['I'],
                                 mode='lines',
                                 opacity=0.7,
                                 name='I',
                                 textposition='bottom center'))  
    
    trace.append(go.Scatter(x=overall_data['Day'],
                                 y=overall_data['R'],
                                 mode='lines',
                                 opacity=0.7,
                                 name='R',
                                 textposition='bottom center'))  

    # STEP 3
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # Define Figure
    # STEP 4
    figure_test = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  height=400,
                  xaxis={'title':'Days', 'range': [1, 100]},
                  yaxis={'title':'Fraction of Population'},                  

              ),

              }
    
    return figure_test

#https://community.plotly.com/t/error-with-gunicorn/8247/12
#app.run_server(debug=True, use_reloader=False)
if __name__ == "__main__":
    app.server.run(debug=True, threaded=True)
