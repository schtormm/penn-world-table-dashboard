import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from plotly.subplots import make_subplots


# Load and prepare the data
def load_data():
    """Load and clean the chess games dataset"""
    df = pd.read_csv('cleaned_V11.csv')

    # Ensure numeric types for columns we use
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['rgdpo'] = pd.to_numeric(df['rgdpo'], errors='coerce')
    df['pop'] = pd.to_numeric(df['pop'], errors='coerce')
    # convert avh from ""1,953" to 1953
    df['avh'] = df['avh'].str.replace(',', '').astype(float)
    df["ccon"] = pd.to_numeric(df["ccon"], errors='coerce')


    # Compute GDP per capita but avoid division by zero / invalid values
    # where population is missing or zero we'll set result to NaN
    df['gdp_per_capita'] = df['rgdpo'] / df['pop']

    # Replace inf/-inf with NaN and drop rows that can't be plotted
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Keep rows that have a valid year, country and finite gdp_per_capita
    df = df.dropna(subset=['year', 'country', 'gdp_per_capita'])

    # Convert year to integer for plotting and sort
    df['year'] = df['year'].astype(int)
    df.sort_values(['country', 'year'], inplace=True)

    #population is in millions, convert to absolute numbers
    df['pop'] = df['pop'] * 1_000_000

    # Calculate year-over-year GDP growth rate for each country
    # First sort by country and year to ensure correct calculation
    df = df.sort_values(['country', 'year'])
    
    # Calculate the growth rate using pct_change within each country group
    df['gdp_growth'] = df.groupby('country')['gdp_per_capita'].pct_change() * 100
    
    # Calculate average growth rate for each country
    country_avg_growth = df.groupby('country')['gdp_growth'].mean().round(2)
    # Add it back to the dataframe
    df['avg_growth_rate'] = df['country'].map(country_avg_growth)

    return df

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Penn World Table Dashboard"

# Load the data
df = load_data()
recession = False  # Placeholder for recession prediction logic

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Penn World Table Dashboard", className='text-center text-primary mb-4'), width=12)
    ]),
    
    # Add cards in a row with appropriate spacing
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Average GDP Per Capita Growth Rate", className="card-title"),
                    html.Div(id='avg-growth-rate', className="card-text")
                ])
            ],)
        ], width=6, style={"padding": "2px"}),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Will there be a recession in the next FILL IN years", className="card-title"),
                    # read recession variable: green text if NO, red if YES
                    html.H5("Yes" if recession else "Yes", className="card-text bold", style={"color": "red" if not recession else "green"})
                ])
            ],)
        ], width=6, style={"padding": "2px"}),
    ], className="mb-4"),
  
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': country, 'value': country} for country in df['country'].unique()],
                value='Netherlands',
                clearable=False,
                className='mb-4'
            )
        ], width=12)
    ]),
    
    # 2x2 grid of graphs
    dbc.Row([
        # First row of graphs
        dbc.Col([
            dcc.Graph(id='gdp-graph')
        ]),
        dbc.Col([
            dcc.Graph(id='consumption-graph')
        ]),
    
    ]),
    
    dbc.Row([
        # Second row of graphs
        dbc.Col([
            dcc.Graph(id='hours-worked-graph')
        ], width=6),
    ]),
], fluid=True)

# Callback to update the average growth rate card
@callback(
    Output('avg-growth-rate', 'children'),
    Input('country-dropdown', 'value')
)
def update_growth_rate(selected_country):
    # Get the average growth rate for the selected country
    growth_rate = df[df['country'] == selected_country]['avg_growth_rate'].iloc[0]
    return f"{growth_rate:.2f}% per year"

# Define callbacks to update the graphs
@callback(
    Output('gdp-graph', 'figure'),
    Input('country-dropdown', 'value')
)
def update_gdp_graph(selected_country):
    filtered_df = df[df['country'] == selected_country]

    # If no data for the country, return an empty figure with a message
    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f'No data available for {selected_country}',
            xaxis_title='Year', yaxis_title='GDP Per Capita (constant 2017 US$)'
        )
        return fig

    fig = px.line(filtered_df, x='year', y='gdp_per_capita', title=f'GDP Per Capita Over Time in {selected_country}')
    fig.update_layout(xaxis_title='Year', yaxis_title='GDP Per Capita (constant 2017 US$)')

    # Fix axis ranges per selection so prior selections/empty values don't cause the axes to grow
    # compute min/max and add a small margin
    x_min = int(filtered_df['year'].min())
    x_max = int(filtered_df['year'].max())

    y_min = float(filtered_df['gdp_per_capita'].min())
    y_max = float(filtered_df['gdp_per_capita'].max())
    # If there's only a single year/point, expand a little so the point is visible
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    # Add a relative margin on y (5%) but handle y_min==y_max
    if y_min == y_max:
        if y_min == 0:
            y_min = -1
            y_max = 1
        else:
            y_min = y_min * 0.95
            y_max = y_max * 1.05
    else:
        y_margin = (y_max - y_min) * 0.05
        y_min = max(0, y_min - y_margin)
        y_max = y_max + y_margin

    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])

    return fig

@callback(
    Output('consumption-graph', 'figure'),
    Input('country-dropdown', 'value')
)
def update_consumption_graph(selected_country):
    filtered_df = df[df['country'] == selected_country]
    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f'No data available for {selected_country}',
            xaxis_title='Year', yaxis_title='Consumption, at current PPPs (in mil. 2017US$)'
        )
        return fig

    fig = px.line(filtered_df, x='year', y='ccon', title=f'Consumption Over Time in {selected_country}')
    fig.update_layout(xaxis_title='Year', yaxis_title='Consumption, at current PPPs (in mil. 2017US$)')

    # Fix axis ranges per selection so prior selections/empty values don't cause the axes to grow
    # compute min/max and add a small margin
    x_min = int(filtered_df['year'].min())
    x_max = int(filtered_df['year'].max())

    y_min = float(filtered_df['ccon'].min())
    y_max = float(filtered_df['ccon'].max())

    # If there's only a single year/point, expand a little so the point is visible
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    # Add a relative margin on y (5%) but handle y_min==y_max
    if y_min == y_max:
        if y_min == 0:
            y_min = -1
            y_max = 1
        else:
            y_min = y_min * 0.95
            y_max = y_max * 1.05
    else:
        y_margin = (y_max - y_min) * 0.05
        y_min = max(0, y_min - y_margin)
        y_max = y_max + y_margin

    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])

    return fig


@callback(
    Output('hours-worked-graph', 'figure'),
    Input('country-dropdown', 'value')
)
def update_hours_worked_graph(selected_country):
    filtered_df = df[df['country'] == selected_country]
    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f'No data available for {selected_country}',
            xaxis_title='Year', yaxis_title='Average Annual Hours Worked'
        )
        return fig

    fig = px.line(filtered_df, x='year', y='avh', title=f'Average Annual Hours Worked Over Time in {selected_country}')
    fig.update_layout(xaxis_title='Year', yaxis_title='Average Annual Hours Worked')

    # Fix axis ranges per selection so prior selections/empty values don't cause the axes to grow
    # compute min/max and add a small margin
    x_min = int(filtered_df['year'].min())
    x_max = int(filtered_df['year'].max())

    y_min = float(filtered_df['avh'].min())
    y_max = float(filtered_df['avh'].max())

    # If there's only a single year/point, expand a little so the point is visible
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    # Add a relative margin on y (5%) but handle y_min==y_max
    if y_min == y_max:
        if y_min == 0:
            y_min = -1
            y_max = 1
        else:
            y_min = y_min * 0.95
            y_max = y_max * 1.05
    else:
        y_margin = (y_max - y_min) * 0.05
        y_min = max(0, y_min - y_margin)
        y_max = y_max + y_margin

    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])

    return fig



# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)