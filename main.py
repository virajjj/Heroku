import pandas as pd
import numpy as np
import datetime as dt
import taipy as tp
from taipy import Config, Scope
from taipy.gui import Gui, Markdown
from statsmodels.tsa.ar_model import AutoReg

import os

## Import Datset
def get_data(path_to_csv: str):
    # pandas.read_csv() returns a pd.DataFrame
    dataset = pd.read_csv(path_to_csv)
    dataset["Date"] = pd.to_datetime(dataset["Date"])
    return dataset

# Read the dataframe
path_to_csv = "dataset.csv"
dataset = get_data(path_to_csv)

n_week = 10

# Select the week based on the slider value
dataset_week = dataset[dataset["Date"].dt.isocalendar().week == n_week]

def on_change(state, var_name: str, var_value):
    if var_name == "n_week":
        # Update the dataset when the slider is moved
        #state.a=var_value
        state.dataset_week = dataset[dataset["Date"].dt.isocalendar().week == var_value]

# Import Data Nodes Configurations
# Input Data Nodes
initial_dataset_cfg = Config.configure_data_node(id="initial_dataset",
                                                 storage_type="csv",
                                                 path=path_to_csv,
                                                 scope=Scope.GLOBAL)

# We assume the current day is the 26th of July 2021.
# This day can be changed to simulate multiple executions of scenarios on different days
day_cfg = Config.configure_data_node(id="day", default_data=dt.datetime(2021, 7, 26))

n_predictions_cfg = Config.configure_data_node(id="n_predictions", default_data=40)

max_capacity_cfg = Config.configure_data_node(id="max_capacity", default_data=200)

# Remaining Data Nodes
cleaned_dataset_cfg = Config.configure_data_node(id="cleaned_dataset",
                                             cacheable=True,
                                             validity_period=dt.timedelta(days=1),
                                             scope=Scope.GLOBAL)

predictions_cfg = Config.configure_data_node(id="predictions", scope=Scope.PIPELINE)


## Functions
def clean_data(initial_dataset: pd.DataFrame):
    print("     Cleaning data")
    # Convert the date column to datetime
    initial_dataset["Date"] = pd.to_datetime(initial_dataset["Date"])
    cleaned_dataset = initial_dataset.copy()
    return cleaned_dataset


def predict_baseline(cleaned_dataset: pd.DataFrame, n_predictions: int, day: dt.datetime, max_capacity: int):
    print("     Predicting baseline")
    # Select the train data
    train_dataset = cleaned_dataset[cleaned_dataset["Date"] < day]

    predictions = train_dataset["Value"][-n_predictions:].reset_index(drop=True)
    predictions = predictions.apply(lambda x: min(x, max_capacity))
    return predictions


## Tasks
clean_data_task_cfg = Config.configure_task(id="clean_data",
                                            function=clean_data,
                                            input=initial_dataset_cfg,
                                            output=cleaned_dataset_cfg)

predict_baseline_task_cfg = Config.configure_task(id="predict_baseline",
                                                  function=predict_baseline,
                                                  input=[cleaned_dataset_cfg, n_predictions_cfg, day_cfg, max_capacity_cfg],
                                                  output=predictions_cfg)

## Pipeline
# Create the first pipeline configuration
baseline_pipeline_cfg = Config.configure_pipeline(id="baseline",
                                                  task_configs=[clean_data_task_cfg, predict_baseline_task_cfg])
# Initialize the "predictions" dataset
predictions_dataset = pd.DataFrame({"Date":[dt.datetime(2021, 6, 1)], "Historical values":[np.NaN], "Predicted values":[np.NaN]})


def create_and_submit_pipeline():
    print("Execution of pipeline...")
    # Create the pipeline from the pipeline config
    pipeline = tp.create_pipeline(baseline_pipeline_cfg)
    # Submit the pipeline (Execution)
    tp.submit(pipeline)
    return pipeline


def create_predictions_dataset(pipeline):
    print("Creating predictions dataset...")
    # Read data from the pipeline
    predictions = pipeline.predictions.read()
    day = pipeline.day.read()
    n_predictions = pipeline.n_predictions.read()
    cleaned_data = pipeline.cleaned_dataset.read()

    # Set arbitrarily the time window for the chart as 5 times the number of predictions
    window = 5 * n_predictions

    # Create the historical dataset that will be displayed
    new_length = len(cleaned_data[cleaned_data["Date"] < day]) + n_predictions
    temp_df = cleaned_data[:new_length]
    temp_df = temp_df[-window:].reset_index(drop=True)

    # Create the series that will be used in the concat
    historical_values = pd.Series(temp_df["Value"], name="Historical values")
    predicted_values = pd.Series([np.NaN] * len(temp_df), name="Predicted values")
    predicted_values[-len(predictions):] = predictions

    # Create the predictions dataset
    # Columns : [Date, Historical values, Predicted values]
    return pd.concat([temp_df["Date"], historical_values, predicted_values], axis=1)


def update_predictions_dataset(state, pipeline):
    print("Updating predictions dataset...")
    state.predictions_dataset = create_predictions_dataset(pipeline)


def predict(state):
    print("'Predict' button clicked")
    pipeline = create_and_submit_pipeline()
    update_predictions_dataset(state, pipeline)


## Scenarios (Change prediction model AND day, n_prediction, max_capacity)
# ML prediction function
def predict_ml(cleaned_dataset: pd.DataFrame, n_predictions: int, day: dt.datetime, max_capacity: int):
    print("     Predicting with ML")
    # Select the train data
    train_dataset = cleaned_dataset[cleaned_dataset["Date"] < day]

    # Fit the AutoRegressive model
    model = AutoReg(train_dataset["Value"], lags=7).fit()

    # Get the n_predictions forecasts
    predictions = model.forecast(n_predictions).reset_index(drop=True)
    predictions = predictions.apply(lambda x: min(x, max_capacity))
    return predictions

# ML prediction task
predict_ml_task_cfg = Config.configure_task(id="predict_ml",
                                            function=predict_ml,
                                            input=[cleaned_dataset_cfg, n_predictions_cfg, day_cfg, max_capacity_cfg],
                                            output=predictions_cfg)

# ML prediction pipeline
ml_pipeline_cfg = Config.configure_pipeline(id="ml", task_configs=[clean_data_task_cfg, predict_ml_task_cfg])

# Scenario configure
scenario_cfg = Config.configure_scenario(id="scenario", pipeline_configs=[baseline_pipeline_cfg, ml_pipeline_cfg])

# Set the list of pipelines names
# It will be used in a selector of pipelines
pipeline_selector = ["baseline", "ml"]
selected_pipeline = pipeline_selector[0]

# Initial variables
## Initial variables for the scenario
day = dt.datetime(2021, 7, 26)
n_predictions = 40
max_capacity = 200


def create_scenario():
    global selected_scenario

    print("Creating scenario...")
    scenario = tp.create_scenario(scenario_cfg)

    # the global selected_scenario is a newly created global variable, when scenario is changed,
    # selected scenario is updated, and can be used in other functions
    selected_scenario = scenario.id
    print(selected_scenario)

    tp.submit(scenario)


def update_chart(state):
    # Select the right scenario and pipeline
    scenario = tp.get(selected_scenario)
    pipeline = scenario.pipelines[state.selected_pipeline]
    # Update the chart based on this pipeline
    update_predictions_dataset(state, pipeline)


def submit_scenario(state):
    print("Submitting scenario...")
    # Get the selected scenario: in this current step a single scenario is created then modified here.
    scenario = tp.get(selected_scenario)

    # Conversion to the right format
    state_day = dt.datetime(state.day.year, state.day.month, state.day.day)

    # Change the default parameters by writing in the datanodes
    scenario.day.write(state_day)
    scenario.n_predictions.write(int(state.n_predictions))
    scenario.max_capacity.write(int(state.max_capacity))

    # Execute the pipelines/code
    tp.submit(scenario)

    # Update the chart when we change the scenario
    update_chart(state)

global selected_scenario

Config.configure_global_app(clean_entities_enabled=True)
tp.clean_all_entities()
# Creation of a single scenario
create_scenario()


## GUI
page = """
# Online Dashboard Using Taipy

Select week: *<|{n_week}|>*

<|{n_week}|slider|min=1|max=52|>

<|{dataset_week}|chart|type=bar|x=Date|y=Value|height=100%|width=100%|>

Dataset:
<|{dataset}|table|height=400px|width=95%|>

"""

page_scenario_manager = page + """
# Change your scenario

**Prediction date**\n\n <|{day}|date|not with_time|>

**Max capacity**\n\n <|{max_capacity}|number|>

**Number of predictions**\n\n<|{n_predictions}|number|>

<|Save changes|button|on_action={submit_scenario}|>

Select the pipeline
<|{selected_pipeline}|selector|lov={pipeline_selector}|> <|Update chart|button|on_action={update_chart}|>

<|{predictions_dataset}|chart|x=Date|y[1]=Historical values|type[1]=bar|y[2]=Predicted values|type[2]=scatter|height=80%|width=100%|>
"""

# Get all the scenarios already created
all_scenarios = tp.get_scenarios()

# Delete the scenarios that don't have a name attribute
# All the scenarios of the previous steps do not have an associated name so they will be deleted,
# this will not be the case for those created by this step
[tp.delete(scenario.id) for scenario in all_scenarios if scenario.name is None]

# Initial variable for the scenario selector
# The list of possible values (lov) for the scenario selector is a list of tuples (scenario_id, scenario_name),
# but the selected_scenario is just used to retrieve the scenario id and what gets displayed is the name of the scenario.
scenario_selector = [(scenario.id, scenario.name) for scenario in tp.get_scenarios()]
selected_scenario = None

scenario_manager_page = page + """
# Create your scenario

**Prediction date**\n\n <|{day}|date|not with_time|>

**Max capacity**\n\n <|{max_capacity}|number|>

**Number of predictions**\n\n<|{n_predictions}|number|>

<|Create new scenario|button|on_action=create_scenario|>

## Scenario 
<|{selected_scenario}|selector|lov={scenario_selector}|dropdown|>

## Display the pipeline
<|{selected_pipeline}|selector|lov={pipeline_selector}|>

<|{predictions_dataset}|chart|x=Date|y[1]=Historical values|type[1]=bar|y[2]=Predicted values|type[2]=scatter|height=80%|width=100%|>
"""


def create_name_for_scenario(state) -> str:
    name = f"Scenario ({state.day.strftime('%A, %d %b')}; {state.max_capacity}; {state.n_predictions})"

    # Change the name if it is the same as some scenarios
    if name in [s[1] for s in state.scenario_selector]:
        name += f" ({len(state.scenario_selector)})"
    return name


def update_chart(state):
    # Now, the selected_scenario comes from the state, it is interactive
    scenario = tp.get(state.selected_scenario[0])
    pipeline = scenario.pipelines[state.selected_pipeline]
    update_predictions_dataset(state, pipeline)


# Change the create_scenario function in order to change the default parameters
# and allow the creation of multiple scenarios
def create_scenario(state):
    print("Execution of scenario...")
    # Extra information for the scenario
    creation_date = state.day
    name = create_name_for_scenario(state)
    # Create a scenario
    scenario = tp.create_scenario(scenario_cfg, creation_date=creation_date, name=name)

    state.selected_scenario = (scenario.id, name)
    # Submit the scenario that is currently selected
    submit_scenario(state)


def submit_scenario(state):
    print("Submitting scenario...")
    # Get the currently selected scenario
    scenario = tp.get(state.selected_scenario[0])

    # Conversion to the right format (change?)
    day = dt.datetime(state.day.year, state.day.month, state.day.day)

    # Change the default parameters by writing in the Data Nodes
    scenario.day.write(day)
    scenario.n_predictions.write(int(state.n_predictions))
    scenario.max_capacity.write(int(state.max_capacity))
    scenario.creation_date = state.day

    # Execute the scenario
    tp.submit(scenario)

    # Update the scenario selector and the scenario that is currently selected
    update_scenario_selector(state, scenario)  # change list to scenario

    # Update the chart directly
    update_chart(state)

def update_scenario_selector(state, scenario):
    print("Updating scenario selector...")
    # Update the scenario selector
    state.scenario_selector += [(scenario.id, scenario.name)]


def on_change(state, var_name: str, var_value):
    if var_name == "n_week":
        # Update the dataset when the slider is moved
        state.dataset_week = dataset[dataset["Date"].dt.isocalendar().week == var_value]

    elif var_name == "selected_pipeline" or var_name == "selected_scenario":
        # Update the chart when the scenario or the pipeline is changed
        # Check if we can read the Data Node to update the chart
        if tp.get(state.selected_scenario[0]).predictions.read() is not None:
            update_chart(state)

# First page: slider and the chart that displays a week of the historical data
page_data_visualization = page

# Second page: create scenarios and display results
page_scenario_manager = """
# Create your scenario

<|layout|columns=1 1 1 1|
<|
**Prediction date**\n\n <|{day}|date|not with_time|>
|>

<|
**Max capacity**\n\n <|{max_capacity}|number|>
|>

<|
**Number of predictions**\n\n<|{n_predictions}|number|>
|>

<|
<br/>\n <|Create new scenario|button|on_action=create_scenario|>
|>
|>

<|part|render={len(scenario_selector) > 0}|
<|layout|columns=1 1|
<|
## Scenario \n <|{selected_scenario}|selector|lov={scenario_selector}|dropdown|>
|>

<|
## Display the pipeline \n <|{selected_pipeline}|selector|lov={pipeline_selector}|dropdown|>
|>
|>

<|{predictions_dataset}|chart|x=Date|y[1]=Historical values|type[1]=bar|y[2]=Predicted values|type[2]=scatter|height=80%|width=100%|>
|>
"""

# Create a menu with the pages
multi_pages = """
<|menu|label=Menu|lov={["Data Visualization", "Scenario Manager"]}|on_action=on_menu|>

<|part|render={page=="Data Visualization"}|""" + page_data_visualization + """|>
<|part|render={page=="Scenario Manager"}|""" + page_scenario_manager + """|>
"""


# The initial page is the "Data Visualization" page
page = "Data Visualization"
def on_menu(state, var_name: str, fct: str, var_value: list):
    # Change the value of the state.page variable in order to render the correct page
    state.page = var_value["args"][0]

# Run GUI with menu
if __name__ == '__main__':
    Gui(multi_pages).run(title="Heroku fail",
    		host='0.0.0.0',
    		port=os.environ.get('PORT', '5050'),
    		dark_mode=False,
            use_reloader=False)


