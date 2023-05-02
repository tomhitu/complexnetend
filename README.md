# Welcom to use complex-node project

## Online Web App
:link: https://complexnetwork.uk/

## Local development
:package: The `requirements.txt` file stores all the dependencies for the project


## install dependencies
```pip install -r requirements.txt```

## create requirements.txt file
```pip freeze > requirements.txt```

## File structure
:file_folder: `application.py` work for web app with port. <br>
:file_folder: `__init__.py` work for local toolbox. <br>
:file_folder: `example.py` introduces how to use toolbox. <br>
:file_folder: `test.py` introduces how to read and save data. <br>


## Deploying with PythonAnywhere
:rocket: use PythonAnywhere to deploy the app. <br>
1. console -> bash -> ```cd /home/your-username/my-project```
2. console -> bash -> ```exec $BASH --rcfile <(echo "PS1='file-console \$ '") -i```
3. consoles -> bash -> ```pip install -r requirements.txt```

## Updating Dependencies
:refresh: To update dependencies:<br>
```pip install --upgrade pip```

:bulb: Note: Remember to update requirements.txt if there are any new dependencies or updated versions.


## `test.py` file
1. pre-analysis and clean data<br>
    To preprocessing Chinese railway network dataset.<br>
    :pencil2: input: None. <br>
    :white_check_mark: output: cleaned file. <br>
2. train new model for prediction. <br>
    To train a MLP model for edge prediction with Chinese railway network dataset.<br>
    :pencil2: input: MLP configurations: epochs, lr, batch_size, step_schedule, schudule_gamma.<br>
    :white_check_mark: output: MLP trained model.<br>
3. prediction of edges <br>
    How to add new node into the graph with lon, lat and pre-trained model.<br>
    To input new node location and use MLP to predict a possible connection and feature.<br> 
    :pencil2: input: latitude, longitude.<br>
    :white_check_mark: output: all possible connection node with edge feature prediction (speed, distance, travel time).<br>
4. transfer data into front-end.<br>
    To transfer data from back-end to front-end.<br>
    :pencil2: input: csv file.<br>
    :white_check_mark: output: json file.<br>
5. delete node and choose different map type.<br>
    To delete 1 node in network and observe the changing of network.<br>
    :pencil2: input: node number in Chinese railway network dataset.<br>
    :white_check_mark: output: table compare network properties before vs. after delete note.<br>
6. Dijkstra algorithm with shortest path.<br>
    To find a shortest path between two node.<br>
    :pencil2: input: start node number, target node number.<br>
    :white_check_mark: output: list of travel path.<br>
7. Complex network analysis for resilience properties.<br>
    To analyse the effect of network attack in severl pattern.<br>
    :pencil2: input: None.<br>
    :white_check_mark: output: New network properties.<br>
