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
