# phm2021_data_challenge

WORK IN PROGRESS - much of the code in [/package](https://github.com/darrahts/phm2021_data_challenge/tree/main/package) is getting refactored

database implementation for the N-CMAPSS dataset that uses the same schema as the [uav project](https://github.com/darrahts/uavTestbed) (with some improvements and minor differences) with a python api. 

Check out [jupyter notebook](https://github.com/darrahts/phm2021_data_challenge/blob/main/notebooks/database_api_ncmapss.ipynb) for some api examples of how to use the api for data insertion and extraction.

## Steps  
1. clone the repository  `git clone https://github.com/darrahts/phm2021_data_challenge.git`  
2. make [setup.sh](https://github.com/darrahts/phm2021_data_challenge/blob/main/setup.sh) executable `cd phm2021_data_challenge && chmod +x setup.sh`  
3. get the raw data from the Prognostics Center of Excellence (NASA) [here](https://ti.arc.nasa.gov/c/47/), unzip it, and remove “N-CMAPSS_DS02-006” and “N-CMAPSS_DS08d-010”
4. put the .h5 data files in the [/data_h5 directory](https://github.com/darrahts/phm2021_data_challenge/tree/main/data_h5) (.h5 files are ignored in gitignore)
5. execute [setup.sh](https://github.com/darrahts/phm2021_data_challenge/blob/main/setup.sh) to install the database, configure the user, set up the table schema, and populate the database (user prompts y/n for different steps)  
`./setup.sh`  
6. \[optional] create a conda environment   
`conda env create --file environment.yml` (default name is tfgpu)    
`conda activate tfgpu`   
7. start jupyter lab (or notebook)   
`jupyter lab`   
8. open [database_api_ncmapss.ipynb](https://github.com/darrahts/phm2021_data_challenge/blob/main/notebooks/database_api_ncmapss.ipynb) and if in step 5 you selected the option to populate the database, skip steps 1-9


### NOTES

if you are using aws secrets you want
```
boto3  
base64  
oath2client  
oathlib  
openssl  
```

and in step 4 of [database_api_ncmapss.ipynb](https://github.com/darrahts/phm2021_data_challenge/blob/main/notebooks/database_api_ncmapss.ipynb) you will change   
`params = utils.get_aws_secret("/secret/ncmapssdb")` to match your secret name  

otherwise,   
```
params = {'datasource.username': $USER, # the username of the logged in user
            'datasource.password': <password entered in step 4>, 
            'datasource.database': 'ncmapss_db', # <- NO CHANGE 
            'datasource.url': 'localhost', # <- or your database installation location
            'datasource.port': '5432'} # <- most likely don't change
```

 besides the typical pandas/numpy stack and other packages that are probably already on your system, you will need  
```
psycopg2
h5py
```

If you created a conda environment from the [environment.yml](https://github.com/darrahts/phm2021_data_challenge/blob/main/environment.yml), you have all of all of the required dependencies