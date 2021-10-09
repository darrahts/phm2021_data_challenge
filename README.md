# phm2021_data_challenge

database implementation for the N-CMAPSS dataset that uses the same schema as the [uav project](https://github.com/darrahts/uavTestbed) (with some improvements) with a python api. 

Check out [jupyter notebook](https://github.com/darrahts/phm2021_data_challenge/blob/main/notebooks/database_api_ncmapss.ipynb) for some api examples of how to use the api for data insertion (data extraction in progress).

## Steps  
1. clone the repository  `git clone https://github.com/darrahts/phm2021_data_challenge.git`  
2. make setup.sh executable `cd phm2021_data_challenge && chmod +x setup.sh`  
3. execute setup script to install the database, configure the user, set up the table schema, and populate the database  
`./setup.sh`  

### TODO add python environment file  
having issues exporting... (work in progress)  

basically if you are using aws secrets  you want
```
boto3  
base64  
oath2client  
oathlib  
openssl  
```

otherwise besides the typical pandas/numpy stack and other packages that are probably already on your system, you will need  
```
psycopg2
h5py
```
