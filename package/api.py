import boto3
import base64
import os
from botocore.exceptions import ClientError
import json
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import traceback



class DB:
    """database interface class"""

    @staticmethod
    def connect(params: dict) -> [psycopg2.extensions.connection, psycopg2.extensions.cursor]:
        """
            @brief: connects to the database

            @params:
                params: dictionary of db connection parameters

            @returns:
                db: the database
                cur: the cursor
        """
        if "datasource.username" in params:
            temp = {
                "user": params["datasource.username"],
                "password": params["datasource.password"],
                "database": params["datasource.database"],
                "host": params["datasource.url"],
                "port": params["datasource.port"]
            }
            params = temp
        try:
            print("[INFO] connecting to db.")
            db = psycopg2.connect(**params)
            print("[INFO] connected.")
            cur = db.cursor()
        except Exception as e:
            print("[ERROR] failed to connect to db.")
            print(e)
            return []
        return [db, cur]

    @staticmethod
    def execute(sql_query: str, database: psycopg2.extensions.connection) -> pd.DataFrame:
        """
            @brief: shorthand sql style execution

            @params:
                sql_query: the query string to execute
                database: the database to execute on

            @returns: a pandas table of the query results
        """
        try:
            if ('insert' in sql_query):
                print("insert here")
                pd.read_sql_query(sql_query, database)
            else:
                return pd.read_sql_query(sql_query, database)
        except Exception as e:
            print(e)
            print(traceback.print_exc())
            if ('NoneType' in str(e)):
                print("ignoring error")
            return pd.DataFrame()

    @staticmethod
    def get_tables(db: psycopg2.extensions.connection) -> pd.DataFrame:
        """Returns a DataFrame of the tables in a given database"""
        return DB.execute("""SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'""", db)

    @staticmethod
    def get_fields(tb: str, db: psycopg2.extensions.connection) -> pd.DataFrame:
        """Returns the fields (column headers) for a given table"""
        return DB.execute("""SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE table_name = '{}';""".format(tb),
                          db)


class Utils:
    """
        @brief: static class for utility functions

        @definitions:
            get_aws_secret(secret_name, region_name)
    """

    @staticmethod
    def get_aws_secret(secret_name: str = "", region_name: str = "us-east-1") -> {}:
        """
            @brief: retrieves a secret stored in AWS Secrets Manager. Requires AWS CLI and IAM user profile properly configured.

            @input:
                secret_name: the name of the secret
                region_name: region of use, default=us-east-1

            @output:
                secret: dictionary
        """
        client = boto3.session.Session().client(service_name='secretsmanager', region_name=region_name)
        secret = '{"None": "None"}'
        if (len(secret_name) < 1):
            print("[ERROR] no secret name provided.")
        else:
            try:
                res = client.get_secret_value(SecretId=secret_name)
                if 'SecretString' in res:
                    secret = res['SecretString']
                elif 'SecretBinary' in res:
                    secret = base64.b64decode(res['SecretBinary'])
                else:
                    print("[ERROR] secret keys not found in response.")
            except ClientError as e:
                print(e)

        return json.loads(secret)

    @staticmethod
    def get_config(filename: str = r'', section: str = 'postgresql') -> {}:
        """
            @brief: [DEPRECIATED] parses a database configuration file

            @params:
                filename: configuraiton file with .ini extension
                section: the type of db

            @returns:
                config: dictionary of database configuration settings
        """
        from configparser import ConfigParser
        parser = ConfigParser()
        config = {}

        try:
            parser.read(filename)
        except:
            print("[ERROR] failed to read file. does it exist?")
            return config

        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                config[param[0]] = param[1]
        else:
            print('[ERROR] Section {0} not found in the {1} file'.format(section, filename))
            return config

        return config


# usage
"""
params = Utils.get_aws_secret("/secret/uav_db")
db, cur =  DB.connect(params)
del(params)
db_tables = DB.get_tables(db)
print(db_tables)
"""

# output
"""
   [INFO] connecting to db.
   [INFO] connected.
                     table_name
   0                   model_tb
   1        system_parameter_tb
   2                eq_motor_tb
   3   degradation_parameter_tb
   4                 mission_tb
   5         pg_stat_statements
   6          battery_sensor_tb
   7              experiment_tb
   8             twin_params_tb
   9              trajectory_tb
   10               airframe_tb
   11             asset_type_tb
   12                  asset_tb
   13       default_airframe_tb
   14               dc_motor_tb
   15            eqc_battery_tb
   16                    uav_tb
   17                      test

   Process finished with exit code 0
"""
#

