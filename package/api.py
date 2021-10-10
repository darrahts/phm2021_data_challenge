# import boto3
# import base64
# import os
# from botocore.exceptions import ClientError
# import json
# import numpy as np
# from datetime import datetime, timedelta
# import sys
import psycopg2
import pandas as pd
import sys
import traceback
try:
    import package.utils as utils
except:
    import utils


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
            @brief: shorthand sql style execution, preferred method for select statements

            @params:
                sql_query: the query string to execute
                database: the database to execute on

            @returns: a pandas table of the query results
        """
        try:
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
    def table_exists(tb: str = '',
                     db: psycopg2.extensions.connection = None) -> bool:
        res = DB.execute(f"""select * from information_schema.tables where table_schema = 'public' and table_name = '{tb}';""", db)
        if len(res.table_name.values) > 0:
            return True
        else:
            return False



    @staticmethod
    def get_fields(tb: str = None,
                   as_list: bool = True,
                   db: psycopg2.extensions.connection = None) -> pd.DataFrame or list:
        """Returns the fields (column headers) for a given table"""

        assert tb is not None and db is not None, '[ERROR] must supply the name of the table (tb=__) and psycopg2.extensions.connection (db=__)'

        res = DB.execute("""SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE table_name = '{}';""".format(tb), db)
        if not as_list:
            return res
        else:
            return [col for cols in res.values.tolist() for col in cols]




    @staticmethod
    def batch_insert(df: pd.DataFrame = None,
                     tb: str = '',
                     num_batches: int = 10,
                     db: psycopg2.extensions.connection = None,
                     cur: psycopg2.extensions.cursor = None,
                     verbose: bool = False) -> int:
        """
        returns the id of the last record inserted
        """
        assert tb in DB.get_tables(db).values, f'[ERROR] table <{tb}> does not exist'
        assert all(col in DB.get_fields(f'{tb}', as_list=True, db=db) for col in list(df.columns)), f'[ERROR] target table <{tb}> does not contain all passed columns <{list(df.columns)}>'
        # about 100x faster than a for loop, 2x faster than using executemany or execute_batch
        # uses a generator to bypass memory issues
        values = list(tuple(x) for x in zip(*(df[x].values.tolist() for x in list(df.columns))))
        for i, chunk in utils.chunk_generator(values, int(len(values)/num_batches)):
            if verbose:
                print(f"inserting batch {i} of {num_batches}...")
            vals = str(chunk).replace('[', '').replace(']', '')
            try:
                cur.execute(f"""INSERT INTO {tb} {str(tuple(df.columns)).replace("'", '"')} VALUES {vals};""")
                db.commit()
            except Exception as e:
                print(e)
                db.rollback()
        return DB.execute(f"select max(id) from {tb};", db).values[0][0]




    @staticmethod
    def _create_asset_type(asset_type: str = None,
                           subtype: str = None,
                           description: str = None,
                           db: psycopg2.extensions.connection = None,
                           cur: psycopg2.extensions.cursor = None) -> pd.DataFrame:
        """
        returns the asset type as a dataframe
        """
        assert asset_type is not None and subtype is not None, "[ERROR] must supply <asset_type>(str) and <subtype>(str)."

        try:
            if description is not None:
                cur.execute(f"""INSERT INTO asset_type_tb ("type", "subtype","description") values ('{asset_type}', '{subtype}', '{description}');""")
            else:
                cur.execute(f"""INSERT INTO asset_type_tb ("type", "subtype") values ('{asset_type}', '{subtype}');""")
            db.commit()
        except psycopg2.errors.UniqueViolation:
            print("[INFO] asset_type already exists.")
        asset_type_df = DB._get_asset_type(asset_type=asset_type, subtype=subtype, id_only=False, db=db)

        return asset_type_df




    @staticmethod
    def _get_asset_type(asset_type: str = None,
                        subtype: str = None,
                        type_id: int = None,
                        id_only: bool = True,
                        db: psycopg2.extensions.connection = None) -> int:
        """
        returns the asset type id or the asset type as dataframe
        """
        assert type_id or (asset_type is not None and subtype is not None) is not None, "[ERROR] must supply <asset_type>(str) and <subtype>(str), or <type_id>(int)."
        try:
            if type_id is not None:
                id_only = False
                statement = f"""select * from asset_type_tb where "id" = {type_id}"""
            else:
                statement = f"""select * from asset_type_tb where "type" ilike '%{asset_type}%' and "subtype" ilike '%{subtype}%';"""

            res = DB.execute(statement, db)
            if id_only:
                return int(res.id.values[0])
            else:
                return res
        except IndexError:
            print("[ERROR] asset_type does not exist or invalid parameters passed")
            return -1



    @staticmethod
    def _create_asset(type_id: int = None,
                      owner: str = '',
                      process_id: int = None,
                      serial_number: str = '',
                      common_name: str = '',
                      age: float = None,
                      eol: float = None,
                      rul: float = None,
                      units: str = None,
                      db: psycopg2.extensions.connection = None,
                      cur: psycopg2.extensions.cursor = None,
                      sandbox=False):
        """
        flexible function to create asset based on any combination of required and optional parameters
        owner, serial_number, and age have default db values so they can be ignored if desired
        process_id, common_name, eol, rul, and units are not required
        """
        assert type_id is not None and type(type_id) == int, '[ERROR] must supply <type_id>(int)'
        statement = 'insert into asset_tb("type_id"'
        values = [type_id]
        if len(owner) > 2:
            statement = statement + ',"owner"'
            values.append(owner)
        if process_id is not None and type(process_id) == int:
            statement = statement + ',"process_id"'
            values.append(process_id)
        if len(serial_number) > 2:
            statement = statement + ',"serial_number"'
            values.append(serial_number)
        if len(common_name) > 2:
            statement = statement + ',"common_name"'
            values.append(common_name)
        if age is not None and type(age) == float:
            statement = statement + ',"age"'
            values.append(age)
        if eol is not None and type(eol) == float:
            statement = statement + ',"eol"'
            values.append(eol)
        if rul is not None and type(rul) == float:
            statement = statement + ',"rul"'
            values.append(rul)
        if units is not None:
            statement = statement + ',"units"'
            values.append(units)
        statement = statement + f""") values {tuple(values)};"""
        if sandbox:
            print(statement)
            return statement
        else:
            try:
                cur.execute(statement)
                db.commit()
            except psycopg2.errors.UniqueViolation:
                print("[ERROR] asset already exists (serial numbers must be unique).")
            return DB._get_asset(serial_number=serial_number, db=db)


    @staticmethod
    def _get_asset(serial_number: str = None,
                   id: int = None,
                   db: psycopg2.extensions.connection = None):
        """
        returns the asset as a dataframe
        """
        assert serial_number is not None or id is not None, '[ERROR] must supply <serial_number>(str) or <id>(int)'
        assert db is not None, '[ERROR] must pass <db>(psycopg2.extensions.connection)'
        if serial_number is not None:
            statement = f"""select * from asset_tb where "serial_number" = '{serial_number}';"""
        else:
            statement = f"""select * from asset_tb where "id" = {id};"""
        return DB.execute(statement, db)




    @staticmethod
    def _create_component(asset: pd.DataFrame = None,
                          group_id: int = None,
                          Fc: int = None,
                          unit: int = None,
                          dataset: str = None,
                          db: psycopg2.extensions.connection = None,
                          cur: psycopg2.extensions.cursor = None):
        """
        creates a component in the db, the asset must be created first
        """
        assert asset is not None and group_id is not None and Fc is not None and unit is not None and dataset is not None, '[ERROR] must supply all parameters'
        asset_type = DB._get_asset_type(type_id=asset.type_id.values[0], db=db)
        assert len(asset_type) > 0, f'[ERROR] a valid asset type was not found with id <{asset.type_id.values[0]}>'

        table_name = f"{asset_type.type.values[0]}_{asset_type.subtype.values[0]}_tb"
        assert DB.table_exists(table_name, db), f'[ERROR] table <{table_name}> does not exist'

        statement = f"""insert into {table_name}("id", "group_id", "Fc", "unit", "dataset") values ({asset.id.values[0]}, {group_id}, {Fc}, {unit}, '{dataset}');"""
        try:
            cur.execute(statement)
            db.commit()
        except Exception as e:
            print(e)





    #
    # @staticmethod
    # def _get_units(by: str = 'Fc',
    #                db: psycopg2.extensions.connection = None) -> pd.DataFrame:























