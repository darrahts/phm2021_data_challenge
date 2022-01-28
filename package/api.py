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



    @staticmethod
    def _get_units(group_id: [] = None,
                   Fc: [] = None,
                   datasets: [] = None,
                   order_by: str = 'id',
                   db: psycopg2.extensions.connection = None):
        valid_datasets = [
                'DS01-005',
                'DS03-012',
                'DS04',
                'DS05',
                'DS06',
                'DS07',
                'DS08a-009',
                'DS08c-008'
            ]
        valid_order_by = [
            'id',
            'age',
            'rul'
        ]
        statement = """select ast."id", 
        ast."serial_number", 
        ast."age", 
        ast."eol",  
        ast."rul", 
        ent."group_id", 
        ent."Fc", 
        ent."unit", 
        ent."dataset" 
        from asset_tb ast 
        join engine_ncmapss_tb ent 
        on ast.id = ent.id """
        assert isinstance(group_id, list) or isinstance(group_id, type(None)), '[ERROR] pass arguments as lists'
        assert isinstance(Fc, list) or isinstance(Fc, type(None)), '[ERROR] pass arguments as lists'
        assert isinstance(datasets, list) or isinstance(datasets, type(None)), '[ERROR] pass dataset as a list'
        assert order_by in valid_order_by, f'[ERROR] <order_by> bust be in {valid_order_by}'

        if group_id is not None:
            if len(group_id) > 1:
                group_id = tuple(group_id)
            elif len(group_id) == 1:
                group_id = f'({group_id[0]})'

        if Fc is not None:
            if len(Fc) > 1:
                Fc = tuple(Fc)
            elif len(Fc) == 1:
                Fc = f'({Fc[0]})'

        if datasets is not None:
            if datasets[0] == 'all':
                datasets = tuple(valid_datasets)
            elif len(datasets) > 1:
                datasets = tuple(datasets)
            elif len(datasets) == 1:
                datasets = f"('{datasets[0]}')"

        clause = ''
        if group_id is not None:
            clause = f'where ent."group_id" in {group_id} '

        if Fc is not None:
            if group_id is not None:
                clause = clause + f'and ent."Fc" in {Fc} '
            else:
                clause = f'where ent."Fc" in {Fc} '

        if datasets is not None:
            if isinstance(datasets, tuple):
                for ds in datasets:
                    assert (any(ds in sub for sub in valid_datasets)), f'[ERROR] valid dataset was not supplied. valid datasets are {valid_datasets}'
            elif isinstance(datasets, str):
                assert eval(datasets) in valid_datasets, f'[ERROR] valid dataset was not supplied. valid datasets are {valid_datasets}'
            if group_id is not None or Fc is not None:
                clause = clause + f'and ent."dataset" in {datasets} '
            else:
                clause = f'where ent."dataset" in {datasets} '

        statement = statement + clause
        statement = statement + f'order by ast."{order_by}" asc;'

        return DB.execute(statement, db)




    @staticmethod
    def _get_unit_counts(group_by: str = 'both',
                         db: psycopg2.extensions.connection = None) -> pd.DataFrame:
        valid_group_by = ['group_id', 'dataset', 'both']
        assert group_by in valid_group_by, f'[ERROR], <group_by> must be in {valid_group_by}'
        if group_by == 'both':
            statement = """select count(*), group_id, dataset from engine_ncmapss_tb group by dataset, group_id order by dataset, group_id;"""
        elif group_by == 'group_id':
            statement = """select count(*), group_id from engine_ncmapss_tb group by group_id order by group_id;"""
        else:
            statement = """select count(*), dataset from engine_ncmapss_tb group by dataset order by dataset;"""
        return DB.execute(statement, db)




    @staticmethod
    def _get_data(units: [] = None,
                  downsample: int = 5,
                  hs: int = None,
                  limit: int = None,
                  tables: [] = None,
                  drop_cols: [] = None,
                  db: psycopg2.extensions.connection = None) -> pd.DataFrame:
        valid_units = DB.execute("select id from asset_tb;", db).values
        valid_tables = [
            'summary_tb',
            'telemetry_tb',
            'degradation_tb'
        ]
        assert units is None or all(unit in valid_units for unit in units), '[ERROR], either do not pass a value for <units> or ensure all values passed are valid'
        for t in tables:
            assert t in valid_tables, f'[ERROR], invalid table specified. Select a table in <{valid_tables}>'
        if units is None and downsample < 2:
            choice = input("It is highly recommended to pass a list of units and/or downsampling factor greater than 2 as selecting all units will take a few minutes (this query has not been optimized). proceed? (y/n): ")
            if len(tables) == 1:
                if choice == 'y' or choice == 'Y':
                    statement = f"""select tb.* from {tables[0]} tb;"""
            else:
                return pd.DataFrame()
        else:
            if len(tables) == 1:
                statement = f"""select tb.* from (select * from {tables[0]} order by id asc) tb where asset_id in {tuple(units)} and tb.id % {downsample} = 0;"""
            elif len(tables) == 2:
                statement = f"""select s."cycle", 
                                       s.hs, 
                                       s.alt, 
                                       s."Mach", 
                                       s."TRA", 
                                       s."T2",
                                       e."Fc",
                                       t.*
                                       from summary_tb s 
                                       inner join telemetry_tb t on s.id = t.id 
                                       inner join engine_ncmapss_tb e on s.asset_id = e.id"""
                if len(units) > 1:
                    statement = statement + f""" where s.asset_id in {tuple(units)}"""
                else:
                    statement = statement + f""" where s.asset_id = {units[0]}"""

                if hs is not None:
                    statement = statement + f""" and s.hs = {hs}"""

                statement = statement + f""" and s.id % {downsample} = 0 order by t.id asc;"""
        if drop_cols is None:
            return DB.execute(statement, db)
        else:
            return DB.execute(statement, db).drop(columns=drop_cols)


















