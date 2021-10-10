
-- NOTE tables are not timescaledb hyptertables yet.....

-- NOTE processes are not implemented with the N-CMAPSS set yet as these tables hold the process models, 
-- and we do not have the process models for the different types of degradation
-- im sure someone does......

/*
	fields:
		type: refers to the process such as degradation, environment, etc
		subtype:1: refers to the component such as battery, motor, wind, etc
		subtype2: refers to what within the component such as capacitance, resistance, gust, etc
*/
create table process_type_tb(
	"id" serial primary key not null,
	"type" varchar(32) not null,
	"subtype1" varchar(64) not null,
	"subtype2" varchar(64),
	unique("type", "subtype1", "subtype2")
);


/*
	fields:
		description: details about how the process evolves such as continuous or discrete, etc
*/
create table process_tb(
	"id" serial primary key not null,
	"type_id" int not null references process_type_tb,
	"description" varchar(256) not null,
    "source" varchar(256) not null,
	"parameters" json not null,
	unique("type_id", "description", "source")
);


/*
        NOTE - all derived component tables (dc motor, electrochemical battery, ncmapss engine, etc) MUST
        be defined as <type>_<subtype>_tb. This is one of many forms of data integrity enforcement
*/
create table asset_type_tb(
    "id" serial primary key not null,
    "type" varchar(32) not null,
    "subtype" varchar(32) not null,
    "description" varchar(256),
    unique ("type", "subtype")
);


/*
    Table to hold asset data

	There is not a table-wide unique constraint on this table because we can have more than one component of the same type,
	only the serial number has to be unique.

    ALL CUSTOM COMPONENT TABLES MUST HAVE THE FOLLOWING COLUMN DEFINITION REFERENCING THIS TABLE
        "id" int primary key not null references asset_tb(id)
*/
create table asset_tb(
    "id" serial primary key not null,
    "type_id" int not null references asset_type_tb(id),
    "owner" varchar(32) not null default(current_user),
	"process_id" int references process_tb(id),
    "serial_number" varchar(32) unique not null default upper(substr(md5(random()::text), 0, 9)),
	"common_name" varchar(32),
    "age" float not null default 0,
    "eol" float,
    "rul" float,
    "units" varchar(32),
    unique("type_id", "owner", "process_id", "common_name", "age", "eol", "rul", "units")
);


/*
	this table is used for flight classes
*/
create table group_tb(
	id serial primary key not null,
	info varchar(256) unique not null
);

insert into group_tb(info) values ('dev');
insert into group_tb(info) values ('test');
insert into group_tb(info) values ('val');

-- insert into group_tb(info) values('flight length 1 - 3 hours');
-- insert into group_tb(info) values('flight length 3 - 5 hours');
-- insert into group_tb(info) values('flight length > 5 hours');


/*
    This is a custom component table for the N-CMAPSS dataset. 
    
*/
create table engine_ncmapss_tb(
    "id" int primary key not null references asset_tb(id),
    "group_id" int not null references group_tb(id),
    "Fc" int not null,
    "unit" int not null,
    "dataset" varchar(32) not null,
    constraint flight_class check ("Fc" = 1 or "Fc" = 2 or "Fc" = 3),
    unique(id, group_id, unit, dataset)
);


/*
    scenario descriptors and cycle, links to asset_id which contains aux data
*/
create table summary_tb(
    "id" serial primary key not null,
    "asset_id" int not null references asset_tb(id),
    "cycle" int not null,
    "hs" float not null,
    "alt" float not null,
    "Mach" float not null,
    "TRA" float not null,
    "T2" float not null
);
-- select create_hypertable('summary_tb', 'asset_id', 'id', chunk_time_interval => 1000000);


/*
    measurements, dt is not unique because multiple units could (and will) have the same dt
*/
create table telemetry_tb(
    "id" int references summary_tb(id),
    "dt" timestamptz(6) not null,
    "Wf" float not null,
    "Nf" float not null,
    "Nc" float not null,
    "T24" float not null,
    "T30" float not null,
    "T48" float not null,
    "T50" float not null,
    "P15" float not null,
    "P2" float not null,
    "P21" float not null,
    "P24" float not null,
    "Ps30" float not null,
    "P40" float not null,
    "P50" float not null,
    unique("id", "Wf", "Nf", "Nc", "T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24", "Ps30", "P40", "P50"),
    unique("id", "dt")
);
-- create partitions on id, then dt
-- select create_hypertable('telemetry_tb', 'id', 'dt');

create table virtual_tb(
    "id" int references summary_tb(id),
    "dt" timestamptz(6) not null,
    "T40" float not null,
    "P30" float not null,
    "P45" float not null,
    "W21" float not null,
    "W22" float not null,
    "W25" float not null,
    "W31" float not null,
    "W32" float not null,
    "W48" float not null,
    "W50" float not null,
    "SmFan" float not null,
    "SmLPC" float not null,
    "SmHPC" float not null,
    "phi" float not null,
    unique("id", "T40", "P30", "P45", "W21", "W22", "W25", "W31", "W32", "W48", "W50", "SmFan", "SmLPC", "SmHPC", "phi"),
    unique("id", "dt")
);


create table degradation_tb(
    "id" int references summary_tb(id),
    "fan_eff_mod" float not null,
    "fan_flow_mod" float not null,
    "LPC_eff_mod" float not null,
    "LPC_flow_mod" float not null,
    "HPC_eff_mod" float not null,
    "HPC_flow_mod" float not null,
    "HPT_eff_mod" float not null,
    "HPT_flow_mod" float not null,
    "LPT_eff_mod" float not null,
    "LPT_flow_mod" float not null
);
-- create the partitions based on id, then health state
-- select create_hypertable('degradation_tb', 'id', 'hs');






