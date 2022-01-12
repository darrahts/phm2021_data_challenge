



select * from group_tb;

select * from engine_ncmapss_tb where group_id = 2 and "Fc" = 2 and dataset = 'DS08a-009';

select * from summary_tb;

select * from engine_ncmapss_tb;

select * from asset_tb;

select * 

select ent.* from (select *, row_number() over() rn from engine_ncmapss_tb) ent where ent.rn % 2 = 0;


select * from telemetry_tb;

select * from degradation_tb;





select * from information_schema.tables where table_schema = 'public';






