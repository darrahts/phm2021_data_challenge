



select * from group_tb;

select * from engine_ncmapss_tb where group_id = 2 and "Fc" = 2 and dataset = 'DS08a-009';

select * from summary_tb where hs = 0;

select * from engine_ncmapss_tb;

select * from asset_tb;

select * 

select ent.* from engine_ncmapss_tb ent;

select tb.* from (select * from telemetry_tb ortder by id asc) tb where tb.asset_id in (1,2) and tb.

select s."cycle", 
	   s.hs, 
	   s.alt, 
	   s."Mach", 
	   s."TRA", 
	   s."T2",
	   e."Fc",
	   t.*
	   from summary_tb s 
	   inner join telemetry_tb t on s.id = t.id 
	   inner join engine_ncmapss_tb e on s.asset_id = e.id where s.id % 100 = 0 and s.hs = 1;

select s.* from summary_tb s where s.id % 10 = 0 order by id asc;
select t.* from telemetry_tb t where t.id % 10 = 0 order by id asc limit 100;

--t.* from summary_tb s join telemetry_tb t on s.asset_id = t.asset_id where s.hs = 0; 

select st.* from summary_tb st order by st.id asc;
select tt.* from telemetry_tb tt order by tt.id asc;

select * from telemetry_tb;

select * from degradation_tb;



select * from information_schema.tables where table_schema = 'public';






