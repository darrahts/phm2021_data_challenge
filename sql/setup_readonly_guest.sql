------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
/*
            This script creates a readonly guest user, new tables will need permissions 
            manually assigned. 

            Tim Darrah
            NASA Fellow
            PhD Student
            Vanderbilt University
            timothy.s.darrah@vanderbilt.edu
*/
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------


-- create a read-only guest account 
create user guest with login encrypted password 'P@$$word1';
alter user guest with connection limit 10;
grant connect on database ncmapss_db to guest;
grant usage on schema public to guest;
grant usage, select on all sequences in schema public to guest;
grant select on all tables in schema public to guest;
