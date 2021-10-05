------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
/*
            This script will create a database named uav_db and a user with the currently logged in user,
            grant permissions to the database and make the user a superuser.

            Tim Darrah
            NASA Fellow
            PhD Student
            Vanderbilt University
            timothy.s.darrah@vanderbilt.edu
*/
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------

-- create a database user (you)
create user :user with encrypted password :passwd;

-- give yourself admin access
alter user :user with superuser;

-- create the database
create database ncmapss_db with owner :user;
