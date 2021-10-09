#!/bin/bash

################################################################################
################################################################################
#
#            This script will promt the user to 
#                1. install PostgreSQL
#                    copy and pasted code from https://www.postgresql.org/download/linux/ubuntu/
#                2. install timescale db
#                3. setup the database and user
#                    creates ncmapss_db and the current user
#                4. setup the table schema and enable timescaledb extension
#                    executes table_schema.sql
#                5. execute package/populate_db.py to insert the h5 data into the db
#            Tim Darrah
#            NASA Fellow
#            PhD Student
#            Vanderbilt University
#            timothy.s.darrah@vanderbilt.edu
#
################################################################################
################################################################################

# install PostgreSQL
read -p "install postgreSQL? (y/n): " ans
if [[ $ans = y ]]
then
    # Create the file repository configuration:
    sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

    # # Import the repository signing key:
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -

    # # Update the package lists:
    sudo apt-get update

    # # Install the latest version of PostgreSQL.
    # # If you want a specific version, use 'postgresql-12' or similar instead of 'postgresql':
    sudo apt-get -y install postgresql
fi
unset ans

# install timescale
read -p "install timescaledb? (y/n): " ans
if [[ $ans = y ]]
then
    ./install_timescale.sh
fi
unset ans

# create the database and user
read -p "setup database? (y/n): " ans
if [[ $ans = y ]]
then
    read -p "enter your password: " passwd
    sudo -u postgres psql -f sql/setup_db_user.sql -v user="$USER" -v passwd="'$passwd'"
fi
unset ans


# create the tables
read -p "setup table schema? (y/n): " ans
if [[ $ans = y ]]
then
    read -p "enable timescaledb extension? (y/n): " res
    if [[ $res = y ]]
    then
        psql -d ncmapss_db -c "create extension if not exists timescaledb cascade;" -U $USER
    fi
    unset res
    psql -d ncmapss_db -f sql/setup_table_schema.sql -U $USER 
fi
unset ans


# setup a readonly user? Note, access to future tables will have to be granted manually.
read -p "create a guest user account? (y/n): " ans
if [[ $ans = y ]]
then
    psql -d ncmapss_db -f sql/setup_readonly_guest.sql
fi
unset ans

echo 'restarting postgresql service...'
sudo service postgresql restart
echo 'service restarted...'
echo 'configuration complete.'
echo ' '

read -p "populate db with h5 data (ensure the data files are unzipped in data_h5/ directory) ? (y/n): " ans
if [[ $ans = y ]]
then
    echo 'this will take some time.........'
    python package/populate_db.py
    echo '****************************************'
    echo '****************************************'
    echo '****************************************'
    echo 'data upload complete.'
fi

