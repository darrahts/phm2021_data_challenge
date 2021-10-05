#
#       This script installs timescaledb, 
#
#


sudo sh -c "echo 'deb [signed-by=/usr/share/keyrings/timescale.keyring] https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main' > /etc/apt/sources.list.d/timescaledb.list"
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/timescale.keyring
sudo apt-get update

# Now install appropriate package for PG version
sudo apt install timescaledb-2-postgresql-13

# handle the case when the auto tune complains about not finding postgres
PG_FILE=$(locate postgresql.conf | head -1)
PG_PATH=$(echo $PG_FILE | sed 's/postgresql.conf//')

# now call autotune.
sudo timescaledb-tune --quiet --yes --conf-path=$PG_PATH

# add to environment if desired, although your path might be different
#sudo echo PG_PATH="\""/etc/postgresql/13/main/"\"" >> /etc/environment

sudo service postgresql restart