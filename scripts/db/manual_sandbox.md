# Create a postgresql sandbox database


## Download the backup file from the production database

```shell
source .env

# Good ref: https://postgres.ai/docs/postgres-howtos/database-administration/backup-recovery/how-to-speed-up-pg-dump#:~:text=Monitoring%20Dump%20Progress%E2%80%8B,%7C%20gzip%20.

# 91m32.398s
# time pg_dump -Fd \
#   --host=db.$SUPABASE_PROD_ID.supabase.co \
#   --port=5432 \
#   --dbname=postgres \
#   --username=postgres \
#   --file=scripts/db/$(date +%Y_%m_%d)_production.backup \
#   -Z3 \
#   --data-only \
#   --schema=public \
#   --large-objects \
#   --no-sync \
#   --verbose \
#   --jobs=4

# 56m57.809s with no compression
time pg_dump -Fd \
  --host=db.$SUPABASE_PROD_ID.supabase.co \
  --port=5432 \
  --username=postgres \
  --dbname=postgres \
  --file=scripts/db/$(date +%Y_%m_%d)_production.backup \
  -Z0 \
  -j 4 \
  --data-only \
  --schema=public \
  --large-objects \
  --no-sync \
  --verbose

# 1m6.688s
time tar -I 'zstd -3 -T0' -cvf scripts/db/$(date +%Y_%m_%d)_production.backup.tar.zst scripts/db/$(date +%Y_%m_%d)_production.backup/

mkdir -p $DATABASE_DIR/$(date +%Y_%m_%d) && cp scripts/db/$(date +%Y_%m_%d)_production.backup/ $DATABASE_DIR/$(date +%Y_%m_%d)_production.backup/

# or copy from the shared storage if it exists
# cp $DATABASE_DIR/$(date +%Y_%m_%d)_production.backup/ scripts/db/$(date +%Y_%m_%d)_production.backup/ 

```

## Create an emtpy sandbox database from scripts/db/my-postgres.conf file
```shell

# 0m9.057s
# postgres:17 needs to match the version in the Dockerfile and supabase.co
time apptainer build --bind /groups/betzig/betziglab:/groups/betzig/betziglab -F --sandbox scripts/db/sandbox/ docker://postgres:17 \
    && cp --force /workspace/cell_observatory_platform/scripts/db/my-postgres.conf scripts/db/sandbox/etc/postgresql/postgresql.conf 

# start the sandbox database
apptainer run --writable \
  --pwd /var/lib/postgresql \
  --env POSTGRES_PASSWORD=postgres \
  scripts/db/sandbox/ -c 'port=5433' -c 'config_file=/etc/postgresql/postgresql.conf'

```

## In a separate terminal, run the following commands to import the backup file into the sandbox database
```shell

# 0m0.313s to create the roles and extensions
source .env && time apptainer exec scripts/db/sandbox/ psql -h localhost -p 5433 -U postgres -d postgres --command="
  CREATE ROLE anon; 
  CREATE ROLE authenticated;
  CREATE ROLE authenticator;
  CREATE ROLE authenticated_role; 
  CREATE ROLE service_role;  
  DROP SCHEMA public CASCADE; CREATE SCHEMA public;
  CREATE EXTENSION IF NOT EXISTS intarray;
  CREATE EXTENSION IF NOT EXISTS cube;
  CREATE EXTENSION IF NOT EXISTS pgcrypto; CREATE EXTENSION IF NOT EXISTS hstore;
"

# 0m1.027s to build the schema
time apptainer exec --bind /groups/betzig/betziglab:/groups/betzig/betziglab scripts/db/sandbox/ \
  pg_restore -h localhost -p 5433 -U postgres -d postgres --no-owner --no-privileges -F d --section=pre-data \
  scripts/db/$(date +%Y_%m_%d)_production.backup/

# 0m0.353s to restore the prepared table first because everything else depends on it
time apptainer exec --bind /groups/betzig/betziglab:/groups/betzig/betziglab scripts/db/sandbox/ \
  pg_restore -h localhost -p 5433 -U postgres -d postgres -F d --table prepared \
  scripts/db/$(date +%Y_%m_%d)_production.backup/

# 3m16.973s to restore the rest of the data in parallel -j16 (~5GB for 2026_03_05_full_backup_production_postgres_db_custom.backup)
time apptainer exec --bind /groups/betzig/betziglab:/groups/betzig/betziglab scripts/db/sandbox/ \
  pg_restore -h localhost -p 5433 -U postgres -d postgres -F d -j16 \
  scripts/db/$(date +%Y_%m_%d)_production.backup/
  
```

## Verify the database is restored correctly

```shell

# 0m19.421s -> 34467912 rows
time apptainer exec scripts/db/sandbox/ psql -h localhost -p 5433 -U postgres --command="SELECT COUNT(*) FROM PREPARED_CUBES;"

# 9m11.176s to create the prepared tiles view table
time apptainer exec scripts/db/sandbox/ psql -h localhost -p 5433 -U postgres --command="SELECT ID, REFRESH_PREPARED_TILES_VIEW_TABLE (ID) FROM PREPARED;"

#  0m0.286s to test prepared tiles view table with 5426 rows
time apptainer exec scripts/db/sandbox/ psql -h localhost -p 5433 -U postgres --command="SELECT 'PREPARED_TILES_VIEW_TABLE' name, count(*) row_count from PREPARED_tiles_view_table;"

```

## Populate materialized views

```shell

# 42m10.787s
time apptainer exec scripts/db/sandbox/ psql -h localhost -p 5433 -U postgres --file=scripts/db/populate_mviews.sql

# 8m20.130s
time apptainer exec scripts/db/sandbox/ psql -h localhost -p 5433 -U postgres --command="ANALYZE;"

# 0m52.998s
time apptainer exec scripts/db/sandbox/ psql -h localhost -p 5433 -U postgres --file=scripts/db/SQLtest.sql

```

## Archive the sandbox database to use for training sessions


```shell

# 2m28.913s (~5GB for 2026_03_25_production.backup)
time tar -I 'zstd -3 -T0' -cvf scripts/db/$(date +%Y_%m_%d)_sandbox.tar.zst scripts/db/sandbox/

mkdir -p $DATABASE_DIR/$(date +%Y_%m_%d) && cp scripts/db/$(date +%Y_%m_%d)_sandbox.tar.zst $DATABASE_DIR/$(date +%Y_%m_%d)/sandbox.tar.zst

```
