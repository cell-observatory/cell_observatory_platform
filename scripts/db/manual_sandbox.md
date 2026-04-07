# Create a Postgres sandbox database from a backup file

This is the manual restore path for creating a `sandbox.tar.zst` that can be
copied to the cluster filesystem and loaded by the training launch scripts.

## First copy the repo to scratch

Run this workflow from a repo copy under `/scratch`, not from `/groups`.

```shell
# Example only; adjust paths as needed.
cd /scratch/$USER
rsync -av /groups/betzig/home/hamiltonh/git_managed/cell_observatory_platform/ ./cell_observatory_platform/
cd /scratch/$USER/cell_observatory_platform
```

## Optional: stage the backup under `scripts/db`

```shell
source .env

# Example copy from shared storage to the repo for a one-off restore.
# cp "${DATABASE_DIR}/YYYY_MM_DD/production.backup" scripts/db/
# cp -r "${DATABASE_DIR}/YYYY_MM_DD/production.backup.dir" scripts/db/
```

## Build an empty sandbox image

```shell
source .env

# postgres:17 needs to match the production major version.
time apptainer build --bind /groups/betzig/betziglab:/groups/betzig/betziglab \
  -F --sandbox scripts/db/sandbox/ docker://postgres:17

# Apptainer on this cluster auto-binds /groups for writable runs.
mkdir -p scripts/db/sandbox/groups

cp --force scripts/db/my-postgres.conf scripts/db/sandbox/etc/postgresql/postgresql.conf
```

## Start the sandbox Postgres server

Run this in a dedicated terminal and leave it running during restore.

```shell
apptainer run --no-mount bind-paths --writable \
  --pwd /var/lib/postgresql \
  --env POSTGRES_PASSWORD=postgres \
  scripts/db/sandbox/ \
  -c 'port=5433' \
  -c 'config_file=/etc/postgresql/postgresql.conf'
```

## Prepare roles and extensions

```shell
source .env

time apptainer exec scripts/db/sandbox/ \
  psql -h localhost -p 5433 -U postgres -d postgres --command="
    CREATE ROLE anon;
    CREATE ROLE authenticated;
    CREATE ROLE authenticator;
    CREATE ROLE authenticated_role;
    CREATE ROLE service_role;
    DROP SCHEMA public CASCADE;
    CREATE SCHEMA public;
    CREATE EXTENSION IF NOT EXISTS intarray;
    CREATE EXTENSION IF NOT EXISTS cube;
    CREATE EXTENSION IF NOT EXISTS pgcrypto;
    CREATE EXTENSION IF NOT EXISTS hstore;
  "
```

## Restore the backup

Use exactly one of the following commands depending on the backup format.

```shell
# Option A: pg_dump -Fc custom-format backup file
# time apptainer exec --bind /groups/betzig/betziglab:/groups/betzig/betziglab scripts/db/sandbox/ \
#   pg_restore -h localhost -p 5433 -U postgres -d postgres \
#   --no-owner --no-privileges \
#   scripts/db/production.backup

# Option B: pg_dump -Fd directory-format backup
time apptainer exec --bind /groups/betzig/betziglab:/groups/betzig/betziglab scripts/db/sandbox/ \
  pg_restore -h localhost -p 5433 -U postgres -d postgres -F d -j 16 \
  --no-owner --no-privileges \
  scripts/db/production.backup.dir/
```

## Verify the newer training-table schema

List the restored training relations:

```shell
time apptainer exec scripts/db/sandbox/ \
  psql -P pager=off -h localhost -p 5433 -U postgres -d postgres --command="
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND (
        table_name LIKE 'prepared_cube\_%_agg\_%' ESCAPE '\'
        OR table_name LIKE 'prepared_tile\_%_agg\_%' ESCAPE '\'
      )
    ORDER BY table_name;
  "
```

Check that the expected metadata columns exist on the restored tables:

```shell
time apptainer exec scripts/db/sandbox/ \
  psql -P pager=off -h localhost -p 5433 -U postgres -d postgres --command="
    SELECT table_name, column_name
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND (
        table_name LIKE 'prepared_cube\_%_agg\_%' ESCAPE '\'
        OR table_name LIKE 'prepared_tile\_%_agg\_%' ESCAPE '\'
      )
      AND column_name IN (
        'prepared_id',
        'tile_name',
        'time_start',
        'z_start',
        'y_start',
        'x_start',
        'is_test_split',
        'channel_mapping',
        'channels_metadata',
        'annotations_metadata',
        'annotation_count',
        'has_annotations'
      )
    ORDER BY table_name, ordinal_position;
  "
```

Optional: inspect one representative restored table directly:

```shell
# Replace the table name below with one of the restored relations from the query above.
# time apptainer exec scripts/db/sandbox/ \
#   psql -P pager=off -h localhost -p 5433 -U postgres -d postgres --command="
#     SELECT *
#     FROM public.prepared_cube_channel_agg_16_128_128_128
#     LIMIT 5;
#   "
```

Collect fresh planner stats after restore:

```shell
time apptainer exec scripts/db/sandbox/ \
  psql -P pager=off -h localhost -p 5433 -U postgres -d postgres --command="ANALYZE;"
```

## Archive the sandbox for training sessions

```shell
source .env

time tar -I 'zstd -3 -T0' -cvf scripts/db/$(date +%Y_%m_%d)_sandbox.tar.zst \
  -C scripts/db sandbox

mkdir -p "${DATABASE_DIR}/$(date +%Y_%m_%d)"
cp scripts/db/$(date +%Y_%m_%d)_sandbox.tar.zst \
  "${DATABASE_DIR}/$(date +%Y_%m_%d)/sandbox.tar.zst"
```

## Clean up the scratch working copy

Only do this after:
- the sandbox Postgres process has been stopped
- the tarball exists where you copied it under `${DATABASE_DIR}`

```shell
# Optional sanity check before cleanup.
ls -lh "${DATABASE_DIR}/$(date +%Y_%m_%d)/sandbox.tar.zst"

# Remove the scratch working copy when you are done.
cd /scratch/$USER
rm -rf /scratch/$USER/cell_observatory_platform
```