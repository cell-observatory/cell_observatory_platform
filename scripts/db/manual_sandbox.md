# Create a Postgres sandbox database from a backup file

This is the manual restore path for creating a `sandbox.tar.zst` that can be
copied to the cluster filesystem and loaded by the training launch scripts.

---

## LSF version (Janelia / groups filesystem)

### First copy the repo to scratch

Run this workflow from a repo copy under `/scratch`, not from `/groups`.

```shell
# Example only; adjust paths as needed.
cd /scratch/$USER
rsync -av /groups/betzig/home/hamiltonh/git_managed/cell_observatory_platform/ ./cell_observatory_platform/
cd /scratch/$USER/cell_observatory_platform
```

### Optional: stage the backup under `scripts/db`

```shell
source .env

# Example copy from shared storage to the repo for a one-off restore.
# cp "${DATABASE_DIR}/YYYY_MM_DD/production.backup" scripts/db/
# cp -r "${DATABASE_DIR}/YYYY_MM_DD/production.backup.dir" scripts/db/
```

### Build an empty sandbox image

```shell
source .env

# postgres:17 needs to match the production major version.
time apptainer build --bind /groups/betzig/betziglab:/groups/betzig/betziglab \
  -F --sandbox scripts/db/sandbox/ docker://postgres:17

# Apptainer on this cluster auto-binds /groups for writable runs.
mkdir -p scripts/db/sandbox/groups

cp --force scripts/db/my-postgres.conf scripts/db/sandbox/etc/postgresql/postgresql.conf
```

### Start the sandbox Postgres server

Run this in a dedicated terminal and leave it running during restore.

```shell
apptainer run --no-mount bind-paths --writable \
  --pwd /var/lib/postgresql \
  --env POSTGRES_PASSWORD=postgres \
  scripts/db/sandbox/ \
  -c 'port=5433' \
  -c 'config_file=/etc/postgresql/postgresql.conf'
```

### Prepare roles and extensions

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

### Restore the backup

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

### Verify the restore

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

### Archive the sandbox for training sessions

```shell
source .env

time tar -I 'zstd -3 -T0' -cvf scripts/db/$(date +%Y_%m_%d)_sandbox.tar.zst \
  -C scripts/db sandbox

mkdir -p "${DATABASE_DIR}/$(date +%Y_%m_%d)"
cp scripts/db/$(date +%Y_%m_%d)_sandbox.tar.zst \
  "${DATABASE_DIR}/$(date +%Y_%m_%d)/sandbox.tar.zst"
```

### Clean up the scratch working copy

Only do this after:
- the sandbox Postgres process has been stopped
- the tarball exists where you copied it under `${DATABASE_DIR}`

```shell
ls -lh "${DATABASE_DIR}/$(date +%Y_%m_%d)/sandbox.tar.zst"

cd /scratch/$USER
rm -rf /scratch/$USER/cell_observatory_platform
```

---

## ABC version (clusterfs)

All steps must run on the **same compute node**. Get an interactive
allocation first, or use a shared path for `WORK` if you need to run
steps from different sessions (e.g. `WORK=/clusterfs/nvme/hph/git_managed/databases/sandbox_build`).

### 1. Create a backup from the remote database

Skip this if you already have a `.dump` file.

```shell
export PGPASSWORD=postgres
STAMP="$(date +%Y_%m_%d_%H%M)"
DUMP_FILE="/clusterfs/nvme/hph/git_managed/databases/acquisition_db_${STAMP}.dump"

pg_dump \
  --host 127.0.0.1 \
  --port 54322 \
  --username postgres \
  --dbname postgres \
  --format custom \
  --file "$DUMP_FILE"

echo "Wrote: $DUMP_FILE"
ls -lh "$DUMP_FILE"
```

### 2. Set variables

```shell
export REPO=/clusterfs/nvme/hph/git_managed/cell_observatory_platform
export DB_DIR=/clusterfs/nvme/hph/git_managed/databases
export DUMP_FILE=/clusterfs/nvme/hph/git_managed/databases/acquisition_db_2026_04_16_1827.dump
export WORK=/tmp/$USER/sandbox_build

mkdir -p "$WORK"
cd "$REPO"
source .env
```

### 3. Build an empty sandbox image

```shell
time apptainer build \
  -F --sandbox "$WORK/sandbox/" docker://postgres:17

mkdir -p "$WORK/sandbox/global" "$WORK/sandbox/clusterfs"

cp --force scripts/db/my-postgres.conf "$WORK/sandbox/etc/postgresql/postgresql.conf"
```

### 4. Start the sandbox Postgres server

Run this in a **dedicated terminal** on the same node and leave it running.

```shell
export WORK=/tmp/$USER/sandbox_build

apptainer run --no-mount bind-paths --writable \
  --pwd /var/lib/postgresql \
  --env POSTGRES_PASSWORD=postgres \
  "$WORK/sandbox/" \
  -c 'port=5433' \
  -c 'config_file=/etc/postgresql/postgresql.conf'
```

### 5. Prepare roles and extensions

Back in the first terminal. Since Postgres is listening on `localhost:5433`,
run `psql` directly on the host — no need to `apptainer exec`.

```shell
export PGPASSWORD=postgres

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

### 6. Restore the `.dump` backup

```shell
export PGPASSWORD=postgres

time pg_restore -h localhost -p 5433 -U postgres -d postgres \
  --no-owner --no-privileges \
  /clusterfs/nvme/hph/git_managed/databases/acquisition_db_2026_04_16_1827.dump
```

### 7. Verify the restore

```shell
export PGPASSWORD=postgres

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

### 8. Stop the sandbox server

Go to the terminal running the server (step 4) and press **Ctrl+C**.

### 9. Archive the sandbox

```shell
export WORK=/tmp/$USER/sandbox_build
export DB_DIR=/clusterfs/nvme/hph/git_managed/databases

STAMP="$(date +%Y_%m_%d)"

time tar -I 'zstd -3 -T0' -cvf "$WORK/${STAMP}_sandbox.tar.zst" \
  -C "$WORK" sandbox

cp "$WORK/${STAMP}_sandbox.tar.zst" "$DB_DIR/${STAMP}_sandbox.tar.zst"

ls -lh "$DB_DIR/${STAMP}_sandbox.tar.zst"
```

Then update `DATABASE_SANDBOX` in `.env` to point to the new tarball:

```
DATABASE_SANDBOX=${DATABASE_DIR}/2026_04_16_sandbox.tar.zst
```

### 10. Clean up

```shell
rm -rf /tmp/$USER/sandbox_build
```
