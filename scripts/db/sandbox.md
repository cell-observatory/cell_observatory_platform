# Create a postgresql sandbox database

### Longterm read replica

Use this option to create a longterm read replica of the production database on a local machine. 
The initial `clone` is a one-time operation but can take a long time depending on the size of the production database.
You can then use `sync` to update the sandbox with the latest production data.
You'll need to run `archive` to create a `sandbox.tar.zst` file that can be used for training sessions.
If something goes wrong, you can run `cleanup` to reset the sandbox database and start over.

#### Clone

```shell
./sandbox_cli build
```

```shell
build: apptainer build -F --sandbox /workspace/cell_observatory_platform/scripts/db/sandbox docker://postgres:17
INFO:    Starting build...
INFO:    Fetching OCI image...
INFO:    Extracting OCI image...
INFO:    Inserting Apptainer configuration...
INFO:    Creating sandbox directory...
INFO:    Build complete: /workspace/cell_observatory_platform/scripts/db/sandbox
 build: installed my-postgres.conf -> /workspace/cell_observatory_platform/scripts/db/sandbox/etc/postgresql/postgresql.conf

 ```shell
./sandbox_cli clone

Starting sandbox Postgres via Apptainer for this command...
Sandbox Postgres is ready on 127.0.0.1:5433.
Removing stale work dir: /workspace/cell_observatory_platform/scripts/db/pgcopydb_work
DO
CREATE EXTENSION
Work dir: /workspace/cell_observatory_platform/scripts/db/pgcopydb_work
Filter:   /workspace/cell_observatory_platform/scripts/db/filter.ini
2026-03-31 18:56:05.120 114 INFO   main.c:136                Running pgcopydb version 0.17-1.pgdg24.04+1 from "/usr/bin/pgcopydb"
2026-03-31 18:56:05.124 114 INFO   cli_common.c:1225         [SOURCE] Copying database from "postgres://postgres@db.cdgqohnoqldocuiwptmt.supabase.co:5432/postgres?keepalives=1&keepalives_idle=10&keepalives_interval=10&keepalives_count=60&sslmode=disable"
2026-03-31 18:56:05.125 114 INFO   cli_common.c:1226         [TARGET] Copying database into "postgres://postgres@127.0.0.1:5433/postgres?keepalives=1&keepalives_idle=10&keepalives_interval=10&keepalives_count=60"
2026-03-31 18:56:05.177 114 INFO   copydb.c:105              Using work dir "/workspace/cell_observatory_platform/scripts/db/pgcopydb_work"
2026-03-31 18:56:05.486 114 INFO   snapshot.c:107            Exported snapshot "0000005B-00001296-1" from the source database
2026-03-31 18:56:05.499 116 INFO   cli_clone_follow.c:543    STEP 1: fetch source database tables, indexes, and sequences
2026-03-31 18:56:05.966 116 INFO   copydb_schema.c:761       Fetched information for 378 tables (including 0 tables split in 0 partitions total), with an estimated total of 52 million tuples and 64 GB on-disk
2026-03-31 18:56:06.074 116 INFO   copydb_schema.c:968       Fetched information for 749 indexes (supporting 176 constraints)
2026-03-31 18:56:06.083 116 INFO   sequences.c:78            Fetching information for 3 sequences
2026-03-31 18:56:06.152 116 INFO   copydb_schema.c:1122      Fetched information for 12 extensions
2026-03-31 18:56:07.760 116 INFO   copydb_schema.c:1538      Found 0 indexes (supporting 0 constraints) in the target database
2026-03-31 18:56:07.767 116 INFO   cli_clone_follow.c:584    STEP 2: dump the source database schema (pre/post data)
2026-03-31 18:56:07.768 116 INFO   pgcmd.c:475                /usr/bin/pg_dump -Fc --snapshot 0000005B-00001296-1 --section=pre-data --section=post-data --schema public --file /workspace/cell_observatory_platform/scripts/db/pgcopydb_work/schema/schema.dump 'postgres://postgres@db.cdgqohnoqldocuiwptmt.supabase.co:5432/postgres?keepalives=1&keepalives_idle=10&keepalives_interval=10&keepalives_count=60&sslmode=disable'
2026-03-31 18:56:09.240 116 INFO   cli_clone_follow.c:592    STEP 3: restore the pre-data section to the target database
2026-03-31 18:56:09.259 116 INFO   dump_restore.c:398        ALTER DATABASE "postgres" SET "statement_timeout" TO '4h';
2026-03-31 18:56:09.468 116 INFO   dump_restore.c:443        Drop tables on the target database, per --drop-if-exists
2026-03-31 18:56:09.487 116 INFO   pgcmd.c:1008               /usr/bin/pg_restore --dbname 'postgres://postgres@127.0.0.1:5433/postgres?keepalives=1&keepalives_idle=10&keepalives_interval=10&keepalives_count=60' --section pre-data --jobs 4 --clean --if-exists --no-owner --no-acl --use-list /workspace/cell_observatory_platform/scripts/db/pgcopydb_work/schema/pre-filtered.list /workspace/cell_observatory_platform/scripts/db/pgcopydb_work/schema/schema.dump
2026-03-31 18:56:10.336 133 INFO   table-data.c:656          STEP 4: starting 8 table-data COPY processes
2026-03-31 18:56:10.354 136 INFO   vacuum.c:143              STEP 8: starting 8 VACUUM processes
2026-03-31 18:56:10.361 134 INFO   indexes.c:182             STEP 6: starting 4 CREATE INDEX processes
2026-03-31 18:56:10.361 134 INFO   indexes.c:183             STEP 7: constraints are built by the CREATE INDEX processes
2026-03-31 18:56:10.407 116 INFO   blobs.c:74                Skipping large objects: none found.
2026-03-31 18:56:10.410 116 INFO   sequences.c:194           STEP 9: reset sequences values
2026-03-31 18:56:10.412 166 INFO   sequences.c:290           Set sequences values on the target database
2026-03-31 19:07:08.549 116 INFO   cli_clone_follow.c:608    STEP 10: restore the post-data section to the target database
2026-03-31 19:07:08.819 116 INFO   pgcmd.c:1008               /usr/bin/pg_restore --dbname 'postgres://postgres@127.0.0.1:5433/postgres?keepalives=1&keepalives_idle=10&keepalives_interval=10&keepalives_count=60' --section post-data --jobs 4 --clean --if-exists --no-owner --no-acl --use-list /workspace/cell_observatory_platform/scripts/db/pgcopydb_work/schema/post-filtered.list /workspace/cell_observatory_platform/scripts/db/pgcopydb_work/schema/schema.dump
2026-03-31 19:18:09.473 116 INFO   cli_clone_follow.c:639    All step are now done, 22m01s elapsed
2026-03-31 19:18:09.477 116 INFO   summary.c:3173            Printing summary for 378 tables and 749 indexes

   OID | Schema |                                       Name | Parts | copy duration | transmitted bytes | indexes | create index duration 
-------+--------+--------------------------------------------+-------+---------------+-------------------+---------+----------------------
532625 | public |  mview_table_1_128_128_128_2_p00090_p00100 |     1 |         5m17s |           2396 MB |       2 |                 5m20s
532618 | public |  mview_table_1_128_128_128_2_p00080_p00090 |     1 |         3m35s |           1654 MB |       2 |                 4m07s
246190 | public |               prepared_cubes_p00090_p00095 |     1 |         3m06s |           1546 MB |       2 |                 3s570
246177 | public |               prepared_cubes_p00085_p00090 |     1 |         2m26s |           1218 MB |       2 |                 2s631
532786 | public |  mview_table_1_128_128_128_2_p00320_p00330 |     1 |         2m01s |            964 MB |       2 |                 1m08s
532646 | public |  mview_table_1_128_128_128_2_p00120_p00130 |     1 |         1m50s |            850 MB |       2 |                   44s
331013 | public |               prepared_cubes_p00363_p00374 |     1 |         1m59s |           1012 MB |       2 |                 2s334
246203 | public |               prepared_cubes_p00095_p00100 |     1 |         1m55s |            944 MB |       2 |                 2s247
532793 | public |  mview_table_1_128_128_128_2_p00330_p00340 |     1 |         1m47s |            849 MB |       2 |                   42s
532800 | public |  mview_table_1_128_128_128_2_p00340_p00350 |     1 |         1m46s |            842 MB |       2 |                   41s
532807 | public |  mview_table_1_128_128_128_2_p00350_p00360 |     1 |         1m45s |            835 MB |       2 |                   41s
532814 | public |  mview_table_1_128_128_128_2_p00360_p00370 |     1 |         1m47s |            839 MB |       2 |                   40s
330987 | public |               prepared_cubes_p00344_p00355 |     1 |         1m48s |            960 MB |       2 |                 1s937
532604 | public |  mview_table_1_128_128_128_2_p00060_p00070 |     1 |         1m37s |            743 MB |       2 |                   33s
532821 | public |  mview_table_1_128_128_128_2_p00370_p00380 |     1 |         1m34s |            731 MB |       2 |                 1m06s
532597 | public |  mview_table_1_128_128_128_2_p00050_p00060 |     1 |         1m33s |            718 MB |       2 |                   40s
532632 | public |  mview_table_1_128_128_128_2_p00100_p00110 |     1 |         1m32s |            696 MB |       2 |                   50s
532590 | public |  mview_table_1_128_128_128_2_p00040_p00050 |     1 |         1m32s |            703 MB |       2 |                   35s
532667 | public |  mview_table_1_128_128_128_2_p00150_p00160 |     1 |         1m23s |            626 MB |       2 |                 3m00s
532653 | public |  mview_table_1_128_128_128_2_p00130_p00140 |     1 |         1m18s |            316 MB |       2 |                 7s210
331000 | public |               prepared_cubes_p00355_p00363 |     1 |         1m19s |            695 MB |       2 |                 1s705
246346 | public |               prepared_cubes_p00150_p00155 |     1 |         1m18s |            651 MB |       2 |                 1s287
532611 | public |  mview_table_1_128_128_128_2_p00070_p00080 |     1 |         1m11s |            540 MB |       2 |                   24s
532583 | public |  mview_table_1_128_128_128_2_p00030_p00040 |     1 |         1m10s |            527 MB |       2 |                   18s
246788 | public |               prepared_cubes_p00320_p00325 |     1 |         1m07s |            561 MB |       2 |                 1s590
246268 | public |               prepared_cubes_p00120_p00125 |     1 |         1m05s |            532 MB |       2 |                 1s019
246086 | public |               prepared_cubes_p00050_p00055 |     1 |         1m04s |            530 MB |       2 |                 1s332
246216 | public |               prepared_cubes_p00100_p00105 |     1 |         1m01s |            514 MB |       2 |                 1s290
532660 | public |  mview_table_1_128_128_128_2_p00140_p00150 |     1 |           55s |            433 MB |       2 |                   14s
246164 | public |               prepared_cubes_p00080_p00085 |     1 |           56s |            499 MB |       2 |                 3s825
246073 | public |               prepared_cubes_p00045_p00050 |     1 |           54s |            481 MB |       2 |                 3s253
532576 | public |  mview_table_1_128_128_128_2_p00020_p00030 |     1 |           48s |            405 MB |       2 |                   13s
246801 | public |               prepared_cubes_p00325_p00330 |     1 |           49s |            440 MB |       2 |                 922ms
246814 | public |               prepared_cubes_p00330_p00335 |     1 |           48s |            440 MB |       2 |                 1s110
246827 | public |               prepared_cubes_p00335_p00340 |     1 |           48s |            440 MB |       2 |                 1s201
246112 | public |               prepared_cubes_p00060_p00065 |     1 |           45s |            388 MB |       2 |                 3s500
246125 | public |               prepared_cubes_p00065_p00070 |     1 |           43s |            382 MB |       2 |                 1s143
532639 | public |  mview_table_1_128_128_128_2_p00110_p00120 |     1 |           37s |            306 MB |       2 |                   11s
331026 | public |               prepared_cubes_p00374_p00374 |     1 |           38s |            358 MB |       2 |                 687ms
246281 | public |               prepared_cubes_p00125_p00130 |     1 |           40s |            352 MB |       2 |                 721ms
246840 | public |               prepared_cubes_p00340_p00344 |     1 |           37s |            347 MB |       2 |                 799ms
246255 | public |               prepared_cubes_p00115_p00120 |     1 |           35s |            311 MB |       2 |                 795ms
246047 | public |               prepared_cubes_p00035_p00040 |     1 |           35s |            309 MB |       2 |                 697ms
246333 | public |               prepared_cubes_p00145_p00150 |     1 |           33s |            293 MB |       2 |                 611ms
246138 | public |               prepared_cubes_p00070_p00075 |     1 |           32s |            277 MB |       2 |                 924ms
246151 | public |               prepared_cubes_p00075_p00080 |     1 |           32s |            283 MB |       2 |                 626ms
245995 | public |               prepared_cubes_p00015_p00020 |     1 |           29s |            221 MB |       2 |                 425ms
246060 | public |               prepared_cubes_p00040_p00045 |     1 |           27s |            248 MB |       2 |                 559ms
532569 | public |  mview_table_1_128_128_128_2_p00010_p00020 |     1 |           25s |            213 MB |       2 |                 7s284
246034 | public |               prepared_cubes_p00030_p00035 |     1 |           26s |            238 MB |       2 |                 489ms
246008 | public |               prepared_cubes_p00020_p00025 |     1 |           26s |            238 MB |       2 |                 509ms
246099 | public |               prepared_cubes_p00055_p00060 |     1 |           24s |            214 MB |       2 |                 436ms
246229 | public |               prepared_cubes_p00105_p00110 |     1 |           23s |            210 MB |       2 |                 501ms
246021 | public |               prepared_cubes_p00025_p00030 |     1 |           21s |            182 MB |       2 |                 451ms
531711 | public | mview_table_16_128_128_128_2_p00090_p00100 |     1 |           22s |            187 MB |       2 |                 1s778
246294 | public |               prepared_cubes_p00130_p00135 |     1 |           19s |            173 MB |       2 |                 413ms
246307 | public |               prepared_cubes_p00135_p00140 |     1 |           18s |            156 MB |       2 |                 376ms
246320 | public |               prepared_cubes_p00140_p00145 |     1 |           17s |            156 MB |       2 |                 321ms
531704 | public | mview_table_16_128_128_128_2_p00080_p00090 |     1 |           14s |            129 MB |       2 |                 1s195
531872 | public | mview_table_16_128_128_128_2_p00320_p00330 |     1 |         8s690 |             73 MB |       2 |                 486ms
531732 | public | mview_table_16_128_128_128_2_p00120_p00130 |     1 |         7s624 |             64 MB |       2 |                 383ms
531879 | public | mview_table_16_128_128_128_2_p00330_p00340 |     1 |         7s718 |             64 MB |       2 |                 419ms
531886 | public | mview_table_16_128_128_128_2_p00340_p00350 |     1 |         7s682 |             64 MB |       2 |                 410ms
531893 | public | mview_table_16_128_128_128_2_p00350_p00360 |     1 |         7s362 |             65 MB |       2 |                 450ms
531900 | public | mview_table_16_128_128_128_2_p00360_p00370 |     1 |         8s220 |             64 MB |       2 |                 359ms
531690 | public | mview_table_16_128_128_128_2_p00060_p00070 |     1 |         6s740 |             58 MB |       2 |                 341ms
531907 | public | mview_table_16_128_128_128_2_p00370_p00380 |     1 |         6s202 |             56 MB |       2 |                 407ms
531683 | public | mview_table_16_128_128_128_2_p00050_p00060 |     1 |         6s308 |             55 MB |       2 |                 336ms
531718 | public | mview_table_16_128_128_128_2_p00100_p00110 |     1 |         6s578 |             53 MB |       2 |                 343ms
531676 | public | mview_table_16_128_128_128_2_p00040_p00050 |     1 |         6s629 |             55 MB |       2 |                 356ms
531753 | public | mview_table_16_128_128_128_2_p00150_p00160 |     1 |         6s891 |             50 MB |       2 |                 663ms
531697 | public | mview_table_16_128_128_128_2_p00070_p00080 |     1 |         5s078 |             42 MB |       2 |                 248ms
531669 | public | mview_table_16_128_128_128_2_p00030_p00040 |     1 |         5s748 |             41 MB |       2 |                 220ms
531746 | public | mview_table_16_128_128_128_2_p00140_p00150 |     1 |         4s486 |             33 MB |       2 |                 162ms
531662 | public | mview_table_16_128_128_128_2_p00020_p00030 |     1 |         3s019 |             31 MB |       2 |                 156ms
531739 | public | mview_table_16_128_128_128_2_p00130_p00140 |     1 |         3s531 |             24 MB |       2 |                 117ms
531725 | public | mview_table_16_128_128_128_2_p00110_p00120 |     1 |         2s663 |             24 MB |       2 |                 129ms
533080 | public |  mview_table_1_128_128_128_2_p00740_p00750 |     1 |         3s447 |             24 MB |       2 |                 150ms
532166 | public | mview_table_16_128_128_128_2_p00740_p00750 |     1 |         3s197 |             24 MB |       2 |                 150ms
529562 | public |                  prepared_tiles_view_table |     1 |         3s889 |             25 MB |       0 |                   0ms
325007 | public |               prepared_cubes_p00745_p00750 |     1 |         2s162 |             19 MB |       2 |                  93ms
533094 | public |  mview_table_1_128_128_128_2_p00760_p00770 |     1 |         3s317 |             26 MB |       2 |                 123ms
532180 | public | mview_table_16_128_128_128_2_p00760_p00770 |     1 |         3s212 |             26 MB |       2 |                 131ms
531655 | public | mview_table_16_128_128_128_2_p00010_p00020 |     1 |         1s920 |             16 MB |       2 |                  82ms
325046 | public |               prepared_cubes_p00760_p00765 |     1 |         2s159 |             15 MB |       2 |                  61ms
533087 | public |  mview_table_1_128_128_128_2_p00750_p00760 |     1 |         1s667 |             14 MB |       2 |                  96ms
532173 | public | mview_table_16_128_128_128_2_p00750_p00760 |     1 |         1s558 |             14 MB |       2 |                  91ms
325020 | public |               prepared_cubes_p00750_p00755 |     1 |         1s425 |             11 MB |       2 |                  71ms
325059 | public |               prepared_cubes_p00765_p00770 |     1 |         1s606 |             12 MB |       2 |                  72ms
533073 | public |  mview_table_1_128_128_128_2_p00730_p00740 |     1 |         1s457 |             10 MB |       2 |                  70ms
532159 | public | mview_table_16_128_128_128_2_p00730_p00740 |     1 |         1s431 |             10 MB |       2 |                  66ms
533031 | public |  mview_table_1_128_128_128_2_p00670_p00680 |     1 |         1s318 |             10 MB |       2 |                  69ms
532117 | public | mview_table_16_128_128_128_2_p00670_p00680 |     1 |         963ms |             10 MB |       2 |                 149ms
533101 | public |  mview_table_1_128_128_128_2_p00770_p00780 |     1 |         1s337 |             10 MB |       2 |                  67ms
532187 | public | mview_table_16_128_128_128_2_p00770_p00780 |     1 |         1s354 |             10 MB |       2 |                  54ms
533115 | public |  mview_table_1_128_128_128_2_p00790_p00800 |     1 |         1s087 |           9459 kB |       2 |                  54ms
532201 | public | mview_table_16_128_128_128_2_p00790_p00800 |     1 |         779ms |           9471 kB |       2 |                  55ms
 81821 | public |                             prepared_tiles |     1 |         3s500 |             22 MB |       1 |                   7ms
533045 | public |  mview_table_1_128_128_128_2_p00690_p00700 |     1 |         1s252 |          10125 kB |       2 |                  54ms
532131 | public | mview_table_16_128_128_128_2_p00690_p00700 |     1 |         1s241 |          10136 kB |       2 |                  69ms
533059 | public |  mview_table_1_128_128_128_2_p00710_p00720 |     1 |         1s246 |           9442 kB |       2 |                  60ms
532145 | public | mview_table_16_128_128_128_2_p00710_p00720 |     1 |         1s161 |           9453 kB |       2 |                  54ms
533108 | public |  mview_table_1_128_128_128_2_p00780_p00790 |     1 |         842ms |           8303 kB |       2 |                  54ms
532194 | public | mview_table_16_128_128_128_2_p00780_p00790 |     1 |         1s066 |           8313 kB |       2 |                  50ms
246242 | public |               prepared_cubes_p00110_p00115 |     1 |         815ms |           7041 kB |       2 |                  30ms
533038 | public |  mview_table_1_128_128_128_2_p00680_p00690 |     1 |         1s036 |           7934 kB |       2 |                  59ms
533052 | public |  mview_table_1_128_128_128_2_p00700_p00710 |     1 |         1s006 |           7886 kB |       2 |                  37ms
532124 | public | mview_table_16_128_128_128_2_p00680_p00690 |     1 |         904ms |           7943 kB |       2 |                  65ms
532138 | public | mview_table_16_128_128_128_2_p00700_p00710 |     1 |         981ms |           7895 kB |       2 |                  53ms
324981 | public |               prepared_cubes_p00735_p00740 |     1 |         780ms |           6014 kB |       2 |                  35ms
324812 | public |               prepared_cubes_p00670_p00675 |     1 |         650ms |           5904 kB |       2 |                  39ms
325072 | public |               prepared_cubes_p00770_p00775 |     1 |         846ms |           6382 kB |       2 |                  37ms
324994 | public |               prepared_cubes_p00740_p00745 |     1 |         645ms |           5968 kB |       2 |                  35ms
324864 | public |               prepared_cubes_p00690_p00695 |     1 |         667ms |           5841 kB |       2 |                  35ms
324968 | public |               prepared_cubes_p00730_p00735 |     1 |         612ms |           5312 kB |       2 |                  31ms
325137 | public |               prepared_cubes_p00795_p00800 |     1 |         586ms |           5051 kB |       2 |                  36ms
324916 | public |               prepared_cubes_p00710_p00715 |     1 |         639ms |           5720 kB |       2 |                  33ms
533017 | public |  mview_table_1_128_128_128_2_p00650_p00660 |     1 |         631ms |           6246 kB |       2 |                  48ms
533024 | public |  mview_table_1_128_128_128_2_p00660_p00670 |     1 |         577ms |           6452 kB |       2 |                  46ms
324825 | public |               prepared_cubes_p00675_p00680 |     1 |         412ms |           4969 kB |       2 |                  33ms
532103 | public | mview_table_16_128_128_128_2_p00650_p00660 |     1 |         687ms |           6253 kB |       2 |                  48ms
532110 | public | mview_table_16_128_128_128_2_p00660_p00670 |     1 |         681ms |           6459 kB |       2 |                  50ms
325098 | public |               prepared_cubes_p00780_p00785 |     1 |         513ms |           4800 kB |       2 |                  30ms
325085 | public |               prepared_cubes_p00775_p00780 |     1 |         524ms |           4952 kB |       2 |                  33ms
533066 | public |  mview_table_1_128_128_128_2_p00720_p00730 |     1 |         573ms |           5645 kB |       2 |                  49ms
324890 | public |               prepared_cubes_p00700_p00705 |     1 |         530ms |           4583 kB |       2 |                  36ms
324851 | public |               prepared_cubes_p00685_p00690 |     1 |         505ms |           4787 kB |       2 |                  27ms
532152 | public | mview_table_16_128_128_128_2_p00720_p00730 |     1 |         661ms |           5652 kB |       2 |                  43ms
325124 | public |               prepared_cubes_p00790_p00795 |     1 |         516ms |           4793 kB |       2 |                  48ms
324877 | public |               prepared_cubes_p00695_p00700 |     1 |         536ms |           4667 kB |       2 |                  38ms
324955 | public |               prepared_cubes_p00725_p00730 |     1 |         457ms |           4355 kB |       2 |                  31ms
324773 | public |               prepared_cubes_p00655_p00660 |     1 |         514ms |           4296 kB |       2 |                  37ms
324929 | public |               prepared_cubes_p00715_p00720 |     1 |         506ms |           4088 kB |       2 |                  29ms
325111 | public |               prepared_cubes_p00785_p00790 |     1 |         484ms |           3835 kB |       2 |                  25ms
324838 | public |               prepared_cubes_p00680_p00685 |     1 |         339ms |           3455 kB |       2 |                  22ms
324903 | public |               prepared_cubes_p00705_p00710 |     1 |         306ms |           3617 kB |       2 |                  28ms
324786 | public |               prepared_cubes_p00660_p00665 |     1 |         363ms |           3336 kB |       2 |                  49ms
325033 | public |               prepared_cubes_p00755_p00760 |     1 |         353ms |           3734 kB |       2 |                  53ms
324799 | public |               prepared_cubes_p00665_p00670 |     1 |         334ms |           3365 kB |       2 |                  69ms
324760 | public |               prepared_cubes_p00650_p00655 |     1 |         262ms |           2200 kB |       2 |                  52ms
324942 | public |               prepared_cubes_p00720_p00725 |     1 |         198ms |           1516 kB |       2 |                  29ms
533010 | public |  mview_table_1_128_128_128_2_p00640_p00650 |     1 |         159ms |           1022 kB |       2 |                   7ms
532096 | public | mview_table_16_128_128_128_2_p00640_p00650 |     1 |         158ms |           1023 kB |       2 |                  19ms
324747 | public |               prepared_cubes_p00645_p00650 |     1 |          87ms |            575 kB |       2 |                  29ms
325207 | public |               prepared_cubes_p00640_p00645 |     1 |          64ms |            477 kB |       2 |                  56ms
 17781 | public |                                   prepared |     1 |          38ms |            187 kB |       1 |                  16ms
 92313 | public |                g_sheet_master_imaging_list |     1 |          29ms |             41 kB |       1 |                  37ms
242840 | public |               prepared_cubes_p00416_p00432 |     1 |          41ms |               0 B |       2 |                  59ms
242851 | public |               prepared_cubes_p00432_p00440 |     1 |          39ms |               0 B |       2 |                  58ms
242862 | public |               prepared_cubes_p00440_p00450 |     1 |          40ms |               0 B |       2 |                  28ms
242873 | public |               prepared_cubes_p00450_p00470 |     1 |          43ms |               0 B |       2 |                  39ms
242884 | public |               prepared_cubes_p00470_p00480 |     1 |          43ms |               0 B |       2 |                  34ms
242895 | public |               prepared_cubes_p00480_p00490 |     1 |          33ms |               0 B |       2 |                  35ms
242906 | public |               prepared_cubes_p00490_p00500 |     1 |          39ms |               0 B |       2 |                  33ms
242917 | public |               prepared_cubes_p00500_p00510 |     1 |          41ms |               0 B |       2 |                  33ms
242928 | public |               prepared_cubes_p00510_p00560 |     1 |          44ms |               0 B |       2 |                  25ms
242939 | public |               prepared_cubes_p00560_p00570 |     1 |          42ms |               0 B |       2 |                  30ms
242950 | public |               prepared_cubes_p00570_p00575 |     1 |          39ms |               0 B |       2 |                  23ms
242961 | public |               prepared_cubes_p00575_p00577 |     1 |          43ms |               0 B |       2 |                  33ms
242972 | public |               prepared_cubes_p00577_p00578 |     1 |          32ms |               0 B |       2 |                  29ms
242983 | public |               prepared_cubes_p00578_p00582 |     1 |          41ms |               0 B |       2 |                  27ms
242994 | public |               prepared_cubes_p00582_p00583 |     1 |          42ms |               0 B |       2 |                  27ms
243005 | public |               prepared_cubes_p00583_p00586 |     1 |          47ms |               0 B |       2 |                  34ms
243016 | public |               prepared_cubes_p00586_p00587 |     1 |          45ms |               0 B |       2 |                  27ms
243027 | public |               prepared_cubes_p00587_p00589 |     1 |          48ms |               0 B |       2 |                  31ms
243038 | public |               prepared_cubes_p00589_p00600 |     1 |          38ms |               0 B |       2 |                  34ms
243049 | public |               prepared_cubes_p00600_p00605 |     1 |          50ms |               0 B |       2 |                  32ms
243060 | public |               prepared_cubes_p00605_p00610 |     1 |          40ms |               0 B |       2 |                  39ms
243071 | public |               prepared_cubes_p00610_p00615 |     1 |          49ms |               0 B |       2 |                  38ms
243082 | public |               prepared_cubes_p00615_p00620 |     1 |          41ms |               0 B |       2 |                  29ms
243093 | public |               prepared_cubes_p00620_p00624 |     1 |          46ms |               0 B |       2 |                  23ms
243104 | public |               prepared_cubes_p00624_p00626 |     1 |          60ms |               0 B |       2 |                  32ms
243365 | public |               prepared_cubes_p00626_p00630 |     1 |          58ms |               0 B |       2 |                  29ms
246372 | public |               prepared_cubes_p00160_p00165 |     1 |          63ms |               0 B |       2 |                  29ms
246385 | public |               prepared_cubes_p00165_p00170 |     1 |          81ms |               0 B |       2 |                  33ms
246398 | public |               prepared_cubes_p00170_p00175 |     1 |          69ms |               0 B |       2 |                  27ms
246411 | public |               prepared_cubes_p00175_p00180 |     1 |          66ms |               0 B |       2 |                  28ms
246424 | public |               prepared_cubes_p00180_p00185 |     1 |          65ms |               0 B |       2 |                  30ms
246437 | public |               prepared_cubes_p00185_p00190 |     1 |          56ms |               0 B |       2 |                  25ms
246450 | public |               prepared_cubes_p00190_p00195 |     1 |          49ms |               0 B |       2 |                  31ms
246463 | public |               prepared_cubes_p00195_p00200 |     1 |          52ms |               0 B |       2 |                  34ms
246476 | public |               prepared_cubes_p00200_p00205 |     1 |          65ms |               0 B |       2 |                  36ms
246489 | public |               prepared_cubes_p00205_p00210 |     1 |          61ms |               0 B |       2 |                  35ms
246502 | public |               prepared_cubes_p00210_p00215 |     1 |          65ms |               0 B |       2 |                  29ms
246515 | public |               prepared_cubes_p00215_p00220 |     1 |          59ms |               0 B |       2 |                  30ms
246528 | public |               prepared_cubes_p00220_p00225 |     1 |          56ms |               0 B |       2 |                  27ms
246541 | public |               prepared_cubes_p00225_p00230 |     1 |          58ms |               0 B |       2 |                  20ms
246554 | public |               prepared_cubes_p00230_p00235 |     1 |          63ms |               0 B |       2 |                  14ms
246567 | public |               prepared_cubes_p00235_p00240 |     1 |          60ms |               0 B |       2 |                  16ms
246580 | public |               prepared_cubes_p00240_p00245 |     1 |          77ms |               0 B |       2 |                  14ms
246593 | public |               prepared_cubes_p00245_p00250 |     1 |          76ms |               0 B |       2 |                  13ms
246606 | public |               prepared_cubes_p00250_p00255 |     1 |          67ms |               0 B |       2 |                  13ms
246619 | public |               prepared_cubes_p00255_p00260 |     1 |          74ms |               0 B |       2 |                  12ms
246632 | public |               prepared_cubes_p00260_p00265 |     1 |          62ms |               0 B |       2 |                  13ms
246645 | public |               prepared_cubes_p00265_p00270 |     1 |          64ms |               0 B |       2 |                  13ms
246658 | public |               prepared_cubes_p00270_p00275 |     1 |          57ms |               0 B |       2 |                  20ms
246671 | public |               prepared_cubes_p00275_p00280 |     1 |          70ms |               0 B |       2 |                  12ms
246684 | public |               prepared_cubes_p00280_p00285 |     1 |          53ms |               0 B |       2 |                  33ms
246697 | public |               prepared_cubes_p00285_p00290 |     1 |          54ms |               0 B |       2 |                  30ms
246710 | public |               prepared_cubes_p00290_p00295 |     1 |          44ms |               0 B |       2 |                  16ms
246723 | public |               prepared_cubes_p00295_p00300 |     1 |          41ms |               0 B |       2 |                  12ms
246736 | public |               prepared_cubes_p00300_p00305 |     1 |          44ms |               0 B |       2 |                  17ms
246749 | public |               prepared_cubes_p00305_p00310 |     1 |          57ms |               0 B |       2 |                  42ms
246762 | public |               prepared_cubes_p00310_p00315 |     1 |          44ms |               0 B |       2 |                  14ms
246775 | public |               prepared_cubes_p00315_p00320 |     1 |          39ms |               0 B |       2 |                  15ms
242800 | public |                            mview_dashboard |     1 |          33ms |             423 B |       0 |                   0ms
245956 | public |               prepared_cubes_p00000_p00005 |     1 |          40ms |               0 B |       2 |                  16ms
245969 | public |               prepared_cubes_p00005_p00010 |     1 |          35ms |               0 B |       2 |                  14ms
245982 | public |               prepared_cubes_p00010_p00015 |     1 |          40ms |               0 B |       2 |                  14ms
246359 | public |               prepared_cubes_p00155_p00160 |     1 |          47ms |               0 B |       2 |                  13ms
325243 | public |               prepared_cubes_p00630_p00635 |     1 |          46ms |               0 B |       2 |                  16ms
325256 | public |               prepared_cubes_p00635_p00640 |     1 |          41ms |               0 B |       2 |                  15ms
331053 | public |               prepared_cubes_p00800_p00805 |     1 |          48ms |               0 B |       2 |                  14ms
331066 | public |               prepared_cubes_p00805_p00810 |     1 |          51ms |               0 B |       2 |                  14ms
331079 | public |               prepared_cubes_p00810_p00815 |     1 |          43ms |               0 B |       2 |                  14ms
331092 | public |               prepared_cubes_p00815_p00820 |     1 |          45ms |               0 B |       2 |                  12ms
331105 | public |               prepared_cubes_p00820_p00825 |     1 |          38ms |               0 B |       2 |                  13ms
331118 | public |               prepared_cubes_p00825_p00830 |     1 |          39ms |               0 B |       2 |                  10ms
331131 | public |               prepared_cubes_p00830_p00835 |     1 |          39ms |               0 B |       2 |                  14ms
331144 | public |               prepared_cubes_p00835_p00840 |     1 |          49ms |               0 B |       2 |                  15ms
331157 | public |               prepared_cubes_p00840_p00845 |     1 |          50ms |               0 B |       2 |                  16ms
331170 | public |               prepared_cubes_p00845_p00850 |     1 |          46ms |               0 B |       2 |                  14ms
331183 | public |               prepared_cubes_p00850_p00855 |     1 |          34ms |               0 B |       2 |                  15ms
331196 | public |               prepared_cubes_p00855_p00860 |     1 |          42ms |               0 B |       2 |                  15ms
331209 | public |               prepared_cubes_p00860_p00865 |     1 |          48ms |               0 B |       2 |                  17ms
331222 | public |               prepared_cubes_p00865_p00870 |     1 |          48ms |               0 B |       2 |                  20ms
331235 | public |               prepared_cubes_p00870_p00875 |     1 |          41ms |               0 B |       2 |                  21ms
331248 | public |               prepared_cubes_p00875_p00880 |     1 |          55ms |               0 B |       2 |                  13ms
331261 | public |               prepared_cubes_p00880_p00885 |     1 |          56ms |               0 B |       2 |                  10ms
331274 | public |               prepared_cubes_p00885_p00890 |     1 |          41ms |               0 B |       2 |                  11ms
331287 | public |               prepared_cubes_p00890_p00895 |     1 |          30ms |               0 B |       2 |                  15ms
331300 | public |               prepared_cubes_p00895_p00900 |     1 |          41ms |               0 B |       2 |                  12ms
331313 | public |               prepared_cubes_p00900_p00905 |     1 |          42ms |               0 B |       2 |                  20ms
331326 | public |               prepared_cubes_p00905_p00910 |     1 |          50ms |               0 B |       2 |                  21ms
331339 | public |               prepared_cubes_p00910_p00915 |     1 |          42ms |               0 B |       2 |                  23ms
331352 | public |               prepared_cubes_p00915_p00920 |     1 |          37ms |               0 B |       2 |                  15ms
331365 | public |               prepared_cubes_p00920_p00925 |     1 |          36ms |               0 B |       2 |                  12ms
331378 | public |               prepared_cubes_p00925_p00930 |     1 |          37ms |               0 B |       2 |                  15ms
331391 | public |               prepared_cubes_p00930_p00935 |     1 |          33ms |               0 B |       2 |                  18ms
331404 | public |               prepared_cubes_p00935_p00940 |     1 |          35ms |               0 B |       2 |                  16ms
331417 | public |               prepared_cubes_p00940_p00945 |     1 |          40ms |               0 B |       2 |                  12ms
331430 | public |               prepared_cubes_p00945_p00950 |     1 |          47ms |               0 B |       2 |                  12ms
331443 | public |               prepared_cubes_p00950_p00955 |     1 |          37ms |               0 B |       2 |                  10ms
331456 | public |               prepared_cubes_p00955_p00960 |     1 |          44ms |               0 B |       2 |                  11ms
331469 | public |               prepared_cubes_p00960_p00965 |     1 |          40ms |               0 B |       2 |                  15ms
331482 | public |               prepared_cubes_p00965_p00970 |     1 |          42ms |               0 B |       2 |                  12ms
331495 | public |               prepared_cubes_p00970_p00975 |     1 |          43ms |               0 B |       2 |                  19ms
331508 | public |               prepared_cubes_p00975_p00980 |     1 |          43ms |               0 B |       2 |                   9ms
331521 | public |               prepared_cubes_p00980_p00985 |     1 |          41ms |               0 B |       2 |                  15ms
331534 | public |               prepared_cubes_p00985_p00990 |     1 |          54ms |               0 B |       2 |                  15ms
331547 | public |               prepared_cubes_p00990_p00995 |     1 |          41ms |               0 B |       2 |                  15ms
331560 | public |               prepared_cubes_p00995_p01000 |     1 |          45ms |               0 B |       2 |                  14ms
531648 | public | mview_table_16_128_128_128_2_p00000_p00010 |     1 |          48ms |               0 B |       2 |                  10ms
531760 | public | mview_table_16_128_128_128_2_p00160_p00170 |     1 |          41ms |               0 B |       2 |                   9ms
531767 | public | mview_table_16_128_128_128_2_p00170_p00180 |     1 |          46ms |               0 B |       2 |                  12ms
531774 | public | mview_table_16_128_128_128_2_p00180_p00190 |     1 |          40ms |               0 B |       2 |                  13ms
531781 | public | mview_table_16_128_128_128_2_p00190_p00200 |     1 |          40ms |               0 B |       2 |                  16ms
531788 | public | mview_table_16_128_128_128_2_p00200_p00210 |     1 |          52ms |               0 B |       2 |                  11ms
531795 | public | mview_table_16_128_128_128_2_p00210_p00220 |     1 |          41ms |               0 B |       2 |                  10ms
531802 | public | mview_table_16_128_128_128_2_p00220_p00230 |     1 |          43ms |               0 B |       2 |                  11ms
531809 | public | mview_table_16_128_128_128_2_p00230_p00240 |     1 |          43ms |               0 B |       2 |                  12ms
531816 | public | mview_table_16_128_128_128_2_p00240_p00250 |     1 |          41ms |               0 B |       2 |                  15ms
531823 | public | mview_table_16_128_128_128_2_p00250_p00260 |     1 |          41ms |               0 B |       2 |                  12ms
531830 | public | mview_table_16_128_128_128_2_p00260_p00270 |     1 |          40ms |               0 B |       2 |                  10ms
531837 | public | mview_table_16_128_128_128_2_p00270_p00280 |     1 |          40ms |               0 B |       2 |                  13ms
531844 | public | mview_table_16_128_128_128_2_p00280_p00290 |     1 |          44ms |               0 B |       2 |                   8ms
531851 | public | mview_table_16_128_128_128_2_p00290_p00300 |     1 |          45ms |               0 B |       2 |                  12ms
531858 | public | mview_table_16_128_128_128_2_p00300_p00310 |     1 |          41ms |               0 B |       2 |                  17ms
531865 | public | mview_table_16_128_128_128_2_p00310_p00320 |     1 |          37ms |               0 B |       2 |                  19ms
531914 | public | mview_table_16_128_128_128_2_p00380_p00390 |     1 |          39ms |               0 B |       2 |                  11ms
531921 | public | mview_table_16_128_128_128_2_p00390_p00400 |     1 |          40ms |               0 B |       2 |                  18ms
531928 | public | mview_table_16_128_128_128_2_p00400_p00410 |     1 |          45ms |               0 B |       2 |                  11ms
531935 | public | mview_table_16_128_128_128_2_p00410_p00420 |     1 |          56ms |               0 B |       2 |                  13ms
531942 | public | mview_table_16_128_128_128_2_p00420_p00430 |     1 |          45ms |               0 B |       2 |                   9ms
531949 | public | mview_table_16_128_128_128_2_p00430_p00440 |     1 |          48ms |               0 B |       2 |                  12ms
531956 | public | mview_table_16_128_128_128_2_p00440_p00450 |     1 |          56ms |               0 B |       2 |                  10ms
531963 | public | mview_table_16_128_128_128_2_p00450_p00460 |     1 |          61ms |               0 B |       2 |                   8ms
531970 | public | mview_table_16_128_128_128_2_p00460_p00470 |     1 |          55ms |               0 B |       2 |                  11ms
531977 | public | mview_table_16_128_128_128_2_p00470_p00480 |     1 |          59ms |               0 B |       2 |                  10ms
531984 | public | mview_table_16_128_128_128_2_p00480_p00490 |     1 |          34ms |               0 B |       2 |                   7ms
531991 | public | mview_table_16_128_128_128_2_p00490_p00500 |     1 |          43ms |               0 B |       2 |                   9ms
531998 | public | mview_table_16_128_128_128_2_p00500_p00510 |     1 |          43ms |               0 B |       2 |                  11ms
532005 | public | mview_table_16_128_128_128_2_p00510_p00520 |     1 |          45ms |               0 B |       2 |                  13ms
532012 | public | mview_table_16_128_128_128_2_p00520_p00530 |     1 |          36ms |               0 B |       2 |                  13ms
532019 | public | mview_table_16_128_128_128_2_p00530_p00540 |     1 |          45ms |               0 B |       2 |                  13ms
532026 | public | mview_table_16_128_128_128_2_p00540_p00550 |     1 |          45ms |               0 B |       2 |                  11ms
532033 | public | mview_table_16_128_128_128_2_p00550_p00560 |     1 |          46ms |               0 B |       2 |                  10ms
532040 | public | mview_table_16_128_128_128_2_p00560_p00570 |     1 |          44ms |               0 B |       2 |                  13ms
532047 | public | mview_table_16_128_128_128_2_p00570_p00580 |     1 |          46ms |               0 B |       2 |                  12ms
532054 | public | mview_table_16_128_128_128_2_p00580_p00590 |     1 |          51ms |               0 B |       2 |                  12ms
532061 | public | mview_table_16_128_128_128_2_p00590_p00600 |     1 |          44ms |               0 B |       2 |                  15ms
532068 | public | mview_table_16_128_128_128_2_p00600_p00610 |     1 |          37ms |               0 B |       2 |                  14ms
532075 | public | mview_table_16_128_128_128_2_p00610_p00620 |     1 |          38ms |               0 B |       2 |                  12ms
532082 | public | mview_table_16_128_128_128_2_p00620_p00630 |     1 |          36ms |               0 B |       2 |                  13ms
532089 | public | mview_table_16_128_128_128_2_p00630_p00640 |     1 |          46ms |               0 B |       2 |                  11ms
532208 | public | mview_table_16_128_128_128_2_p00800_p00810 |     1 |          47ms |               0 B |       2 |                  15ms
532215 | public | mview_table_16_128_128_128_2_p00810_p00820 |     1 |          41ms |               0 B |       2 |                  11ms
532222 | public | mview_table_16_128_128_128_2_p00820_p00830 |     1 |          48ms |               0 B |       2 |                  13ms
532229 | public | mview_table_16_128_128_128_2_p00830_p00840 |     1 |          38ms |               0 B |       2 |                  11ms
532236 | public | mview_table_16_128_128_128_2_p00840_p00850 |     1 |          42ms |               0 B |       2 |                  11ms
532243 | public | mview_table_16_128_128_128_2_p00850_p00860 |     1 |          41ms |               0 B |       2 |                  12ms
532250 | public | mview_table_16_128_128_128_2_p00860_p00870 |     1 |          41ms |               0 B |       2 |                  12ms
532257 | public | mview_table_16_128_128_128_2_p00870_p00880 |     1 |          44ms |               0 B |       2 |                   8ms
532264 | public | mview_table_16_128_128_128_2_p00880_p00890 |     1 |          45ms |               0 B |       2 |                  23ms
532271 | public | mview_table_16_128_128_128_2_p00890_p00900 |     1 |          45ms |               0 B |       2 |                  15ms
532278 | public | mview_table_16_128_128_128_2_p00900_p00910 |     1 |          59ms |               0 B |       2 |                  15ms
532285 | public | mview_table_16_128_128_128_2_p00910_p00920 |     1 |          39ms |               0 B |       2 |                   8ms
532292 | public | mview_table_16_128_128_128_2_p00920_p00930 |     1 |          39ms |               0 B |       2 |                  16ms
532299 | public | mview_table_16_128_128_128_2_p00930_p00940 |     1 |          41ms |               0 B |       2 |                   7ms
532306 | public | mview_table_16_128_128_128_2_p00940_p00950 |     1 |          40ms |               0 B |       2 |                  12ms
532313 | public | mview_table_16_128_128_128_2_p00950_p00960 |     1 |          34ms |               0 B |       2 |                  12ms
532320 | public | mview_table_16_128_128_128_2_p00960_p00970 |     1 |          32ms |               0 B |       2 |                  12ms
532327 | public | mview_table_16_128_128_128_2_p00970_p00980 |     1 |          42ms |               0 B |       2 |                  12ms
532334 | public | mview_table_16_128_128_128_2_p00980_p00990 |     1 |          44ms |               0 B |       2 |                  11ms
532341 | public | mview_table_16_128_128_128_2_p00990_p01000 |     1 |          50ms |               0 B |       2 |                  10ms
532562 | public |  mview_table_1_128_128_128_2_p00000_p00010 |     1 |          39ms |               0 B |       2 |                  13ms
532674 | public |  mview_table_1_128_128_128_2_p00160_p00170 |     1 |          45ms |               0 B |       2 |                  14ms
532681 | public |  mview_table_1_128_128_128_2_p00170_p00180 |     1 |          45ms |               0 B |       2 |                  14ms
532688 | public |  mview_table_1_128_128_128_2_p00180_p00190 |     1 |          34ms |               0 B |       2 |                  13ms
532695 | public |  mview_table_1_128_128_128_2_p00190_p00200 |     1 |          35ms |               0 B |       2 |                  17ms
532702 | public |  mview_table_1_128_128_128_2_p00200_p00210 |     1 |          37ms |               0 B |       2 |                  12ms
532709 | public |  mview_table_1_128_128_128_2_p00210_p00220 |     1 |          43ms |               0 B |       2 |                  11ms
532716 | public |  mview_table_1_128_128_128_2_p00220_p00230 |     1 |          46ms |               0 B |       2 |                  12ms
532723 | public |  mview_table_1_128_128_128_2_p00230_p00240 |     1 |          28ms |               0 B |       2 |                   9ms
532730 | public |  mview_table_1_128_128_128_2_p00240_p00250 |     1 |          32ms |               0 B |       2 |                  11ms
532737 | public |  mview_table_1_128_128_128_2_p00250_p00260 |     1 |          36ms |               0 B |       2 |                  13ms
532744 | public |  mview_table_1_128_128_128_2_p00260_p00270 |     1 |          35ms |               0 B |       2 |                  13ms
532751 | public |  mview_table_1_128_128_128_2_p00270_p00280 |     1 |          37ms |               0 B |       2 |                  10ms
532758 | public |  mview_table_1_128_128_128_2_p00280_p00290 |     1 |          33ms |               0 B |       2 |                  17ms
532765 | public |  mview_table_1_128_128_128_2_p00290_p00300 |     1 |          42ms |               0 B |       2 |                  15ms
532772 | public |  mview_table_1_128_128_128_2_p00300_p00310 |     1 |          59ms |               0 B |       2 |                  10ms
532779 | public |  mview_table_1_128_128_128_2_p00310_p00320 |     1 |          39ms |               0 B |       2 |                  11ms
532828 | public |  mview_table_1_128_128_128_2_p00380_p00390 |     1 |          35ms |               0 B |       2 |                  11ms
532835 | public |  mview_table_1_128_128_128_2_p00390_p00400 |     1 |          35ms |               0 B |       2 |                  15ms
532842 | public |  mview_table_1_128_128_128_2_p00400_p00410 |     1 |          46ms |               0 B |       2 |                  12ms
532849 | public |  mview_table_1_128_128_128_2_p00410_p00420 |     1 |          50ms |               0 B |       2 |                  11ms
532856 | public |  mview_table_1_128_128_128_2_p00420_p00430 |     1 |          49ms |               0 B |       2 |                   9ms
532863 | public |  mview_table_1_128_128_128_2_p00430_p00440 |     1 |          48ms |               0 B |       2 |                  12ms
532870 | public |  mview_table_1_128_128_128_2_p00440_p00450 |     1 |          37ms |               0 B |       2 |                  15ms
532877 | public |  mview_table_1_128_128_128_2_p00450_p00460 |     1 |          38ms |               0 B |       2 |                  19ms
532884 | public |  mview_table_1_128_128_128_2_p00460_p00470 |     1 |          33ms |               0 B |       2 |                  13ms
532891 | public |  mview_table_1_128_128_128_2_p00470_p00480 |     1 |          39ms |               0 B |       2 |                  18ms
532898 | public |  mview_table_1_128_128_128_2_p00480_p00490 |     1 |          43ms |               0 B |       2 |                  11ms
532905 | public |  mview_table_1_128_128_128_2_p00490_p00500 |     1 |          51ms |               0 B |       2 |                  12ms
532912 | public |  mview_table_1_128_128_128_2_p00500_p00510 |     1 |          52ms |               0 B |       2 |                  11ms
532919 | public |  mview_table_1_128_128_128_2_p00510_p00520 |     1 |          57ms |               0 B |       2 |                  12ms
532926 | public |  mview_table_1_128_128_128_2_p00520_p00530 |     1 |          38ms |               0 B |       2 |                  12ms
532933 | public |  mview_table_1_128_128_128_2_p00530_p00540 |     1 |          41ms |               0 B |       2 |                  17ms
532940 | public |  mview_table_1_128_128_128_2_p00540_p00550 |     1 |          39ms |               0 B |       2 |                  11ms
532947 | public |  mview_table_1_128_128_128_2_p00550_p00560 |     1 |          46ms |               0 B |       2 |                  17ms
532954 | public |  mview_table_1_128_128_128_2_p00560_p00570 |     1 |          43ms |               0 B |       2 |                  15ms
532961 | public |  mview_table_1_128_128_128_2_p00570_p00580 |     1 |          44ms |               0 B |       2 |                   9ms
532968 | public |  mview_table_1_128_128_128_2_p00580_p00590 |     1 |          44ms |               0 B |       2 |                  17ms
532975 | public |  mview_table_1_128_128_128_2_p00590_p00600 |     1 |          43ms |               0 B |       2 |                  24ms
532982 | public |  mview_table_1_128_128_128_2_p00600_p00610 |     1 |          42ms |               0 B |       2 |                  11ms
532989 | public |  mview_table_1_128_128_128_2_p00610_p00620 |     1 |          42ms |               0 B |       2 |                  11ms
532996 | public |  mview_table_1_128_128_128_2_p00620_p00630 |     1 |          45ms |               0 B |       2 |                  11ms
533003 | public |  mview_table_1_128_128_128_2_p00630_p00640 |     1 |          48ms |               0 B |       2 |                  12ms
533122 | public |  mview_table_1_128_128_128_2_p00800_p00810 |     1 |          49ms |               0 B |       2 |                   8ms
533129 | public |  mview_table_1_128_128_128_2_p00810_p00820 |     1 |          36ms |               0 B |       2 |                   9ms
533136 | public |  mview_table_1_128_128_128_2_p00820_p00830 |     1 |          34ms |               0 B |       2 |                  10ms
533143 | public |  mview_table_1_128_128_128_2_p00830_p00840 |     1 |          36ms |               0 B |       2 |                  11ms
533150 | public |  mview_table_1_128_128_128_2_p00840_p00850 |     1 |          49ms |               0 B |       2 |                  13ms
533157 | public |  mview_table_1_128_128_128_2_p00850_p00860 |     1 |          43ms |               0 B |       2 |                  12ms
533164 | public |  mview_table_1_128_128_128_2_p00860_p00870 |     1 |          44ms |               0 B |       2 |                  12ms
533171 | public |  mview_table_1_128_128_128_2_p00870_p00880 |     1 |          47ms |               0 B |       2 |                  10ms
533178 | public |  mview_table_1_128_128_128_2_p00880_p00890 |     1 |          43ms |               0 B |       2 |                  13ms
533185 | public |  mview_table_1_128_128_128_2_p00890_p00900 |     1 |          40ms |               0 B |       2 |                  10ms
533192 | public |  mview_table_1_128_128_128_2_p00900_p00910 |     1 |          41ms |               0 B |       2 |                  10ms
533199 | public |  mview_table_1_128_128_128_2_p00910_p00920 |     1 |          45ms |               0 B |       2 |                  10ms
533206 | public |  mview_table_1_128_128_128_2_p00920_p00930 |     1 |          38ms |               0 B |       2 |                  11ms
533213 | public |  mview_table_1_128_128_128_2_p00930_p00940 |     1 |          45ms |               0 B |       2 |                  11ms
533220 | public |  mview_table_1_128_128_128_2_p00940_p00950 |     1 |          45ms |               0 B |       2 |                  10ms
533227 | public |  mview_table_1_128_128_128_2_p00950_p00960 |     1 |          44ms |               0 B |       2 |                  11ms
533234 | public |  mview_table_1_128_128_128_2_p00960_p00970 |     1 |          40ms |               0 B |       2 |                  12ms
533241 | public |  mview_table_1_128_128_128_2_p00970_p00980 |     1 |          37ms |               0 B |       2 |                  11ms
533248 | public |  mview_table_1_128_128_128_2_p00980_p00990 |     1 |          41ms |               0 B |       2 |                  14ms
533255 | public |  mview_table_1_128_128_128_2_p00990_p01000 |     1 |          50ms |               0 B |       2 |                  10ms


                                               Step   Connection    Duration    Transfer   Concurrency
 --------------------------------------------------   ----------  ----------  ----------  ------------
   Catalog Queries (table ordering, filtering, etc)       source       2s102                         1
                                        Dump Schema       source       1s471                         1
                                     Prepare Schema       target       1s050                         1
      COPY, INDEX, CONSTRAINTS, VACUUM (wall clock)         both      10m58s                        20
                                  COPY (cumulative)         both       1h12m       64 GB             8
                          CREATE INDEX (cumulative)       target      23m32s                         4
                           CONSTRAINTS (cumulative)       target       343ms                         4
                                VACUUM (cumulative)       target       6m15s                         8
                                    Reset Sequences         both        13ms                         1
                         Large Objects (cumulative)       (null)         0ms                         0
                                    Finalize Schema         both      11m00s                         4
 --------------------------------------------------   ----------  ----------  ----------  ------------
                          Total Wall Clock Duration         both      22m01s                        28

Stopped auto-started sandbox Postgres.

```shell
./sandbox_cli archive
time ./sandbox_cli archive
# real    1m47.954s
```

#### Sync (incremental updates after `clone`).

```shell
time ./sandbox_cli sync
```

```shell
Target Postgres already running on 5433; reusing it.
Work dir: /workspace/cell_observatory_platform/scripts/db/pgcopydb_work
Slot:     cell_observatory_sandbox
Removing clone snapshot marker for follow: /workspace/cell_observatory_platform/scripts/db/pgcopydb_work/snapshot
Resetting existing replication slot before sync: cell_observatory_sandbox
Resetting existing replication origin before sync: cell_observatory_sandbox
End LSN:  A1/970001B0 (one-shot catch-up)
2026-03-31 19:37:11.963 14 INFO   main.c:136                Running pgcopydb version 0.17-1.pgdg24.04+1 from "/usr/bin/pgcopydb"
2026-03-31 19:37:11.965 14 INFO   cli_common.c:1225         [SOURCE] Copying database from "postgres://postgres@db.cdgqohnoqldocuiwptmt.supabase.co:5432/postgres?keepalives=1&keepalives_idle=10&keepalives_interval=10&keepalives_count=60&sslmode=disable"
2026-03-31 19:37:11.965 14 INFO   cli_common.c:1226         [TARGET] Copying database into "postgres://postgres@127.0.0.1:5433/postgres?keepalives=1&keepalives_idle=10&keepalives_interval=10&keepalives_count=60"
2026-03-31 19:37:12.015 14 INFO   copydb.c:105              Using work dir "/workspace/cell_observatory_platform/scripts/db/pgcopydb_work"
2026-03-31 19:37:12.075 14 INFO   pgsql.c:3785              Created logical replication slot "cell_observatory_sandbox" with plugin "test_decoding" at A1/970001B0 and exported snapshot 00000083-0000000F-1
2026-03-31 19:37:12.090 14 INFO   ld_stream.c:2488          Created logical replication origin "cell_observatory_sandbox" at LSN A1/970001B0
2026-03-31 19:37:12.148 14 INFO   copydb_schema.c:363       A previous run has run through completion
2026-03-31 19:37:12.380 14 INFO   copydb_schema.c:1538      Found 749 indexes (supporting 176 constraints) in the target database
2026-03-31 19:37:12.391 18 INFO   ld_apply.c:362            Waiting until the pgcopydb sentinel apply is enabled
2026-03-31 19:37:12.391 16 INFO   ld_stream.c:625           Streaming is setup to end at LSN A1/970001B0
2026-03-31 19:37:12.391 16 INFO   ld_stream.c:640           Resuming streaming at LSN A1/970001B0 from replication slot "cell_observatory_sandbox"
2026-03-31 19:37:12.484 16 INFO   pgsql.c:4511              Reported write_lsn A1/970001B0, flush_lsn A1/970001B0, replay_lsn 0/0
2026-03-31 19:37:12.484 16 INFO   pgsql.c:4511              Reported write_lsn A1/970001B0, flush_lsn A1/970001B0, replay_lsn 0/0
2026-03-31 19:37:12.486 16 INFO   ld_stream.c:518           Streamed up to write_lsn A1/970001B0, flush_lsn A1/970001B0, stopping: endpos is A1/970001B0
2026-03-31 19:37:12.486 16 INFO   follow.c:704              Prefetch process has terminated
2026-03-31 19:37:12.501 17 INFO   follow.c:771              Transform process has terminated
sync: reached end LSN but pgcopydb follow did not exit; stopping follow process.
2026-03-31 19:37:42.067 20 INFO   main.c:136                Running pgcopydb version 0.17-1.pgdg24.04+1 from "/usr/bin/pgcopydb"
2026-03-31 19:37:42.069 20 INFO   cli_common.c:1225         [SOURCE] Copying database from "postgres://postgres@db.cdgqohnoqldocuiwptmt.supabase.co:5432/postgres?keepalives=1&keepalives_idle=10&keepalives_interval=10&keepalives_count=60&sslmode=disable"
2026-03-31 19:37:42.069 20 INFO   cli_common.c:1226         [TARGET] Copying database into "postgres://postgres@127.0.0.1:5433/postgres?keepalives=1&keepalives_idle=10&keepalives_interval=10&keepalives_count=60"
2026-03-31 19:37:42.123 20 INFO   copydb.c:105              Using work dir "/workspace/cell_observatory_platform/scripts/db/pgcopydb_work"
2026-03-31 19:37:42.126 20 INFO   copydb_schema.c:363       A previous run has run through completion
2026-03-31 19:37:42.128 20 INFO   copydb_schema.c:50        Re-using catalog caches
2026-03-31 19:37:42.128 20 INFO   sequences.c:290           Reset sequences values on the target database
2026-03-31 19:37:42.129 20 INFO   sequences.c:78            Fetching information for 3 sequences
#real    0m31.927s

```shell
time ./sandbox_cli archive
# real    1m47.954s
```

### Onetime copy:

Use this option to create a one-time copy of the production database into a new `sandbox.tar.zst` file that can be used for training sessions.

```shell
./sandbox_cli build
./sandbox_cli snapshot
./sandbox_cli archive
```

### CLI ([sandbox_cli.py](sandbox_cli.py))

Run from **repository root**. The repo-root wrapper `./sandbox_cli` runs the CLI in Docker by default.

```shell
./sandbox_cli --help

usage: sandbox_cli.py [-h] {build,run,snapshot,clone,sync,cleanup,archive} ...

positional arguments:
  {build,run,snapshot,clone,sync,cleanup,archive}
    build               Build Apptainer sandbox image from docker://postgres:17 and apply my-postgres.conf.
    run                 Start sandbox Postgres via Apptainer on localhost:5433.
    snapshot            Create a one-time copy of the production database into a new sandbox.tar.zst file that can be used for training sessions.
    clone               Create a longterm read replica of the production database on your local machine.
    sync                Update read replica with the latest changes from the production database.
    cleanup             Reset the sandbox database and start over.
    archive             Create scripts/db/YYYY_MM_DD_sandbox.tar.zst and copy to DATABASE_DIR

options:
  -h, --help            show this help message and exit
```