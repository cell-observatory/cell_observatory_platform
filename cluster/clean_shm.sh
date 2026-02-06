#!/bin/bash
echo "Cleaning up shared memory"
ipcs -m
ls -l /dev/shm | grep $USER

find /dev/shm -user $USER -delete

echo "Shared memory after cleanup"
ipcs -m
ls -l /dev/shm | grep $USER