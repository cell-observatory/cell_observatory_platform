#!/bin/bash

while getopts ":b:c:d:e:g:m:n:o:p:q:s:t:j:x:y:z:" option; do
    case "${option}" in
    b)  bind=${OPTARG} ;;
    c)  cpus=${OPTARG} ;;
    d)  storage_server=${OPTARG} ;;
    e)  env=${OPTARG} ;;
    g)  gpus=${OPTARG} ;;
    m)  mem=${OPTARG} ;;
    n)  nodes=${OPTARG} ;;
    o)  outdir=${OPTARG} ;;
    p)  partition=${OPTARG} ;;
    q)  object_store_memory=${OPTARG} ;;
    s)  workspace=${OPTARG} ;;
    t)  tasks=${OPTARG} ;;
    j)  jobname=${OPTARG} ;;
    x)  exclusive=${OPTARG} ;;
    y)  head_gpus=${OPTARG} ;;
    z)  head_cpus=${OPTARG} ;;
    *) echo "Did not supply the correct arguments"; exit 1 ;;
    esac
done
