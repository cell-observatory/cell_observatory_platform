#!/bin/bash

while getopts ":b:c:e:f:g:m:n:o:p:s:t:x:z:" option; do
    case "${option}" in
    b)  bind=${OPTARG} ;;
    c)  cpus=${OPTARG} ;;
    e)  env=${OPTARG} ;;
    f)  env_flags=${OPTARG} ;;
    g)  gpus=${OPTARG} ;;
    m)  mem=${OPTARG} ;;
    n)  nodes=${OPTARG} ;;
    o)  outdir=${OPTARG} ;;
    p)  partition=${OPTARG} ;;
    s)  workspace=${OPTARG} ;;
    t)  tasks=${OPTARG} ;;
    x)  exclusive=true ;;
    z)  head_cpus=${OPTARG} ;;
    *) echo "Did not supply the correct arguments"; exit 1 ;;
    esac
done
