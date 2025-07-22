#!/bin/bash

# parse args from `args_parser.sh` getopts
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$DIR/args_parser.sh"

alias myjobs="watch bjobs"

alias h200s="watch --color 'bhosts -gpu h200s | grep --color=always -w \"0\"'"
alias h100s="watch --color 'bhosts -gpu h100s | grep --color=always -w \"0\"'"
alias a100s="watch --color 'bhosts -gpu a100s | grep --color=always -w \"0\"'"

alias a100jobs="watch bjobs -q gpu_a100 -u all"
alias h100jobs="watch bjobs -q gpu_h100 -u all"
alias h200jobs="watch bjobs -q gpu_h200 -u all"

if [ $partition = 'gpu_h200' ];then
    AVAL=$(bhosts -o "host_name run:-6"  h200s | grep -w "0" | awk '{print $1}' | wc -l)
    while [ $AVAL -lt $nodes ]
    do
        sleep 1
        echo "Waiting for [$AVAL/$nodes] nodes to be available on $partition queue"
        AVAL=$(bhosts -o "host_name run:-6"  h200s | grep -w "0" | awk '{print $1}' | wc -l)
    done
elif [ $partition = 'gpu_h100' ];then
    AVAL=$(bhosts -o "host_name run:-6"  h100s | grep -w "0" | awk '{print $1}' | wc -l)
    while [ $AVAL -lt $nodes ]
    do
        sleep 1
        echo "Waiting for [$AVAL/$nodes] nodes to be available on $partition queue"
        AVAL=$(bhosts -o "host_name run:-6"  h100s | grep -w "0" | awk '{print $1}' | wc -l)
    done
elif [ $partition = 'gpu_a100' ];then
    AVAL=$(bhosts -o "host_name run:-6"  a100s | grep -w "0" | awk '{print $1}' | wc -l)
    while [ $AVAL -lt $nodes ]
    do
        sleep 1
        echo "Waiting for [$AVAL/$nodes] nodes to be available on $partition queue"
        AVAL=$(bhosts -o "host_name run:-6"  a100s | grep -w "0" | awk '{print $1}' | wc -l)
    done
fi

