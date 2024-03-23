#!/bin/bash
# python3 score_analysis.py $1
# python3 attentions.py $1

if [ -z "$1" ]; then
    echo "Please provide a directory name."
    exit 1
fi

# List all directories in the specified directory
logdirs=($(ls "logs/$1"))


echo ${logdirs[0]}
echo ${logdirs[1]}

tensorboard --logdir logs/$1/${logdirs[0]} --port 6006 &
tensorboard --logdir logs/$1/${logdirs[1]} --port 6007 &
