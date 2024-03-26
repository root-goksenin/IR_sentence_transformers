#!/bin/bash

if [ -z "$1" ]; then
    echo "Please provide a directory name."
    exit 1
fi

python3 ranking_analysis.py $1
python3 visualize_difference.py $1

