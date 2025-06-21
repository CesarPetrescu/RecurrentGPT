#!/bin/bash
set -e

# Load configuration
if [ -f config.env ]; then
    source config.env
fi

iteration=10
outfile=response.txt
init_prompt=init_prompt.json
topic=Aliens
type="science-fiction"

options=" \
    --iter $iteration \
    --r_file $outfile \
    --init_prompt $init_prompt \
    --topic $topic \
    --type $type \
    "

python main.py $options
