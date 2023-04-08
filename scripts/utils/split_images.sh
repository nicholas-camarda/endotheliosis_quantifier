#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_file output_dir"
    exit 1
fi

input_file="$1"
output_dir="$2"
file_prefix=$(basename "$input_file" | cut -d'.' -f1)

if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
    echo "Making new directory: ${output_dir}"
fi

tiffsplit "$input_file" "${output_dir}/${file_prefix}_"

i=0
for file in "${output_dir}/${file_prefix}"*; do
    new_file="${output_dir}/${file_prefix}_${i}.tif"
    mv "$file" "$new_file"
    i=$((i+1))
done
