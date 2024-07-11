#!/bin/bash
# Copilot generated script to render diagrams as SVG

# Set the directory path
DIR="./diagrams/"
SITE="_site/"

# Loop through each subdirectory
for sub_dir in "$DIR"*/; do
    # Loop through each dot file in the subdirectory
    for dotfile in "$sub_dir"*.dot; do
        # Get the base name without extension
        base_name=$(basename "$dotfile" .dot)
        dir_path=${sub_dir//diagrams/_site\/diagrams}
        mkdir -p $dir_path
        output="$dir_path$base_name.svg"

        # Render the dot file to SVG
        dot -Tsvg "$dotfile" -o $output
        
        # Print a success message
        echo "Rendered $dotfile to $output"
    done
done


