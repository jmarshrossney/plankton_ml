#!/bin/bash
# Copilot generated script to render diagrams as SVG

# Set the directory path
DIR="./docs/diagrams/"

# Loop through each subdirectory
for sub_dir in "$DIR"*/; do
    # Loop through each dot file in the subdirectory
    for dotfile in "$sub_dir"*.dot; do
        # Get the base name without extension
        base_name=$(basename "$dotfile" .dot)
        
        # Render the dot file to SVG
        dot -Tsvg "$dotfile" -o "$sub_dir$base_name.svg"
        
        # Print a success message
        echo "Rendered $dotfile to $sub_dir$base_name.svg"
    done
done


