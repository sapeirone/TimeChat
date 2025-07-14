#!/bin/bash

# Set the base directory (you can change '.' to your target directory)
BASE_DIR="./ego4d_hoi_videos_trimmed/"

# Process all video files (you can adjust the extensions as needed)
find "$BASE_DIR" -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.avi" -o -iname "*.mkv" \) | while read -r file; do
    # Get file directory and name without extension
    dir=$(dirname "$file")
    filename=$(basename "$file")
    name="${filename%.*}"

    # Define output path
    output="${dir}/${name}_6fps.mp4"

    # Convert using ffmpeg
    ffmpeg -y -i "$file" -map 0:v -r 6 "$output"
done
