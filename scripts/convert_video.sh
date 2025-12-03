#!/bin/bash

# Script to convert a folder of PNG images into a video
# Automatically calculates fps to normalize video duration to 5 seconds (or longer if fps would exceed 20)
# Usage: ./convert_video.sh <folder_suffix> [output_name]

set -e  # Exit on error

# Check if folder suffix is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder_suffix> [output_name]"
    echo "  folder_suffix: Folder name in visual_output (e.g., food_metabolism_1)"
    echo "  output_name: Output video name (default: output.mp4)"
    echo ""
    echo "Note: Video duration is normalized to 5 seconds (or longer if FPS would exceed 20). FPS is calculated automatically and capped at 20."
    exit 1
fi

FOLDER_SUFFIX="$1"
OUTPUT_NAME="${2:-output.mp4}"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Hardcode the relative path to visual_output
VISUAL_OUTPUT_DIR="$SCRIPT_DIR/../dirt/examples/nomnom/visual_output"

# Construct the full folder path
FOLDER_PATH="$VISUAL_OUTPUT_DIR/$FOLDER_SUFFIX"

# Check if folder exists
if [ ! -d "$FOLDER_PATH" ]; then
    echo "Error: Folder '$FOLDER_PATH' does not exist"
    exit 1
fi

# Check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed or not in PATH"
    exit 1
fi

# Get absolute path of folder
FOLDER_PATH=$(realpath "$FOLDER_PATH")

# Count PNG files
PNG_COUNT=$(find "$FOLDER_PATH" -maxdepth 1 -type f -name "*.png" | wc -l)

if [ "$PNG_COUNT" -eq 0 ]; then
    echo "Error: No PNG files found in '$FOLDER_PATH'"
    exit 1
fi

# Calculate fps to normalize video duration to 5 seconds
# fps = number of frames / 5 seconds
# If fps would exceed 20, cap it at 20 and extend duration accordingly
MAX_FPS=20
DESIRED_DURATION=5
DESIRED_FPS=$(awk "BEGIN {printf \"%.2f\", $PNG_COUNT / $DESIRED_DURATION}")

# Compare using awk for floating point comparison
FPS_CAPPED=$(awk "BEGIN {if ($DESIRED_FPS > $MAX_FPS) print 1; else print 0}")

if [ "$FPS_CAPPED" -eq 1 ]; then
    # Cap FPS at 20 and calculate actual duration needed
    FPS=$MAX_FPS
    VIDEO_DURATION=$(awk "BEGIN {printf \"%.2f\", $PNG_COUNT / $FPS}")
    echo "Found $PNG_COUNT PNG files in '$FOLDER_PATH'"
    echo "Desired fps ($DESIRED_FPS) exceeds maximum ($MAX_FPS). Capping fps to ${FPS}."
    echo "Video duration will be ${VIDEO_DURATION} seconds to accommodate all frames."
else
    # Use calculated FPS and 5 second duration
    FPS=$DESIRED_FPS
    VIDEO_DURATION=$DESIRED_DURATION
    echo "Found $PNG_COUNT PNG files in '$FOLDER_PATH'"
    echo "Calculated fps: ${FPS} (for ${VIDEO_DURATION}-second video)"
fi

echo "Converting to video..."

# Create output path
OUTPUT_PATH="$FOLDER_PATH/$OUTPUT_NAME"

# Use ffmpeg to convert PNG sequence to video
# Try pattern-based input first (for frame_XXXXXX.png naming convention)
# Using H.264 codec with reasonable compression settings for good quality/size balance

cd "$FOLDER_PATH"

# Try frame_%06d.png pattern first (most common format)
if ls frame_*.png 1> /dev/null 2>&1; then
    echo "Using frame sequence pattern..."
    ffmpeg -y \
        -framerate "$FPS" \
        -i "frame_%06d.png" \
        -r "$FPS" \
        -c:v libx264 \
        -pix_fmt yuv420p \
        -crf 23 \
        -preset medium \
        "$OUTPUT_NAME"
else
    # Fallback: use glob pattern for arbitrary PNG naming
    echo "Using glob pattern for PNG frames..."
    ffmpeg -y \
        -framerate "$FPS" \
        -pattern_type glob \
        -i "*.png" \
        -r "$FPS" \
        -c:v libx264 \
        -pix_fmt yuv420p \
        -crf 23 \
        -preset medium \
        "$OUTPUT_NAME"
fi

if [ $? -eq 0 ]; then
    echo "Success! Video saved to: $OUTPUT_PATH"
    # Display file size
    if [ -f "$OUTPUT_PATH" ]; then
        FILE_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
        echo "Video size: $FILE_SIZE"
    fi
else
    echo "Error: Video conversion failed"
    exit 1
fi
