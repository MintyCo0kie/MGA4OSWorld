#!/bin/bash

# Define the base directory for our test
BASE_DIR="/tmp/test_files"

# Function to check if a file is gzipped
is_gzipped() {
    if file "$1" | grep -q 'gzip compressed data'; then
        return 0 # True, file is gzipped
    else
        return 1 # False, file is not gzipped
    fi
}

# Check for compressed old files and uncompressed new files
OLD_FILES_COMPRESSED=true
NEW_FILES_UNCOMPRESSED=true

for file in "$BASE_DIR/old_files"/*; do
    if ! is_gzipped "$file" ; then # Directly use is_gzipped to check
        OLD_FILES_COMPRESSED=false
        break
    fi
done

for file in "$BASE_DIR/new_files"/*; do
    if is_gzipped "$file"; then # Check if a new file is gzipped
        NEW_FILES_UNCOMPRESSED=false
        break
    fi
done

# Evaluate the results
if $OLD_FILES_COMPRESSED && $NEW_FILES_UNCOMPRESSED; then
    echo "Success: The task was completed correctly."
else
    echo "Failure: The task was not completed correctly."
    if ! $OLD_FILES_COMPRESSED; then
        echo "Reason: Not all old files were compressed."
    fi
    if ! $NEW_FILES_UNCOMPRESSED; then
        echo "Reason: Some new files were compressed."
    fi
fi