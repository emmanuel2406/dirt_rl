# Takes in file path to json file and field name, and outputs average value of that field
#!/bin/bash

# Check if the correct number of arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <json_file> <field_name>"
    exit 1
fi

json_file=$1
field_name=$2

# Check if the file exists
if [ ! -f "$json_file" ]; then
    echo "Error: File $json_file does not exist"
    exit 1
fi

# Check if the field name is valid
if [ -z "$field_name" ]; then
    echo "Error: Field name is required"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed. Please install jq to use this script."
    exit 1
fi

# Extract the field values and calculate average
# Handle both array and single value cases
# Check if field exists first
field_exists=$(jq -e "has(\"$field_name\")" "$json_file" 2>/dev/null)

if [ $? -ne 0 ] || [ "$field_exists" != "true" ]; then
    echo "Error: Field '$field_name' not found in JSON file"
    exit 1
fi

# Extract values, handling arrays and single values
values=$(jq -r ".$field_name | if type == \"array\" then .[] else . end" "$json_file" 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "Error: Failed to extract field '$field_name' from JSON file"
    exit 1
fi

# Filter out null values and empty strings
values=$(echo "$values" | grep -v "^null$" | grep -v "^$")

# Check if any values were found
if [ -z "$values" ]; then
    echo "Error: Field '$field_name' contains no valid numeric values"
    exit 1
fi

# Special case: for active_players_per_step return index of first zero value
if [ "$field_name" = "active_players_per_step" ]; then
    first_zero_index=$(echo "$values" | awk -v field="$field_name" '
    $1 == 0 {
        print NR - 1  # zero-based index
        found=1
        exit
    }
    END {
        if (!found) {
            if (NR == 0) {
                print "Error: Field \"" field "\" contains no values" > "/dev/stderr"
                exit 1
            }
            # If no zero found, return length (last index + 1)
            print NR
        }
    }')

    if [ $? -ne 0 ]; then
        exit 1
    fi

    echo "$first_zero_index"
    exit 0
fi

# Calculate average using awk (more portable than bc)
average=$(echo "$values" | awk '
{
    if ($1 != "" && $1 != "null") {
        sum += $1
        count++
    }
}
END {
    if (count == 0) {
        print "Error: No valid numeric values found" > "/dev/stderr"
        exit 1
    }
    printf "%.10f\n", sum / count
}')

if [ $? -ne 0 ]; then
    exit 1
fi

echo "$average"