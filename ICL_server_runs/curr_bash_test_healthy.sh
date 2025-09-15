#!/bin/bash

# Create a temporary file to store the script's output
OUTPUT_FILE=$(mktemp)

# Run the first script and redirect all its output to the temporary file
# python experiment_domain_rand_fullCurr_.py --config-name conf_4_t > "$OUTPUT_FILE"
python experiment.py --config-name conf_healthy_1 > "$OUTPUT_FILE"

# Display the script's full output to the console for you to see all prints
cat "$OUTPUT_FILE"

# Extract the CHECKPOINT_PATH from the output file
SAVE_PATH=$(grep "CHECKPOINT_PATH" "$OUTPUT_FILE" | cut -d: -f2)

# Check if a path was captured
if [ -z "$SAVE_PATH" ]; then
    echo "Error: Could not determine save path from the first script."
    # Clean up the temporary file
    rm "$OUTPUT_FILE"
    exit 1
fi

echo "Captured save path: $SAVE_PATH"

# Clean up the temporary file
rm "$OUTPUT_FILE"

# Run the second script with the captured path
# python experiment_domain_rand_fullCurr__.py --config-name conf_4_tt checkpoint_path="$SAVE_PATH"
python experiment_continue_Curr.py --config-name conf_healthy_2 checkpoint_path="$SAVE_PATH"

