# python experiment_domain_rand_fullCurr_.py --config-name conf_4
# python experiment_domain_rand_fullCurr__.py --config-name conf


# #!/bin/bash

# # Run the first script with its specific config and capture its output
# SAVE_PATH=$(python experiment_domain_rand_fullCurr_.py --config-name conf_4_t | grep "CHECKPOINT_PATH" | cut -d: -f2)

# # Check if a path was captured
# if [ -z "$SAVE_PATH" ]; then
#     echo "Error: Could not determine save path from the first script."
#     exit 1
# fi

# echo "Captured save path: $SAVE_PATH"

# # Run the second script with its specific config, passing the captured path as a command-line override
# python experiment_domain_rand_fullCurr__.py --config-name conf_4_tt checkpoint_path="$SAVE_PATH"



#!/bin/bash
###### REP 1
# # Create a temporary file to store the script's output
# OUTPUT_FILE=$(mktemp)

# # Run the first script and redirect all its output to the temporary file
# # python experiment_domain_rand_fullCurr_.py --config-name conf_4_t > "$OUTPUT_FILE"
# python experiment_domain_rand.py --config-name conf_1 > "$OUTPUT_FILE"

# # Display the script's full output to the console for you to see all prints
# cat "$OUTPUT_FILE"

# # Extract the CHECKPOINT_PATH from the output file
# SAVE_PATH=$(grep "CHECKPOINT_PATH" "$OUTPUT_FILE" | cut -d: -f2)

# # Check if a path was captured
# if [ -z "$SAVE_PATH" ]; then
#     echo "Error: Could not determine save path from the first script."
#     # Clean up the temporary file
#     rm "$OUTPUT_FILE"
#     exit 1
# fi

# echo "Captured save path: $SAVE_PATH"

# # Clean up the temporary file
# rm "$OUTPUT_FILE"

# # Run the second script with the captured path
# # python experiment_domain_rand_fullCurr__.py --config-name conf_4_tt checkpoint_path="$SAVE_PATH"
# # python experiment_domain_rand_continue_wo_value_Curr.py --config-name conf_curr_9 checkpoint_path="$SAVE_PATH"
# python experiment_domain_rand_continue_Curr.py --config-name conf_curr_1 checkpoint_path="$SAVE_PATH"





# # # # # REP 

# # Create a temporary file to store the script's output
# OUTPUT_FILE=$(mktemp)

# # Run the first script and redirect all its output to the temporary file
# # python experiment_domain_rand_fullCurr_.py --config-name conf_4_t > "$OUTPUT_FILE"
# python experiment_domain_rand.py --config-name conf_2 > "$OUTPUT_FILE"

# # Display the script's full output to the console for you to see all prints
# cat "$OUTPUT_FILE"

# # Extract the CHECKPOINT_PATH from the output file
# SAVE_PATH=$(grep "CHECKPOINT_PATH" "$OUTPUT_FILE" | cut -d: -f2)

# # Check if a path was captured
# if [ -z "$SAVE_PATH" ]; then
#     echo "Error: Could not determine save path from the first script."
#     # Clean up the temporary file
#     rm "$OUTPUT_FILE"
#     exit 1
# fi

# echo "Captured save path: $SAVE_PATH"

# # Clean up the temporary file
# rm "$OUTPUT_FILE"

# # Run the second script with the captured path
# # python experiment_domain_rand_fullCurr__.py --config-name conf_4_tt checkpoint_path="$SAVE_PATH"
# # python experiment_domain_rand_continue_wo_value_Curr.py --config-name conf_curr_9 checkpoint_path="$SAVE_PATH"
# python experiment_domain_rand_continue_Curr.py --config-name conf_curr_2 checkpoint_path="$SAVE_PATH"

# python experiment_domain_rand_continue.py --config-name conf_curr_1

# # # # REP 

# # Create a temporary file to store the script's output
# OUTPUT_FILE=$(mktemp)

# # Run the first script and redirect all its output to the temporary file
# # python experiment_domain_rand_fullCurr_.py --config-name conf_4_t > "$OUTPUT_FILE"
# python experiment_domain_rand.py --config-name conf_2 > "$OUTPUT_FILE"

# # Display the script's full output to the console for you to see all prints
# cat "$OUTPUT_FILE"

# # Extract the CHECKPOINT_PATH from the output file
# SAVE_PATH=$(grep "CHECKPOINT_PATH" "$OUTPUT_FILE" | cut -d: -f2)

# # Check if a path was captured
# if [ -z "$SAVE_PATH" ]; then
#     echo "Error: Could not determine save path from the first script."
#     # Clean up the temporary file
#     rm "$OUTPUT_FILE"
#     exit 1
# fi

# echo "Captured save path: $SAVE_PATH"

# # Clean up the temporary file
# rm "$OUTPUT_FILE"

# # Run the second script with the captured path
# # python experiment_domain_rand_fullCurr__.py --config-name conf_4_tt checkpoint_path="$SAVE_PATH"
# # python experiment_domain_rand_continue_wo_value_Curr.py --config-name conf_curr_9 checkpoint_path="$SAVE_PATH"
# python experiment_domain_rand_continue_Curr.py --config-name conf_curr_2 checkpoint_path="$SAVE_PATH"


# # # #####################################
# # ### REP 
# # Create a temporary file to store the script's output
# OUTPUT_FILE=$(mktemp)

# # Run the first script and redirect all its output to the temporary file
# # python experiment_domain_rand_fullCurr_.py --config-name conf_4_t > "$OUTPUT_FILE"
# python experiment_domain_rand.py --config-name conf_3 > "$OUTPUT_FILE"

# # Display the script's full output to the console for you to see all prints
# cat "$OUTPUT_FILE"

# # Extract the CHECKPOINT_PATH from the output file
# SAVE_PATH=$(grep "CHECKPOINT_PATH" "$OUTPUT_FILE" | cut -d: -f2)

# # Check if a path was captured
# if [ -z "$SAVE_PATH" ]; then
#     echo "Error: Could not determine save path from the first script."
#     # Clean up the temporary file
#     rm "$OUTPUT_FILE"
#     exit 1
# fi

# echo "Captured save path: $SAVE_PATH"

# # Clean up the temporary file
# rm "$OUTPUT_FILE"

# # Run the second script with the captured path
# # python experiment_domain_rand_fullCurr__.py --config-name conf_4_tt checkpoint_path="$SAVE_PATH"
# # python experiment_domain_rand_continue_wo_value_Curr.py --config-name conf_curr_9 checkpoint_path="$SAVE_PATH"
# # python experiment_domain_rand_continue_Curr.py --config-name conf_curr_3 checkpoint_path="$SAVE_PATH"
# python experiment_domain_rand_continue_Curr.py --config-name conf_curr_newRew_3 checkpoint_path="$SAVE_PATH"






# # # ########################################
# ### REP
# # Create a temporary file to store the script's output
# OUTPUT_FILE=$(mktemp)

# # Run the first script and redirect all its output to the temporary file
# # python experiment_domain_rand_fullCurr_.py --config-name conf_4_t > "$OUTPUT_FILE"
# python experiment_domain_rand.py --config-name conf_4 > "$OUTPUT_FILE"

# # Display the script's full output to the console for you to see all prints
# cat "$OUTPUT_FILE"

# # Extract the CHECKPOINT_PATH from the output file
# SAVE_PATH=$(grep "CHECKPOINT_PATH" "$OUTPUT_FILE" | cut -d: -f2)

# # Check if a path was captured
# if [ -z "$SAVE_PATH" ]; then
#     echo "Error: Could not determine save path from the first script."
#     # Clean up the temporary file
#     rm "$OUTPUT_FILE"
#     exit 1
# fi

# echo "Captured save path: $SAVE_PATH"

# # Clean up the temporary file
# rm "$OUTPUT_FILE"

# # Run the second script with the captured path
# # python experiment_domain_rand_fullCurr__.py --config-name conf_4_tt checkpoint_path="$SAVE_PATH"
# # python experiment_domain_rand_continue_wo_value_Curr.py --config-name conf_curr_9 checkpoint_path="$SAVE_PATH"
# python experiment_domain_rand_continue_Curr.py --config-name conf_curr_newRew_5 checkpoint_path="$SAVE_PATH"


# python experiment_domain_rand_continue_Curr.py --config-name conf_curr_newRew_5 checkpoint_path="$SAVE_PATH"



# Create a temporary file to store the script's output
OUTPUT_FILE=$(mktemp)

python experiment_domain_rand_continue.py --config-name conf_curr_newRew_5_9

# Run the first script and redirect all its output to the temporary file
# python experiment_domain_rand_fullCurr_.py --config-name conf_4_t > "$OUTPUT_FILE"
python experiment_domain_rand_continue.py --config-name conf_4_1 > "$OUTPUT_FILE"

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
# python experiment_domain_rand_continue_wo_value_Curr.py --config-name conf_curr_9 checkpoint_path="$SAVE_PATH"
python experiment_domain_rand_continue_Curr.py --config-name conf_curr_newRew_5_9 checkpoint_path="$SAVE_PATH"

# # python experiment_domain_rand_continue_Curr.py --config-name conf_curr_newRew_6 checkpoint_path="$SAVE_PATH"

# python experiment_domain_rand_continue.py --config-name conf_curr_newRew_5_1 checkpoint_path="$SAVE_PATH"
# python experiment_domain_rand_continue.py --config-name conf_curr_newRew_5_2 checkpoint_path="$SAVE_PATH"
# python experiment_domain_rand_continue.py --config-name conf_curr_newRew_5_3 checkpoint_path="$SAVE_PATH"
# python experiment_domain_rand_continue.py --config-name conf_curr_newRew_5_4 checkpoint_path="$SAVE_PATH"