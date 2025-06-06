#!/bin/bash

QUINES_CONFIG_FILE="quinerepo.json"

# Get the quine name from command line arguments
quine_name=$1

# Load configuration file
while IFS= read -r line; do
  if [[ $line =~ "name\":\"$quine_name\"" ]]; then
    # Extract parameters for this quine
    branch=$(jq '.[] | select(.name == '"$quine_name"') .branch' "$QUINES_CONFIG_FILE")
    parameters=$(jq '.[] | select(.name == '"$quine_name"') .parameters[]' "$QUINES_CONFIG_FILE")

    # Check out the relevant Git branch
    git checkout $branch

    # Invoke the quine with relevant data
    echo "Invoking Quine $quine_name with parameters: $parameters"

    # Run the quine executable (e.g., `python quine.py`)
    python quine.py "$parameters"
  fi
done < "$QUINES_CONFIG_FILE"