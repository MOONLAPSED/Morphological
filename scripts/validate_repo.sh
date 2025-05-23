#!/bin/bash
set -e

FILE=$1

echo "Validating repository: $FILE"

jq -c '.repository[]' "$FILE" | while read -r instance; do
  id=$(echo "$instance" | jq -r '.id')
  branch=$(echo "$instance" | jq -r '.branch')
  perms=$(echo "$instance" | jq -r '.permissions | join(",")')
  state=$(echo "$instance" | jq -r '.state')

  if [[ "$id" != "null" && "$branch" != "null" && "$perms" != "null" && "$state" != "null" ]]; then
    echo "[INFO] Valid instance: ID=$id Branch=$branch Perms=$perms State=$state"
  else
    echo "[ERROR] Invalid instance data: $instance"
  fi
done
