#!/bin/bash
branch=$(git rev-parse --abbrev-ref HEAD)
commit=$(git log -1 --pretty=format:'%H')
date=$(date +%s)

echo "[GIT SNAPSHOT] Branch=$branch Commit=$commit Time=$date"
