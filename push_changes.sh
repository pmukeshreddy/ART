#!/bin/bash
cd "$(dirname "$0")"
git add -A
git commit -m "Add temperature parameter and better error logging"
git push origin sglang-megatron-integration
echo "Done! Now run 'git pull origin sglang-megatron-integration' on your server"
