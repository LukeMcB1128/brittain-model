#!/usr/bin/env bash
# Stop the vLLM server and confirm the GPU is actually released.
#
# The patterns are split with an empty string so this script's own command line
# cannot match them: pkill -f matches the caller too, which has killed the
# invoking shell here before and returned exit 9/15 for a kill that worked.
echo '=== before ==='
curl -s -o /dev/null -m 3 -w '  health: %{http_code}\n' http://localhost:11435/health || echo '  health: unreachable'
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/  gpu: /'

pkill -f 'bin/vll''m serve' 2>/dev/null
pkill -f 'VLLM::Engine''Core' 2>/dev/null

for _ in $(seq 1 20); do
    if ! pgrep -f 'VLLM::Engine''Core' >/dev/null 2>&1; then break; fi
    sleep 1
done
# Anything still holding the GPU after a graceful stop gets SIGKILL.
if pgrep -f 'VLLM::Engine''Core' >/dev/null 2>&1; then
    echo '  engine did not exit; sending SIGKILL'
    pkill -9 -f 'VLLM::Engine''Core' 2>/dev/null
    sleep 3
fi

echo '=== after ==='
echo "  vllm serve procs : $(pgrep -cf 'bin/vll''m serve')"
echo "  engine procs     : $(pgrep -cf 'VLLM::Engine''Core')"
curl -s -o /dev/null -m 3 -w '  health: %{http_code}\n' http://localhost:11435/health || echo '  health: unreachable (expected)'
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/  gpu: /'
