#!/bin/bash
# Monitor SLURM jobs and collect results
# Run via: watch -n 1800 bash scripts/monitor_jobs.sh
# Or cron: */30 * * * * bash scripts/monitor_jobs.sh >> logs/monitor.log

TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
echo "=== $TIMESTAMP ==="

# Queue status
RUNNING=$(squeue -u tcarr23 --noheader 2>/dev/null | grep "RUNNING" | wc -l)
PENDING=$(squeue -u tcarr23 --noheader 2>/dev/null | grep "PENDING" | wc -l)
echo "Queue: $RUNNING running, $PENDING pending"

# MIRAGE seeds
echo ""
echo "MIRAGE Seeds (target: AR>76.8, RI<17.3):"
for i in $(seq 1 14); do
  F="output/beta_improve/mirage_full_seed${i}/cross_eval_metrics.json"
  if [ -f "$F" ]; then
    python3 -c "
import json
d=json.load(open('$F'))
ar=d['cross_sgn_ar']*100
ri=d['cross_sgn_ri']*100
beat='*** BEATS TARGET ***' if ar>76.8 and ri<17.3 else ''
print(f'  seed{$i}: AR={ar:.1f}% RI={ri:.1f}% {beat}')
"
  fi
done

# Dissertation experiments
echo ""
echo "Dissertation experiments:"
count=0
for f in $(find output/dissertation_beta02 -name "cross_eval_metrics.json" 2>/dev/null | sort); do
  name=$(echo $f | sed 's|output/dissertation_beta02/||;s|/cross_eval_metrics.json||')
  python3 -c "import json; d=json.load(open('$f')); print('  {:<28s} AR={:5.1f}%  RI={:5.1f}%'.format('$name', d.get('cross_sgn_ar',0)*100, d.get('cross_sgn_ri',0)*100))"
  count=$((count+1))
done
echo "  ($count / 22 complete)"
echo ""
