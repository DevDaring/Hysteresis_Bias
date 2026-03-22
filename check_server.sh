#!/bin/bash
echo "=== SERVER RESULT COUNTS ==="
RDIR=/root/Hysteresis_Bias/results
for d in $RDIR/phase*; do
    name=$(basename "$d")
    count=$(find "$d" -type f -name "*.json" | wc -l)
    echo "$name $count"
done
echo "---"
echo "total_json $(find $RDIR -type f -name '*.json' | wc -l)"
echo "figures $(find $RDIR/figures -type f 2>/dev/null | wc -l)"
echo "tables $(find $RDIR/tables -type f 2>/dev/null | wc -l)"
echo "---SIZES---"
for f in $(find $RDIR -maxdepth 2 -name "*.json" -type f | sort); do
    echo "$(stat -c '%s' "$f") $(echo "$f" | sed "s|$RDIR/||")"
done
echo "---QUAL---"
ls -la $RDIR/phase7_qualitative/ 2>/dev/null || echo "NO_QUAL_DIR"
echo "---FIGS---"
ls -la $RDIR/figures/ 2>/dev/null | head -30
echo "---TABS---"
ls -la $RDIR/tables/ 2>/dev/null | head -20
