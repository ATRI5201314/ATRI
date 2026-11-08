#!/bin/bash
set -u
url="$1"; start="$2"; end="$3"; out="$4"; chunk="${5:-8000000}"
rm -f "$out" "$out.part"
while [ "$start" -le "$end" ]; do
  e=$((start + chunk - 1)); if [ "$e" -gt "$end" ]; then e="$end"; fi
  want=$((e - start + 1)); ok=0
  for try in 1 2 3 4 5 6 7 8; do
    curl -L --http1.1 -s --max-time 150 -r "$start-$e" -o "$out.part" "$url" 2>/dev/null
    got=$(stat -c%s "$out.part" 2>/dev/null || echo 0)
    if [ "$got" = "$want" ]; then ok=1; break; fi
    sleep 2
  done
  if [ "$ok" != 1 ]; then echo "FAILED at $start ($got/$want)" >&2; exit 1; fi
  cat "$out.part" >> "$out"; start=$((e + 1)); echo "  $((start-1)) done" >&2
done
rm -f "$out.part"; ls -la "$out" >&2
