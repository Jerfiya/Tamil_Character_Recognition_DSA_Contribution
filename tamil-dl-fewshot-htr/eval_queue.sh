#!/usr/bin/env bash
# Evaluates every FINISHED run (history.json has stop_reason) whose test/DONE.json lacks a checkpoint.
# One evaluation at a time (memory). Each run is attempted at most once by this queue.
# Ends when no run is left unfinished or unevaluated.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
tried=" "
while true; do
  todo=$($PY - <<'PYEOF'
import json, pathlib
for d in sorted(pathlib.Path("runs").iterdir()):
    try: h = json.loads((d / "history.json").read_text(encoding="utf-8"))
    except Exception: h = {}
    fin = bool(h.get("stop_reason"))
    done = json.loads((d / "test/DONE.json").read_text()) if (d / "test/DONE.json").exists() else {}
    cks = [c for c in ("best_val_loss.pt", "best_val_acc.pt", "last.pt") if (d / "checkpoints" / c).exists()]
    print(("READY " if fin else "WAIT ") + d.name if (not fin or any(c not in done for c in cks)) else "", end="\n" if (not fin or any(c not in done for c in cks)) else "")
PYEOF
)
  ran=0
  for r in $(echo "$todo" | awk '$1=="READY"{print $2}'); do
    case "$tried" in *" $r "*) continue;; esac
    tried="$tried$r "
    PYTHONIOENCODING=utf-8 $PY src/evaluate.py "runs/$r" > "logs/eval_$r.log" 2>&1
    echo "$(date +%H:%M) eval $r exit=$?"; ran=1
  done
  waiting=$(echo "$todo" | grep -c "^WAIT ")
  if [ $ran = 0 ] && [ "$waiting" = 0 ]; then echo "EVALQUEUE DONE $(date +%H:%M)"; break; fi
  [ $ran = 0 ] && sleep 60
done
