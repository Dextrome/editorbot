import re
import csv
from pathlib import Path
import shutil

ROOT = Path("F:/editorbot")
PHASEC = ROOT / "models" / "hpo_phaseC"
OUTCSV = PHASEC / "phaseC_summary.csv"
BEST_COPY = ROOT / "models" / "best_hpo.pt"

pattern_best = re.compile(r"Eval deterministic \(best\): reward=([\-0-9\.]+) per-beat-keep_ratio=([0-9\.]+) edited_pct=([0-9\.]+)%")
pattern_final = re.compile(r"Final checkpoint: (.+)$")
pattern_saved = re.compile(r"Saved checkpoint to (.+)$")

rows = []

for d in sorted(PHASEC.glob("hpo_trial_*")):
    if not d.is_dir():
        continue
    log = d / "phaseC_train_stdout.txt"
    if not log.exists():
        continue
    best_reward = None
    best_keep = None
    best_edited = None
    final_ckp = None
    with log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern_best.search(line)
            if m:
                r = float(m.group(1))
                k = float(m.group(2))
                e = float(m.group(3))
                if (best_reward is None) or (r > best_reward):
                    best_reward = r
                    best_keep = k
                    best_edited = e
            m2 = pattern_final.search(line)
            if m2:
                final_ckp = m2.group(1).strip()
            else:
                m3 = pattern_saved.search(line)
                if m3:
                    path = m3.group(1).strip()
                    # prefer final.pt when listed
                    if path.endswith("final.pt"):
                        final_ckp = path
    rows.append({
        "trial": d.name,
        "dir": str(d),
        "best_reward": "" if best_reward is None else f"{best_reward:.4f}",
        "best_keep": "" if best_keep is None else f"{best_keep:.4f}",
        "best_edited_pct": "" if best_edited is None else f"{best_edited:.2f}",
        "final_checkpoint": "" if final_ckp is None else final_ckp,
    })

# write CSV
with OUTCSV.open("w", newline="", encoding="utf-8") as csvf:
    writer = csv.DictWriter(csvf, fieldnames=["trial","dir","best_reward","best_keep","best_edited_pct","final_checkpoint"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# choose winner
valid = [r for r in rows if r["best_reward"]]
if not valid:
    print("No valid trials found or no 'Eval deterministic (best)' lines.")
else:
    best = max(valid, key=lambda x: float(x["best_reward"]))
    print("Winner:", best["trial"], best["best_reward"], best["final_checkpoint"])
    if best["final_checkpoint"]:
        src = Path(best["final_checkpoint"])
        if src.exists():
            shutil.copy2(src, BEST_COPY)
            print(f"Copied {src} -> {BEST_COPY}")
        else:
            # try relative inside trial dir
            alt = Path(best["dir"]) / Path(best["final_checkpoint"]).name
            if alt.exists():
                shutil.copy2(alt, BEST_COPY)
                print(f"Copied {alt} -> {BEST_COPY}")
            else:
                print("Winner final checkpoint not found on disk:", best["final_checkpoint"])