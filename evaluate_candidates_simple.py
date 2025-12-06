#!/usr/bin/env python3
"""
Simple interactive tool to evaluate candidates and collect human feedback.

This creates:
1. Audio files for each candidate edit
2. A JSON form for you to rate them
3. Saves ratings to feedback/preferences.json

Usage:
    python evaluate_candidates_simple.py
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


class FeedbackCollector:
    """Collect human preferences interactively"""
    
    def __init__(self, manifest_path: str = "eval_outputs/evaluation_manifest.json"):
        self.manifest_path = Path(manifest_path)
        self.feedback_dir = Path("feedback")
        self.feedback_dir.mkdir(exist_ok=True)
        
        # Load manifest
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)
        
        self.tasks = self.manifest["evaluation_tasks"]
        self.total_tasks = len(self.tasks)
        self.ratings: List[Dict] = []
    
    def create_csv_template(self) -> Path:
        """Create a CSV template for easy feedback entry"""
        csv_path = self.feedback_dir / "feedback_template.csv"
        
        # Build CSV header
        header = "song_id,candidate_a_id,candidate_b_id,a_keep_ratio,b_keep_ratio,preference,strength,reasoning\n"
        
        # Build rows (one for each pairwise comparison)
        rows = []
        for task in self.tasks:
            song_id = task["song_id"]
            candidates = {c["candidate_id"]: c for c in task["candidates"]}
            
            # Create comparisons
            for comp in task["pairwise_comparisons"]:
                cand_a_id = comp["candidate_a"]
                cand_b_id = comp["candidate_b"]
                
                a_ratio = candidates[cand_a_id]["keep_ratio"]
                b_ratio = candidates[cand_b_id]["keep_ratio"]
                
                row = f"{song_id},{cand_a_id},{cand_b_id},{a_ratio:.2f},{b_ratio:.2f},?,?,\n"
                rows.append(row)
        
        # Write CSV
        with open(csv_path, 'w') as f:
            f.write(header)
            f.writelines(rows)
        
        print(f"✅ CSV template created: {csv_path}")
        print(f"   Total rows to fill: {len(rows)}")
        print(f"\n📝 Instructions:")
        print(f"   1. Open {csv_path} in Excel or Google Sheets")
        print(f"   2. For each row, fill in:")
        print(f"      - preference: 'a', 'b', or 'tie'")
        print(f"      - strength: 1 (slight), 2 (moderate), 3 (strong)")
        print(f"      - reasoning: (optional) why you prefer this one")
        print(f"   3. Save the file")
        print(f"   4. Run: python evaluate_candidates_simple.py --convert-csv feedback/feedback_template.csv")
        
        return csv_path
    
    def create_json_template(self) -> Path:
        """Create a JSON template for direct editing"""
        json_path = self.feedback_dir / "feedback_template.json"
        
        template = {
            "instructions": {
                "preference": "Pick 'a', 'b', or 'tie' (which edit is better?)",
                "strength": "1=slight, 2=moderate, 3=strong (how much better?)",
                "reasoning": "Optional note explaining your choice"
            },
            "preferences": []
        }
        
        # Add template entries
        for task in self.tasks:
            song_id = task["song_id"]
            candidates = {c["candidate_id"]: c for c in task["candidates"]}
            
            # Create comparisons
            for comp in task["pairwise_comparisons"]:
                cand_a_id = comp["candidate_a"]
                cand_b_id = comp["candidate_b"]
                
                cand_a = candidates[cand_a_id]
                cand_b = candidates[cand_b_id]
                
                entry = {
                    "song_id": song_id,
                    "candidate_a_id": cand_a_id,
                    "candidate_b_id": cand_b_id,
                    "candidate_a_info": {
                        "temperature": cand_a["temperature"],
                        "keep_ratio": cand_a["keep_ratio"],
                        "duration_sec": cand_a["estimated_duration_sec"]
                    },
                    "candidate_b_info": {
                        "temperature": cand_b["temperature"],
                        "keep_ratio": cand_b["keep_ratio"],
                        "duration_sec": cand_b["estimated_duration_sec"]
                    },
                    "preference": None,  # ← FILL THIS IN (a/b/tie)
                    "strength": None,    # ← FILL THIS IN (1/2/3)
                    "reasoning": ""      # ← Optional explanation
                }
                template["preferences"].append(entry)
        
        with open(json_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"✅ JSON template created: {json_path}")
        print(f"   Total entries to fill: {len(template['preferences'])}")
        print(f"\n📝 How to fill it:")
        print(f"   1. Open {json_path} in VS Code or text editor")
        print(f"   2. For each preference, change:")
        print(f"      'preference': None  →  'preference': 'a'  (or 'b' or 'tie')")
        print(f"      'strength': None    →  'strength': 2")
        print(f"      'reasoning': \"\"     →  'reasoning': \"Tighter edit\"")
        print(f"   3. Save the file")
        print(f"   4. Run: python evaluate_candidates_simple.py --convert-json {json_path}")
        
        return json_path
    
    def convert_csv_to_preferences(self, csv_path: str) -> Path:
        """Convert CSV ratings to preferences.json format"""
        import csv
        
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"❌ File not found: {csv_path}")
            return None
        
        preferences = []
        skipped = 0
        
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pref = row.get("preference", "").strip().lower()
                strength_str = row.get("strength", "").strip()
                
                # Skip empty rows or rows without preference
                if not pref or pref == "?":
                    skipped += 1
                    continue
                
                # Validate
                if pref not in ["a", "b", "tie"]:
                    print(f"⚠️  Skipping invalid preference: {pref}")
                    skipped += 1
                    continue
                
                if strength_str not in ["1", "2", "3"]:
                    print(f"⚠️  Skipping invalid strength: {strength_str}")
                    skipped += 1
                    continue
                
                preferences.append({
                    "song_id": row.get("song_id", ""),
                    "candidate_a_id": row.get("candidate_a_id", ""),
                    "candidate_b_id": row.get("candidate_b_id", ""),
                    "preference": pref,
                    "strength": int(strength_str),
                    "reasoning": row.get("reasoning", "")
                })
        
        # Save preferences
        output_path = self.feedback_dir / "preferences.json"
        with open(output_path, 'w') as f:
            json.dump({"preferences": preferences}, f, indent=2)
        
        print(f"\n✅ Converted {len(preferences)} ratings to {output_path}")
        print(f"⏭️  Skipped {skipped} empty/invalid rows")
        print(f"\n📊 Preference distribution:")
        a_count = sum(1 for p in preferences if p["preference"] == "a")
        b_count = sum(1 for p in preferences if p["preference"] == "b")
        tie_count = sum(1 for p in preferences if p["preference"] == "tie")
        print(f"   Prefer A: {a_count}")
        print(f"   Prefer B: {b_count}")
        print(f"   Tie: {tie_count}")
        print(f"\n🚀 Next step: python train_from_feedback.py --feedback {output_path}")
        
        return output_path
    
    def convert_json_to_preferences(self, json_path: str) -> Path:
        """Convert JSON ratings to clean preferences.json format"""
        json_path = Path(json_path)
        
        if not json_path.exists():
            print(f"❌ File not found: {json_path}")
            return None
        
        with open(json_path) as f:
            data = json.load(f)
        
        preferences = []
        skipped = 0
        
        for entry in data.get("preferences", []):
            pref = entry.get("preference")
            strength = entry.get("strength")
            
            # Skip if not filled
            if pref is None or strength is None:
                skipped += 1
                continue
            
            # Validate
            if str(pref).lower() not in ["a", "b", "tie"]:
                print(f"⚠️  Skipping invalid preference: {pref}")
                skipped += 1
                continue
            
            if int(strength) not in [1, 2, 3]:
                print(f"⚠️  Skipping invalid strength: {strength}")
                skipped += 1
                continue
            
            preferences.append({
                "song_id": entry.get("song_id"),
                "candidate_a_id": entry.get("candidate_a_id"),
                "candidate_b_id": entry.get("candidate_b_id"),
                "preference": str(pref).lower(),
                "strength": int(strength),
                "reasoning": entry.get("reasoning", "")
            })
        
        # Save preferences
        output_path = self.feedback_dir / "preferences.json"
        with open(output_path, 'w') as f:
            json.dump({"preferences": preferences}, f, indent=2)
        
        print(f"\n✅ Converted {len(preferences)} ratings to {output_path}")
        print(f"⏭️  Skipped {skipped} empty/invalid entries")
        print(f"\n📊 Preference distribution:")
        a_count = sum(1 for p in preferences if p["preference"] == "a")
        b_count = sum(1 for p in preferences if p["preference"] == "b")
        tie_count = sum(1 for p in preferences if p["preference"] == "tie")
        print(f"   Prefer A: {a_count}")
        print(f"   Prefer B: {b_count}")
        print(f"   Tie: {tie_count}")
        print(f"\n🚀 Next step: python train_from_feedback.py --feedback {output_path}")
        
        return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect human feedback for edits")
    parser.add_argument("--format", choices=["csv", "json"], default="csv",
                        help="Format for feedback collection")
    parser.add_argument("--convert-csv", type=str, help="Convert CSV file to preferences.json")
    parser.add_argument("--convert-json", type=str, help="Convert JSON file to preferences.json")
    
    args = parser.parse_args()
    
    collector = FeedbackCollector()
    
    if args.convert_csv:
        collector.convert_csv_to_preferences(args.convert_csv)
    elif args.convert_json:
        collector.convert_json_to_preferences(args.convert_json)
    else:
        # Create template in preferred format
        if args.format == "csv":
            collector.create_csv_template()
        else:
            collector.create_json_template()


if __name__ == "__main__":
    main()
