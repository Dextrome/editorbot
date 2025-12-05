"""Test script to analyze paired training data."""
from rl_editor import PairedAudioDataset, get_default_config
import warnings
warnings.filterwarnings('ignore')

config = get_default_config()
dataset = PairedAudioDataset('./training_data', config, cache_dir='./feature_cache/rl_editor')

print(f"Dataset size: {len(dataset)} total")
print(f"  - Paired (raw/edited): {len(dataset.pairs)}")
print(f"  - Reference tracks: {len(dataset.reference_files)}")
print()

# Show pairs
print("Paired data:")
for raw, edit in dataset.pairs[:5]:  # First 5
    print(f"  {raw.name} <-> {edit.name}")
if len(dataset.pairs) > 5:
    print(f"  ... and {len(dataset.pairs) - 5} more")
print()

# Show reference tracks
if dataset.reference_files:
    print("Reference tracks:")
    for ref in dataset.reference_files[:5]:
        print(f"  {ref.name}")
    if len(dataset.reference_files) > 5:
        print(f"  ... and {len(dataset.reference_files) - 5} more")
print()

# Load one pair to verify
print("Loading first pair...")
item = dataset[0]
print(f"  Pair ID: {item['pair_id']}")
print(f"  Raw: {item['raw']['duration'].item():.0f}s, {len(item['raw']['beats'])} beats")
print(f"  Edited: {item['edited']['duration'].item():.0f}s, {len(item['edited']['beats'])} beats")
print(f"  Keep ratio: {item['edit_labels'].mean().item()*100:.1f}%")
print(f"  Is reference: {item['is_reference']}")
