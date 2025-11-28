import pickle

data = pickle.load(open('data/training/style_zappa.pkl', 'rb'))
sp = data['style_profile']

print('=== Zappa Style Profile ===')
print(f'\nSection Timing:')
for k, v in sp.section_timing.items():
    print(f'  {k}: mean={v["mean"]:.1f}s std={v["std"]:.1f}s')

print(f'\nTransition Patterns: {len(sp.transition_patterns)}')
if sp.transition_patterns:
    for tp in sp.transition_patterns[:5]:
        print(f'  {tp.from_label} -> {tp.to_label}: energy_ratio={tp.energy_ratio:.2f}')

print(f'\nContinuity:')
print(f'  Chroma: {sp.typical_chroma_continuity:.3f}')
print(f'  Timbre: {sp.typical_timbre_continuity:.3f}')
