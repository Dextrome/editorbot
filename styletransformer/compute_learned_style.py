#!/usr/bin/env python
"""Compute the learned style embedding from existing model and training data."""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from trainer_lightning import SongDataset, StyleEncoderNet

def main():
    print("Loading model...")
    model = StyleEncoderNet(embedding_dim=256, hidden_dim=512)
    model.load_state_dict(torch.load('models/style_encoder.pt', weights_only=True))
    model.eval()

    print("Loading dataset...")
    dataset = SongDataset('F:/editorbot/data/reference', cache_features=True, augment=False)
    print(f"Found {len(dataset)} files")

    # Compute embeddings
    embeddings = []
    with torch.no_grad():
        for i in range(min(100, len(dataset))):
            try:
                batch = dataset[i]
                if batch.get('path', '') == '':
                    continue
                time_feat = batch['time_features'].unsqueeze(0)
                chroma_feat = batch['chroma_features'].unsqueeze(0)
                global_feat = batch['global_features'].unsqueeze(0)
                emb = model(time_feat, chroma_feat, global_feat)
                embeddings.append(emb.numpy())
                if (i+1) % 20 == 0:
                    print(f'Processed {i+1} samples...')
            except Exception as e:
                print(f'Skip {i}: {e}')

    # Average and save
    if embeddings:
        learned_style = np.mean(np.vstack(embeddings), axis=0)
        np.save('models/learned_style.npy', learned_style)
        print(f'Saved models/learned_style.npy ({len(embeddings)} samples averaged)')
    else:
        print("No valid embeddings computed!")

if __name__ == "__main__":
    main()
