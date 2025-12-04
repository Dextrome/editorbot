# Add HybridV13Editor to train_edit_policy_v13.py

# Read the file
with open('train_edit_policy_v13.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the insertion point
marker = '''# =============================================================================
# HYBRID V11 EDITOR: V9 base + V11 refinement
# =============================================================================

class HybridV11Editor:'''

hybrid_v13_code = '''# =============================================================================
# HYBRID V13 EDITOR: V9 base + V13 stem-separated beat-level
# =============================================================================

class HybridV13Editor:
    \"\"\"
    Hybrid editor combining V9 quality + V13 stem-separated beat-level.
    \"\"\"
    
    def __init__(self, model_dir: str = \"./models\", v9_weight: float = 0.5, v13_weight: float = 0.5):
        self.model_dir = Path(model_dir)
        self.sr = 22050
        self.v9_weight = v9_weight
        self.v13_weight = v13_weight
        self.stem_separator = StemSeparator(sr=self.sr)
        self._load_v9_model()
        self._load_v13_model()
        logger.info(f\"Hybrid V13 Editor loaded (V9={v9_weight}, V13={v13_weight})\")
    
    def _load_v9_model(self):
        from train_edit_policy_v9 import DualHeadModel
        self.v9_feature_dim = int(np.load(self.model_dir / \"feature_dim_v9.npy\"))
        self.v9_reference_centroid = np.load(self.model_dir / \"reference_centroid_v9.npy\")
        self.v9_similarity_weight = float(np.load(self.model_dir / \"similarity_weight_v9.npy\"))
        self.v9_model = DualHeadModel(self.v9_feature_dim, len(self.v9_reference_centroid)).to(DEVICE)
        checkpoint = torch.load(self.model_dir / \"classifier_v9_best.pt\", weights_only=True)
        self.v9_model.load_state_dict(checkpoint)
        self.v9_model.eval()
        self.v9_ref_centroid_t = torch.FloatTensor(self.v9_reference_centroid).to(DEVICE)
        from train_edit_policy_v10_simple import SegmentFeatureExtractor
        self.v9_extractor = SegmentFeatureExtractor(self.sr)
        logger.info(\"Loaded V9 model\")
    
    def _load_v13_model(self):
        try:
            self.v13_feature_dim = int(np.load(self.model_dir / \"feature_dim_v13.npy\"))
            with open(self.model_dir / \"scaler_v13.pkl\", 'rb') as f:
                self.v13_scaler = pickle.load(f)
            context_beats = DEFAULT_CONTEXT_BEATS
            for model_file in [\"policy_v13_trajectory_best.pt\", \"policy_v13_best.pt\", \"policy_v13_final.pt\"]:
                path = self.model_dir / model_file
                if path.exists():
                    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
                    if 'context_beats' in checkpoint:
                        context_beats = checkpoint['context_beats']
                    self.v13_model = BeatLevelPolicy(self.v13_feature_dim, 64, 256, context_beats).to(DEVICE)
                    self.v13_model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f\"Loaded V13 model: {model_file}\")
                    break
            self.v13_model.eval()
            ref_path = self.model_dir / \"reference_centroid_v13.npy\"
            self.v13_reference_centroid = np.load(ref_path) if ref_path.exists() else np.zeros(self.v13_feature_dim)
            self.v13_loaded = True
        except Exception as e:
            logger.warning(f\"Could not load V13: {e}\")
            self.v13_loaded = False
    
    def compute_v9_scores_for_beats(self, beat_positions, audio):
        segment_samples = int(3.0 * self.sr)
        hop_samples = int(1.5 * self.sr)
        segments, segment_positions = [], []
        start = 0
        while start + segment_samples <= len(audio):
            segments.append(audio[start:start + segment_samples])
            segment_positions.append((start, start + segment_samples))
            start += hop_samples
        if not segments:
            return np.full(len(beat_positions), 0.5)
        features = np.array([self.v9_extractor.extract(seg) for seg in segments])
        from train_edit_policy_v10_simple import create_context_windows
        windowed = create_context_windows(features)
        windowed_tensor = torch.FloatTensor(windowed).to(DEVICE)
        with torch.no_grad():
            quality_logits, style_emb = self.v9_model(windowed_tensor)
            quality = torch.sigmoid(quality_logits).squeeze().cpu().numpy()
            ref_sim = torch.mm(style_emb, self.v9_ref_centroid_t.unsqueeze(1)).squeeze()
            ref_sim = ((ref_sim + 1) / 2).cpu().numpy()
        segment_scores = quality + self.v9_similarity_weight * ref_sim
        beat_scores = []
        for beat_start, beat_end in beat_positions:
            best_seg, best_overlap = 0, 0
            for seg_idx, (seg_start, seg_end) in enumerate(segment_positions):
                overlap = max(0, min(beat_end, seg_end) - max(beat_start, seg_start))
                if overlap > best_overlap:
                    best_overlap, best_seg = overlap, seg_idx
            beat_scores.append(segment_scores[best_seg])
        return np.array(beat_scores)
    
    def compute_v13_scores(self, features_tensor, beat_info):
        if not self.v13_loaded:
            return np.zeros(len(features_tensor))
        ref_scaled = self.v13_scaler.transform(self.v13_reference_centroid.reshape(1, -1))
        ref_tensor = torch.tensor(ref_scaled[0], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            style_emb = self.v13_model.compute_style_embedding(ref_tensor)
            scores, beats_since_keep = [], 0
            for i in range(len(features_tensor)):
                logit, _ = self.v13_model.forward_single(features_tensor, i, style_emb, beat_info, beats_since_keep)
                score = torch.sigmoid(logit).item()
                scores.append(score)
                beats_since_keep = 0 if score > 0.5 else beats_since_keep + 1
        return np.array(scores)
    
    def process_track(self, input_path, output_path, keep_ratio=0.35):
        input_path_obj = Path(input_path)
        audio = load_audio_fast(input_path_obj, self.sr)
        duration = len(audio) / self.sr
        logger.info(\"Separating stems...\")
        melodic_audio = self.stem_separator.get_guitar_focused_audio(audio, input_path_obj)
        logger.info(\"Detecting beats...\")
        beat_info = detect_beats(audio, self.sr)
        n_beats = len(beat_info.beat_times) - 1
        if n_beats <= 0:
            raise ValueError(\"Beat detection failed\")
        logger.info(f\"Detected {n_beats} beats @ {beat_info.tempo:.1f} BPM\")
        beat_samples = (beat_info.beat_times * self.sr).astype(int)
        melodic_beat_list, beat_positions = [], []
        for i in range(n_beats):
            start, end = beat_samples[i], beat_samples[i+1] if i+1 < len(beat_samples) else len(audio)
            if start < len(melodic_audio) and end > start:
                melodic_beat_list.append(melodic_audio[start:min(end, len(melodic_audio))])
                beat_positions.append((start, end))
        logger.info(\"Extracting features...\")
        features = np.array([extract_features_fast(b, self.sr) for b in melodic_beat_list])
        features_scaled = self.v13_scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE)
        logger.info(\"Computing V9 scores...\")
        v9_scores = self.compute_v9_scores_for_beats(beat_positions, audio)
        v9_min, v9_max = v9_scores.min(), v9_scores.max()
        v9_normalized = (v9_scores - v9_min) / (v9_max - v9_min) if v9_max > v9_min else np.full(len(v9_scores), 0.5)
        logger.info(\"Computing V13 scores...\")
        v13_scores = self.compute_v13_scores(features_tensor, beat_info)
        hybrid_scores = self.v9_weight * v9_normalized + self.v13_weight * v13_scores
        n_keep = max(1, int(n_beats * keep_ratio))
        threshold = np.sort(hybrid_scores)[::-1][min(n_keep - 1, len(hybrid_scores) - 1)]
        keep_mask = hybrid_scores >= threshold
        kept_regions, in_region, region_start = [], False, 0
        for i, keep in enumerate(keep_mask):
            if i < len(beat_positions):
                if keep and not in_region:
                    region_start, in_region = beat_positions[i][0], True
                elif not keep and in_region:
                    kept_regions.append((region_start, beat_positions[i-1][1]))
                    in_region = False
        if in_region and beat_positions:
            kept_regions.append((region_start, beat_positions[-1][1]))
        avg_beat = (beat_info.beat_times[1] - beat_info.beat_times[0]) if len(beat_info.beat_times) > 1 else 0.5
        min_gap = int(2 * avg_beat * self.sr)
        merged_regions = []
        for start, end in kept_regions:
            if merged_regions and start - merged_regions[-1][1] < min_gap:
                merged_regions[-1] = (merged_regions[-1][0], end)
            else:
                merged_regions.append((start, end))
        output_segments = [audio[s:e] for s, e in merged_regions]
        output_audio = np.concatenate(output_segments) if output_segments else audio[:int(30*self.sr)]
        sf.write(output_path, output_audio, self.sr)
        return {
            'input_duration': duration,
            'output_duration': len(output_audio) / self.sr,
            'n_beats': n_beats,
            'tempo': beat_info.tempo,
            'n_regions': len(merged_regions),
            'keep_ratio_actual': len(output_audio) / len(audio),
            'v9_score_stats': {'min': float(v9_normalized.min()), 'max': float(v9_normalized.max()), 'mean': float(v9_normalized.mean())},
            'v13_score_stats': {'min': float(v13_scores.min()), 'max': float(v13_scores.max()), 'mean': float(v13_scores.mean())},
            'hybrid_score_stats': {'min': float(hybrid_scores.min()), 'max': float(hybrid_scores.max()), 'mean': float(hybrid_scores.mean())}
        }


''' + marker

if marker in content:
    content = content.replace(marker, hybrid_v13_code)
    with open('train_edit_policy_v13.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Added HybridV13Editor successfully!')
else:
    print('Marker not found!')
