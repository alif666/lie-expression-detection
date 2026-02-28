# Lie Detection using Facial Micro-Expressions

## Architecture

Video → OpenFace 2.0 → Action Units (AUs)
→ Filter (confidence ≥ 0.7, success=1)
→ Feature engineering (AU_presence × AU_intensity)
→ MinMax scaling
→ ANN (ReLU + Adam)
→ Windowed inference (2s window)
→ Clip-level deception probability

## Hard Constraints
- 30 FPS recommended
- ≥ 480p resolution
- Face must be visible
- Use confidence gate

## Model Baseline
- Hidden layers: [128, 64, 32]
- ReLU
- Dropout 0.2
- Adam
- Batch 128
- Epochs ~200

## Inference Strategy
- No per-frame judgment
- Sliding window aggregation
- Output timeline + summary score

## Milestones
1. OpenFace integration
2. Dataset builder
3. ANN training
4. Windowed inference
5. FastAPI endpoint