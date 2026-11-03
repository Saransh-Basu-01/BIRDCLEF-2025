# BirdCLEF-2025: CNN vs Hybrid vs Pure Attention (AST)

Converted bird sounds into images and tested 3 architectures as per AST paper.

**Dataset:** BirdCLEF+ 2025 (28,564 recordings, 206 species, 12GB) from Kaggle. Middle Magdalena Valley, Colombia. Task: multi-taxa classification.

**Approach:**
- Fixed 5-sec clips @ 32kHz -> Log-Mel Spectrogram (128 mels, 50-14k Hz) -> treated as image
- Stratified 80/20 split

**3 Models Tested:**
1. **Pure CNN** - EfficientNet-B0, in_chans=1, 5M params
2. **Hybrid CNN+Attention** - EfficientNet backbone (num_classes=0) + TransformerEncoder (1 layer, 8 heads) + mean pooling + classifier
3. **Pure Attention (AST)** - MIT/ast-finetuned-audioset-10-10-0.4593, 87M params. Fixed pos-embed interpolation bug: original 1214 -> 362 for 5-sec input.

**Results (10% data, 2.2k files, 5 epochs):**
| Model | Val Accuracy | Notes |
|-------|--------------|-------|
| Pure CNN | 41.27% (234/567) | Baseline |
| Hybrid | **56.28% (963/1711)** | Best |
| AST | 41.91% (717/1711) | Needs more data/epochs |

**Key Learnings:**
- Audio -> Mel Spectrogram -> CNN works
- Hybrid: extract features with CNN, classify with attention - best trade-off on small data
- AST: freeze backbone epoch 1 (lr 1e-3 for head), unfreeze all lr 1e-5 after. Positional embedding must be interpolated for custom duration.

**Tech:** PyTorch, torchaudio, librosa, timm, HuggingFace Transformers, Kaggle P100

**Next:** BirdCLEF+ 2026 multi-taxa (birds + amphibians + mammals)
