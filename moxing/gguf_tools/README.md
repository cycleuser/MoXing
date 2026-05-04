# GGUF Tools for MoXing

This module provides tools for working with GGUF files, specifically for handling Ollama's multimodal models.

## extract_mmproj.py

Extract multimodal projector (mmproj) from Ollama's unified GGUF files.

### Background

Ollama stores both language model and multimodal components (vision/audio) in a single GGUF file with tensor prefixes:
- `blk.*` - Language model tensors
- `v.*` - Vision model tensors
- `a.*` - Audio model tensors
- `mm.*` - Multimodal projector tensors

llama.cpp expects separate files:
- Language model: `model.gguf` (no v.* or a.* tensors)
- Multimodal projector: `mmproj.gguf` (only v.*, a.*, mm.* tensors + clip.* metadata)

### Usage

```bash
python moxing/gguf_tools/extract_mmproj.py <input.gguf> -o <mmproj.gguf>
```

### Example

```bash
# Extract mmproj from Ollama's Gemma4 model
python moxing/gguf_tools/extract_mmproj.py \
    /usr/share/ollama/.ollama/models/blobs/sha256-xxx \
    -o mmproj-gemma4.gguf

# Use with llama-server
llama-server -m model.gguf --mmproj mmproj-gemma4.gguf --port 8080
```

### Key Parameters Added

The tool automatically adds required metadata:
- `clip.has_vision_encoder` - Whether vision encoder is present
- `clip.has_audio_encoder` - Whether audio encoder is present
- `clip.vision.image_mean` - Image normalization mean (Gemma4: [0.5, 0.5, 0.5])
- `clip.vision.image_std` - Image normalization std (Gemma4: [0.5, 0.5, 0.5])
- `clip.vision.image_size` - Default image size
- `clip.vision.patch_size` - Patch size for ViT
- And other vision/audio parameters

### Supported Models

- Gemma 4 (Vision + optional Audio)
- Other Ollama multimodal models (with proper tensor prefixes)

## Implementation Details

### GGUF Tensor Shape Convention

GGUF stores tensor shapes in reverse order (column-major format). When extracting:
- Original shape from GGUFReader: [ne0, ne1, ne2, ...]
- Must be reversed when writing: [..., ne2, ne1, ne0]

### Normalization

For Gemma 4, image normalization uses:
- mean = 0.5
- std = 0.5
- Formula: `(pixel/255 - 0.5) / 0.5 = 2*pixel/255 - 1`

This normalizes pixel values from [0, 255] to [-1, 1].

## References

- Ollama source: https://github.com/ollama/ollama
- llama.cpp mtmd: https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd
- GGUF format: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
