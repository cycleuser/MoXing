# Changelog

All notable changes to this project will be documented in this file.

## [0.1.6] - 2025-03-20

### Added

#### GGUF Compression Support
- `moxing compress pack <file>` - Compress GGUF files with zstd
- `moxing compress unpack <file>` - Decompress compressed files
- `moxing compress cache --size/--clear` - Manage decompression cache
- `moxing compress split <file>` - Split GGUF into chunks
- `moxing compress merge <pattern> <output>` - Merge split chunks
- Transparent decompression when serving compressed files (.gguf.zst)
- Automatic caching of decompressed files in `~/.cache/moxing/decompressed/`

## [0.1.5] - 2025-03-20

### Added

#### Ollama Integration
- `moxing ollama list` - List all local Ollama models
- `moxing ollama list --select` - Interactive model selection
- `moxing ollama serve <model>` - Run Ollama models with llama.cpp
- `moxing ollama info <model>` - Show detailed model information
- `moxing ollama run <model>` - Quick run an Ollama model
- `moxing serve ollama:<model>` - Use Ollama models via main command
- Performance: Up to 50% faster than native Ollama runtime for compatible models

#### MLX Backend
- MLX backend for Apple Silicon (`-b mlx`)
- Automatic MLX detection for HuggingFace models
- Support for latest models (Gemma3, Qwen3.5) via MLX

#### GGUF Compatibility
- `moxing check <model>` - Check GGUF file compatibility
- Automatic compatibility detection when serving
- Automatic backend switching for incompatible models

#### Improved Error Handling
- Detailed error messages with llama.cpp output
- Helpful suggestions for fixing issues
- Better UTF-8 encoding handling

### Changed

- Improved `serve` command with automatic backend selection
- Better device detection and configuration display
- Fixed path resolution for platform-specific binaries (`darwin-arm64`)
- Fixed model download pattern matching

### Performance

- +50% speed improvement over Ollama for compatible models
- Latest llama.cpp (b8429+) with all optimizations

### Known Issues

- Not all Ollama models are compatible
- lfm2.5-thinking fails with missing tensor error

## [0.1.0] - 2025-01-XX

### Added

- Initial release
- Auto GPU detection (Vulkan, CUDA, ROCm, Metal)
- Model downloading from HuggingFace and ModelScope
- OpenAI-compatible API server
- Function calling support
- Pre-built llama.cpp binaries
- Benchmark and speed test commands