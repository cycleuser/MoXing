#!/usr/bin/env python3
"""
Split Ollama's unified GGUF file into separate language model and mmproj files.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, cast

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "llama.cpp" / "gguf-py"))

from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter, Keys

MODEL_CONFIGS = {
    "gemma4": {
        "projector_type": "gemma4v",
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "prefixes": {"vision": "v.", "audio": "a.", "mm": "mm."},
    },
    "gemma3": {
        "projector_type": "gemma3",
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "prefixes": {"vision": "v.", "audio": None, "mm": "mm."},
    },
    "qwen2vl": {
        "projector_type": "qwen2vl",
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "prefixes": {"vision": "v.", "audio": None, "mm": "mm."},
    },
    "qwen25vl": {
        "projector_type": "qwen25vl",
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "prefixes": {"vision": "v.", "audio": None, "mm": "mm."},
    },
}


def detect_model_type(reader):
    """Detect model type from GGUF metadata."""
    for field in reader.fields.values():
        name = field.name
        if "architecture" in name.lower():
            arch = field.contents()
            if isinstance(arch, str):
                arch = arch.lower()
                for model_type in MODEL_CONFIGS:
                    if model_type in arch:
                        return model_type
    return "gemma4"


def is_mmproj_tensor(name, config):
    """Check if tensor is mmproj tensor."""
    prefixes = cast(Dict[str, Any], config["prefixes"])
    vision_prefix = prefixes["vision"]
    audio_prefix = prefixes["audio"]
    mm_prefix = prefixes["mm"]

    return (
        (vision_prefix and name.startswith(vision_prefix))
        or (audio_prefix and name.startswith(audio_prefix))
        or (mm_prefix and name.startswith(mm_prefix))
    )


def split_ollama_gguf(input_path: str, output_dir: str, model_type: Optional[str] = None):
    """
    Split Ollama GGUF file into language model and multimodal projector.

    Args:
        input_path: Path to the input GGUF file
        output_dir: Directory to write output files to
        model_type: Model type override (e.g., 'llava', 'minicpmv')

    Returns:
        bool: True if split was successful, False otherwise
    """
    from pathlib import Path

    reader = GGUFReader(input_path)

    if not model_type:
        model_type = detect_model_type(reader)

    config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["gemma4"])

    language_tensors = []
    mmproj_tensors = []

    for tensor in reader.tensors:
        if is_mmproj_tensor(tensor.name, config):
            mmproj_tensors.append(tensor)
        else:
            language_tensors.append(tensor)

    if not mmproj_tensors:
        print("No multimodal tensors found!")
        return False

    input_file = Path(input_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    base_name = input_file.stem
    language_path = output_dir_path / f"{base_name}-language.gguf"
    mmproj_path = output_dir_path / f"{base_name}-mmproj.gguf"

    prefixes = cast(Dict[str, Any], config["prefixes"])
    has_vision = (
        any(t.name.startswith(prefixes["vision"]) for t in mmproj_tensors)
        if prefixes["vision"]
        else False
    )
    has_audio = (
        any(t.name.startswith(prefixes["audio"]) for t in mmproj_tensors)
        if prefixes["audio"]
        else False
    )

    print(f"Model: {model_type}")
    print(f"Language tensors: {len(language_tensors)}")
    print(f"Vision: {has_vision}, Audio: {has_audio}")
    print(f"Multimodal tensors: {len(mmproj_tensors)}")
    print(f"Output directory: {output_dir}")

    print(f"\nWriting language model to: {language_path}")
    writer = GGUFWriter(str(language_path), "unknown")

    for field in reader.fields.values():
        name = field.name
        if not (
            "vision" in name.lower()
            or "audio" in name.lower()
            or name.startswith("clip.")
            or name.startswith("mmproj.")
        ):
            writer.add_key_value(name, field.contents(), field.types[0])

    for tensor in language_tensors:
        raw_dtype = GGMLQuantizationType(tensor.tensor_type) if tensor.tensor_type != 0 else None
        shape = tuple(tensor.data.shape) if raw_dtype else tuple(tensor.shape)
        writer.add_tensor(tensor.name, tensor.data, raw_shape=shape, raw_dtype=raw_dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"✓ Language model written: {language_path}")

    print(f"\nWriting mmproj to: {mmproj_path}")
    writer = GGUFWriter(str(mmproj_path), "unknown")

    writer.add_key_value(Keys.Clip.HAS_VISION_ENCODER, has_vision, GGUFValueType.BOOL)
    writer.add_key_value(Keys.Clip.HAS_AUDIO_ENCODER, has_audio, GGUFValueType.BOOL)

    if has_vision:
        writer.add_key_value(
            Keys.Clip.PROJECTOR_TYPE, config["projector_type"], GGUFValueType.STRING
        )
        writer.add_key_value("clip.vision.image_mean", config["image_mean"], GGUFValueType.ARRAY)
        writer.add_key_value("clip.vision.image_std", config["image_std"], GGUFValueType.ARRAY)
        writer.add_key_value(
            "clip.vision.image_size", 0, GGUFValueType.UINT32
        )  # 0 for dynamic image size

        vision_metadata = {}
        for field in reader.fields.values():
            name = field.name
            if ".vision." in name:
                key_parts = name.split(".vision.")
                if len(key_parts) > 1:
                    value = field.contents()
                    vision_metadata[key_parts[-1]] = value

        key_mapping = {
            "embedding_length": "embedding_length",
            "attention.head_count": "attention.head_count",
            "feed_forward_length": "feed_forward_length",
            "block_count": "block_count",
            "attention.layer_norm_epsilon": "attention.layer_norm_epsilon",
            "patch_size": "patch_size",
            "num_channels": "n_channels",
        }

        language_embd = None
        for field in reader.fields.values():
            if "embedding_length" in field.name and "vision" not in field.name:
                language_embd = field.contents()
                break

        for orig_key, mapped_key in key_mapping.items():
            if orig_key in vision_metadata:
                value = vision_metadata[orig_key]
                value_type = (
                    GGUFValueType.UINT32 if isinstance(value, int) else GGUFValueType.FLOAT32
                )
                writer.add_key_value(f"clip.vision.{mapped_key}", value, value_type)

        if language_embd:
            writer.add_key_value("clip.vision.projection_dim", language_embd, GGUFValueType.UINT32)

        if "projector.scale_factor" in vision_metadata:
            writer.add_key_value(
                "clip.vision.projector.scale_factor",
                vision_metadata["projector.scale_factor"],
                GGUFValueType.UINT32,
            )

    for tensor in mmproj_tensors:
        data = tensor.data
        shape = tuple(tensor.shape)
        raw_dtype = GGMLQuantizationType(tensor.tensor_type) if tensor.tensor_type != 0 else None
        if len(shape) > 1:
            shape = shape[::-1]
        writer.add_tensor(tensor.name, data, raw_shape=shape, raw_dtype=raw_dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"✓ mmproj written: {mmproj_path}")
    print("\nDone! Use llama-server with:")
    print(f"  --model {language_path}")
    print(f"  --mmproj {mmproj_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Split Ollama GGUF into language model and mmproj files"
    )
    parser.add_argument("input", help="Input Ollama GGUF file path")
    parser.add_argument("-o", "--output-dir", default="./", help="Output directory")
    parser.add_argument(
        "-t",
        "--model-type",
        choices=MODEL_CONFIGS.keys(),
        help="Model type (auto-detected if not specified)",
    )

    args = parser.parse_args()

    split_ollama_gguf(args.input, args.output_dir, args.model_type)


if __name__ == "__main__":
    main()
