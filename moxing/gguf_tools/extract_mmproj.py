#!/usr/bin/env python3
"""
Extract mmproj from Ollama's unified GGUF file.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'llama.cpp' / 'gguf-py'))

from gguf import GGUFReader, GGUFWriter, Keys, GGMLQuantizationType, GGUFValueType


MODEL_CONFIGS = {
    'gemma4': {
        'projector_type': 'gemma4v',
        'image_mean': [0.5, 0.5, 0.5],
        'image_std': [0.5, 0.5, 0.5],
        'prefixes': {'vision': 'v.', 'audio': 'a.', 'mm': 'mm.'}
    },
    'gemma3': {
        'projector_type': 'gemma3',
        'image_mean': [0.5, 0.5, 0.5],
        'image_std': [0.5, 0.5, 0.5],
        'prefixes': {'vision': 'v.', 'audio': None, 'mm': 'mm.'}
    },
    'qwen2vl': {
        'projector_type': 'qwen2vl',
        'image_mean': [0.48145466, 0.4578275, 0.40821073],
        'image_std': [0.26862954, 0.26130258, 0.27577711],
        'prefixes': {'vision': 'v.', 'audio': None, 'mm': 'mm.'}
    },
    'qwen25vl': {
        'projector_type': 'qwen25vl',
        'image_mean': [0.48145466, 0.4578275, 0.40821073],
        'image_std': [0.26862954, 0.26130258, 0.27577711],
        'prefixes': {'vision': 'v.', 'audio': None, 'mm': 'mm.'}
    },
}


def detect_model_type(reader):
    """Detect model type from GGUF metadata."""
    for field in reader.fields.values():
        name = field.name
        if 'architecture' in name.lower():
            arch = field.contents()
            if isinstance(arch, str):
                arch = arch.lower()
                for model_type in MODEL_CONFIGS.keys():
                    if model_type in arch:
                        return model_type
    return 'gemma4'


def extract_mmproj(input_path: str, output_path: str, model_type: str = None):
    """Extract mmproj tensors from Ollama GGUF file."""
    reader = GGUFReader(input_path)
    
    if not model_type:
        model_type = detect_model_type(reader)
    
    config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS['gemma4'])
    
    vision_prefix = config['prefixes']['vision']
    audio_prefix = config['prefixes']['audio']
    mm_prefix = config['prefixes']['mm']
    
    mmproj_tensors = []
    for tensor in reader.tensors:
        name = tensor.name
        if (vision_prefix and name.startswith(vision_prefix)) or \
           (audio_prefix and name.startswith(audio_prefix)) or \
           (mm_prefix and name.startswith(mm_prefix)):
            mmproj_tensors.append(tensor)
    
    if not mmproj_tensors:
        print("No multimodal tensors found!")
        return False
    
    has_vision = any(t.name.startswith(vision_prefix) for t in mmproj_tensors) if vision_prefix else False
    has_audio = any(t.name.startswith(audio_prefix) for t in mmproj_tensors) if audio_prefix else False
    
    print(f"Model: {model_type}")
    print(f"Vision: {has_vision}, Audio: {has_audio}")
    print(f"Multimodal tensors: {len(mmproj_tensors)}")
    
    vision_params = {}
    for field in reader.fields.values():
        name = field.name
        if '.vision.' in name:
            key = name.split('.vision.')[-1]
            vision_params[key] = field.contents()
    
    writer = GGUFWriter(output_path, 'unknown')
    
    writer.add_key_value(Keys.Clip.HAS_VISION_ENCODER, has_vision, GGUFValueType.BOOL)
    writer.add_key_value(Keys.Clip.HAS_AUDIO_ENCODER, has_audio, GGUFValueType.BOOL)
    
    if has_vision:
        if 'attention.head_count' in vision_params:
            writer.add_key_value(Keys.ClipVision.Attention.HEAD_COUNT, vision_params['attention.head_count'], GGUFValueType.UINT32)
        if 'block_count' in vision_params:
            writer.add_key_value(Keys.ClipVision.BLOCK_COUNT, vision_params['block_count'], GGUFValueType.UINT32)
        if 'embedding_length' in vision_params:
            writer.add_key_value(Keys.ClipVision.EMBEDDING_LENGTH, vision_params['embedding_length'], GGUFValueType.UINT32)
            writer.add_key_value(Keys.ClipVision.PROJECTION_DIM, vision_params['embedding_length'], GGUFValueType.UINT32)
        if 'feed_forward_length' in vision_params:
            writer.add_key_value(Keys.ClipVision.FEED_FORWARD_LENGTH, vision_params['feed_forward_length'], GGUFValueType.UINT32)
        if 'patch_size' in vision_params:
            writer.add_key_value(Keys.ClipVision.PATCH_SIZE, vision_params['patch_size'], GGUFValueType.UINT32)
        
        writer.add_key_value(Keys.ClipVision.IMAGE_SIZE, 224, GGUFValueType.UINT32)
        writer.add_key_value(Keys.ClipVision.IMAGE_MEAN, config['image_mean'], GGUFValueType.ARRAY, GGUFValueType.FLOAT32)
        writer.add_key_value(Keys.ClipVision.IMAGE_STD, config['image_std'], GGUFValueType.ARRAY, GGUFValueType.FLOAT32)
        writer.add_key_value(Keys.Clip.PROJECTOR_TYPE, config['projector_type'], GGUFValueType.STRING)
    
    for tensor in mmproj_tensors:
        data = tensor.data
        shape = list(tensor.shape)[::-1]
        dtype = GGMLQuantizationType(tensor.tensor_type)
        writer.add_tensor(tensor.name, data, shape, dtype)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"✓ Extracted {len(mmproj_tensors)} tensors to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Extract mmproj from Ollama GGUF')
    parser.add_argument('input', help='Input GGUF file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('-m', '--model-type', help='Model type (auto-detect if not specified)')
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        return 1
    
    output = args.output or f"mmproj-{Path(args.input).stem}.gguf"
    return 0 if extract_mmproj(args.input, output, args.model_type) else 1


if __name__ == '__main__':
    sys.exit(main())