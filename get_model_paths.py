"""
Get all Ollama model blob paths
"""
import os
import json
from pathlib import Path

ollama_models = Path.home() / '.ollama' / 'models'
manifests = ollama_models / 'manifests' / 'registry.ollama.ai' / 'library'

models = []

for model_dir in manifests.iterdir():
    if model_dir.is_dir():
        for tag_dir in model_dir.iterdir():
            if tag_dir.is_file():
                try:
                    manifest = json.loads(tag_dir.read_text())
                    model_name = f"{model_dir.name}:{tag_dir.name}"
                    for layer in manifest.get('layers', []):
                        mt = layer.get('mediaType', '')
                        if 'model' in mt:
                            digest = layer['digest'].replace('sha256:', '')
                            blob = ollama_models / 'blobs' / f'sha256-{digest}'
                            if blob.exists():
                                size_gb = blob.stat().st_size / (1024**3)
                                models.append({
                                    "name": model_name,
                                    "path": str(blob),
                                    "size_gb": round(size_gb, 2),
                                    "size_mb": round(blob.stat().st_size / (1024**2), 1)
                                })
                                print(f"{model_name}: {blob} ({size_gb:.2f} GB)")
                except Exception as e:
                    print(f"Error processing {model_dir.name}:{tag_dir.name}: {e}")

print(f"\nFound {len(models)} models")

# Save to file
with open("ollama_model_paths.json", 'w') as f:
    json.dump(models, f, indent=2)
