import os
import json
import argparse
from typing import Dict, List
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for testing
    torch = None
from PIL import Image
import torchvision.transforms as T

# Make sure CLAT is importable when running from the repo root.  The
# actual CLAT code lives under ``clat/src`` so we append that path at
# runtime instead of requiring installation as a package.
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'clat', 'src'))

try:
    # CLAT is a PyTorch Lightning module.  Import lazily so that the script
    # still works even if the optional dependency is missing.  In that case
    # random features will be generated as a fallback which keeps the rest
    # of the pipeline functional for testing purposes.
    from clat.model import CLAT  # type: ignore
    CLAT_AVAILABLE = True
except Exception:  # pragma: no cover - the exact exception type depends on env
    CLAT_AVAILABLE = False


def build_model(device: torch.device) -> torch.nn.Module:
    """Instantiate a minimal CLAT model for feature extraction.

    The pretrained weights are not bundled with the repository, therefore the
    model may be randomly initialised.  For real use one should load the
    official checkpoints.  When CLAT is not available the function returns
    ``None`` and the caller should generate random features instead.
    """

    if not CLAT_AVAILABLE or torch is None:
        return None

    # CLAT requires lists of disease and lesion names.  For demonstration
    # purposes we provide the four common DR lesions.
    lesion_names = ["EX", "HE", "MA", "SE"]
    disease_names = ["No DR", "DR"]
    model = CLAT(disease_names=disease_names, lesion_names=lesion_names,
                 pretrained=False)
    model.eval()
    model.to(device)
    return model


def extract_feature(model: torch.nn.Module, image_path: str,
                     device) -> 'torch.Tensor | np.ndarray':
    """Extract lesion tokens from ``image_path`` using ``model``.

    If ``model`` is ``None`` random features of shape ``(4, 384)`` are
    returned.  This keeps the script light-weight for unit testing while the
    surrounding code remains identical to the real pipeline.
    """

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    if torch is not None:
        tensor = transform(img).unsqueeze(0).to(device)
    else:
        tensor = None

    if model is None or torch is None:
        return np.random.rand(4, 384)

    with torch.no_grad():
        output = model(tensor)
        # ``output.lesion_tokens`` has shape ``(1, num_lesions, embed_dim)``.
        feats = output.lesion_tokens.squeeze(0).cpu().numpy()
    return feats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CLAT lesion features for use in mm-cot")
    parser.add_argument('--data_root', type=str, default='images',
                        help='directory containing raw images')
    parser.add_argument('--output_dir', type=str, default='vision_features',
                        help='where to store the extracted features')
    args = parser.parse_args()

    device = torch.device('cuda' if (torch and torch.cuda.is_available()) else 'cpu') if torch is not None else None
    model = build_model(device)

    # Collect features and build an index mapping from question id to tensor
    # location.  The mapping allows the dataset loader to fetch features by
    # the original question ids later on.
    image_files = sorted(os.listdir(args.data_root))
    features: List = []
    name_map: Dict[str, int] = {}
    for idx, img_name in enumerate(image_files, start=1):
        path = os.path.join(args.data_root, img_name)
        feats = extract_feature(model, path, device)
        features.append(feats)
        stem, _ = os.path.splitext(img_name)
        name_map[stem] = idx

    os.makedirs(args.output_dir, exist_ok=True)
    if torch is not None:
        stacked = torch.stack([torch.tensor(f) if not isinstance(f, torch.Tensor) else f for f in features])
        torch.save(stacked, os.path.join(args.output_dir, 'clat.pth'))
    else:
        stacked = np.stack(features)
        np.save(os.path.join(args.output_dir, 'clat.npy'), stacked)
    with open(os.path.join(args.output_dir, 'name_map.json'), 'w') as f:
        json.dump(name_map, f)
    print(f"Saved features: {stacked.shape}")


if __name__ == '__main__':
    main()
