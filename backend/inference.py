from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision import transforms

from optimization.pca_eigen import EigenPCA
from optimization.swan import SwanFeatureSelector, load_swan_result
from preprocessing.image_pipeline import preprocess_image
from training.softmax_head import SoftmaxClassifier
from training.transformer import BackboneSpec, create_feature_extractor, load_backbone_for_features


class InferenceEngine:
    def __init__(self, artifacts_dir: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[1]
        self.artifacts_dir = artifacts_dir or (root / "logs" / "artifacts")

        metadata = json.loads((self.artifacts_dir / "metadata.json").read_text(encoding="utf-8"))
        self.image_size = int(metadata["image_size"])
        self.backbone_name = str(metadata["backbone_name"])

        class_map = json.loads((self.artifacts_dir / "class_to_idx.json").read_text(encoding="utf-8"))
        self.class_names = [name for name, _ in sorted(class_map.items(), key=lambda item: item[1])]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        spec = BackboneSpec(model_name=self.backbone_name, pretrained=False)
        self.backbone = create_feature_extractor(spec)

        backbone_state = torch.load(self.artifacts_dir / "backbone_best.pth", map_location="cpu")
        load_backbone_for_features(self.backbone, backbone_state)
        self.backbone.to(self.device)
        self.backbone.eval()

        self.pca = EigenPCA.load(self.artifacts_dir / "pca_state.npz")
        self.swan = load_swan_result(self.artifacts_dir / "swan_result.npz")

        self.classifier = SoftmaxClassifier(
            in_features=int(len(self.swan.indices)),
            num_classes=len(self.class_names),
        )
        self.classifier.load_state_dict(torch.load(self.artifacts_dir / "softmax_head_best.pth", map_location="cpu"))
        self.classifier.to(self.device)
        self.classifier.eval()

        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Invalid image data")

        preprocessed = preprocess_image(image_bgr)
        rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

        tensor = self.tensor_transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.backbone(tensor).cpu().numpy()

        pca_features = self.pca.transform(features)
        optimized = SwanFeatureSelector.transform(pca_features, self.swan.indices, self.swan.weights)

        x = torch.from_numpy(optimized.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.classifier(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        probabilities = {self.class_names[idx]: float(probs[idx]) for idx in range(len(self.class_names))}

        return {
            "predicted_class": self.class_names[top_idx],
            "confidence": float(probs[top_idx]),
            "probabilities": probabilities,
            "disease_info": self._disease_info(self.class_names[top_idx]),
        }

    @staticmethod
    def _disease_info(label: str) -> str:
        readable = label.replace("_", " ").title()
        return (
            f"{readable} detected by model screening. This output is assistive and requires dermatologist confirmation."
        )
