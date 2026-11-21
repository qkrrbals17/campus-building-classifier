# model.py — 6공학관(0) vs IT관(1) 분류 앙상블 모델 정의
import torch
import torch.nn as nn
from torchvision import models

IMG_SIZE = 224
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


def make_mobilenet_large():
    m = models.mobilenet_v3_large(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    return m


def make_mobilenet_small():
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    return m


class CampusEnsemble(nn.Module):
    """MobileNetV3-Large (2-Fold) + MobileNetV3-Small (2-Fold) 앙상블 모델.

    - 총 파라미터 수: 약 11.44M (20M 제한 안전하게 충족)
    - 추론 시 외부 다운로드 전혀 없음 (weights=None 초기화 후 model.pt 로드)
    - Temperature Scaling 버퍼(temperature)를 내장하여 보정된 로짓을 반환
    """

    def __init__(self):
        super().__init__()
        self.large_models = nn.ModuleList([
            make_mobilenet_large(),
            make_mobilenet_large(),
        ])
        self.small_models = nn.ModuleList([
            make_mobilenet_small(),
            make_mobilenet_small(),
        ])
        # 모델별 가중치 (Large 0.32 * 2 + Small 0.18 * 2 = 1.0)
        self.register_buffer("ensemble_weights", torch.tensor([0.32, 0.32, 0.18, 0.18], dtype=torch.float32))
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

    def forward_single_view(self, x):
        logits = []
        for m in self.large_models:
            logits.append(m(x))
        for m in self.small_models:
            logits.append(m(x))
        # (B, 4)
        stacked = torch.cat(logits, dim=1)
        w = self.ensemble_weights / self.ensemble_weights.sum()
        weighted_logit = (stacked * w.unsqueeze(0)).sum(dim=1, keepdim=True)
        return weighted_logit / self.temperature.clamp(min=0.1)

    def forward(self, x):
        return self.forward_single_view(x)


def build_model():
    """predict.py가 호출하는 진입점 함수."""
    return CampusEnsemble()
