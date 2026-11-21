# predict.py — 운영진 Colab CPU 재현 및 검증용 추론 스크립트
# 실행 예시: python predict.py --input <이미지 폴더> --output pred.csv
import argparse
import glob
import os

import torch
from PIL import Image
from torchvision import transforms

from model import IMG_SIZE, NORM_MEAN, NORM_STD, build_model

PARAM_LIMIT = 20_000_000
PROB_CLIP_MIN = 0.002
PROB_CLIP_MAX = 0.998


def load_images(input_dir):
    exts = (".jpg", ".jpeg", ".png")
    return sorted(
        f for f in glob.glob(os.path.join(input_dir, "*"))
        if f.lower().endswith(exts)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="테스트 이미지 폴더")
    parser.add_argument("--output", required=True, help="출력 CSV 경로")
    parser.add_argument("--weights", default="model.pt", help="모델 가중치 파일")
    args = parser.parse_args()

    device = torch.device("cpu")

    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.weights)
    model = build_model()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {n_params:,} (제한 {PARAM_LIMIT:,})")
    assert n_params <= PARAM_LIMIT, "파라미터 20M 초과! 실격 대상입니다."

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    files = load_images(args.input)
    assert len(files) > 0, f"{args.input} 에서 이미지를 찾지 못했습니다."

    rows = ["id,prob"]
    with torch.no_grad():
        for i in range(0, len(files), 32):
            batch_files = files[i:i + 32]
            batch = torch.stack([
                transform(Image.open(f).convert("RGB")) for f in batch_files
            ]).to(device)
            batch_flip = torch.flip(batch, dims=[3])

            logits_orig = model(batch)
            logits_flip = model(batch_flip)
            probs = 0.5 * (torch.sigmoid(logits_orig) + torch.sigmoid(logits_flip)).squeeze(1)
            probs = torch.clamp(probs, min=PROB_CLIP_MIN, max=PROB_CLIP_MAX)

            for f, p in zip(batch_files, probs):
                rows.append(f"{os.path.splitext(os.path.basename(f))[0]},{p.item():.6f}")

    with open(args.output, "w") as fp:
        fp.write("\n".join(rows) + "\n")
    print(f"{args.output} 저장 완료 ({len(files)}장)")


if __name__ == "__main__":
    main()
