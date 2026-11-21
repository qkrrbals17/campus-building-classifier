# train_local.py — 로컬 학습, 확률 보정(Temperature Scaling), 제출 파일 및 패키지 생성·검증 파이프라인
import copy
import glob
import os
import random
import shutil
import ssl
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from model import IMG_SIZE, NORM_MEAN, NORM_STD, build_model

# macOS 로컬 환경 SSL 인증서 이슈 방지 (사전학습 가중치 캐시 로드용)
ssl._create_default_https_context = ssl._create_unverified_context
# Apple M5 CPU 컨텍스트 스위칭 병목 방지를 위해 연산 스레드 최적화
torch.set_num_threads(1)

SEED = 42
PARAM_LIMIT = 20_000_000
PROB_CLIP_MIN = 0.002
PROB_CLIP_MAX = 0.998


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def extract_stem_features(stem, ds_orig, ds_flip):
    stem.eval()
    loader_orig = DataLoader(ds_orig, batch_size=64, shuffle=False)
    loader_flip = DataLoader(ds_flip, batch_size=64, shuffle=False)
    out_orig, out_flip = [], []
    with torch.no_grad():
        for x, _ in loader_orig:
            out_orig.append(stem(x).half())
        for x, _ in loader_flip:
            out_flip.append(stem(x).half())
    return torch.cat(out_orig, dim=0), torch.cat(out_flip, dim=0)


def train_fold_top(
    base_model,
    split_idx,
    H_orig,
    H_flip,
    y,
    tr_idx,
    va_idx,
    epochs=8,
    batch_size=64,
    lr_backbone=3e-4,
    lr_head=1e-3,
    label_smoothing=0.02,
    mixup_prob=0.3,
    seed=42,
):
    seed_everything(seed)
    m = copy.deepcopy(base_model)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, 1)

    # Stem 가중치 고정
    for p in m.features[:split_idx].parameters():
        p.requires_grad = False

    top = nn.Sequential(m.features[split_idx:], m.avgpool, nn.Flatten(), m.classifier)
    opt = torch.optim.AdamW([
        {"params": m.features[split_idx:].parameters(), "lr": lr_backbone, "weight_decay": 1e-2},
        {"params": m.classifier.parameters(), "lr": lr_head, "weight_decay": 1e-2},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = nn.BCEWithLogitsLoss()

    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    best_loss = float("inf")
    best_state = None
    best_va_logits_orig = None
    best_va_logits_flip = None

    for ep in range(1, epochs + 1):
        t0 = time.time()
        top.train()
        perm = torch.randperm(len(tr_idx))
        for i in range(0, len(tr_idx), batch_size):
            idx = tr_idx[perm[i:i + batch_size]]
            use_flip = torch.rand(len(idx)) < 0.5
            h_batch = torch.where(use_flip.view(-1, 1, 1, 1), H_flip[idx], H_orig[idx]).float()
            targets = y_t[idx] * (1.0 - label_smoothing) + 0.5 * label_smoothing

            # Feature-space Mixup regularization
            if random.random() < mixup_prob and len(idx) > 1:
                lam = float(np.random.beta(0.4, 0.4))
                perm_mix = torch.randperm(len(idx))
                h_batch = lam * h_batch + (1.0 - lam) * h_batch[perm_mix]
                targets = lam * targets + (1.0 - lam) * targets[perm_mix]

            # Slight feature noise for robustness
            if random.random() < 0.3:
                h_batch = h_batch + torch.randn_like(h_batch) * 0.02

            logits = top(h_batch)
            loss = crit(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()

        sched.step()
        top.eval()
        with torch.no_grad():
            va_lo = top(H_orig[va_idx].float()).squeeze(1).numpy()
            va_lf = top(H_flip[va_idx].float()).squeeze(1).numpy()
            va_probs = 0.5 * (1.0 / (1.0 + np.exp(-va_lo)) + 1.0 / (1.0 + np.exp(-va_lf)))
            va_loss = log_loss(y[va_idx], va_probs)
            va_acc = accuracy_score(y[va_idx], va_probs >= 0.5)

        dt = time.time() - t0
        mark = ""
        if va_loss < best_loss:
            best_loss = va_loss
            best_state = copy.deepcopy(m.state_dict())
            best_va_logits_orig = va_lo.copy()
            best_va_logits_flip = va_lf.copy()
            mark = " *best*"
        print(f"    Epoch {ep:2d}/{epochs} ({dt:.1f}s) | Val TTA LogLoss: {va_loss:.4f} | Acc: {va_acc:.4f}{mark}")

    m.load_state_dict(best_state)
    return m, best_loss, best_va_logits_orig, best_va_logits_flip


def main():
    t_start = time.time()
    seed_everything(SEED)
    print("=== [Step 1] 데이터셋 로드 및 전처리 구성 ===")
    tf_orig = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    tf_flip = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    ds_orig = datasets.ImageFolder("./train", tf_orig)
    ds_flip = datasets.ImageFolder("./train", tf_flip)
    y = np.array([label for _, label in ds_orig.samples], dtype=np.float32)
    print(f"클래스 매핑: {ds_orig.class_to_idx}, 전체 학습 이미지: {len(y)}장")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    # Fold 0과 Fold 1 (서로 겹치지 않는 40% 검증셋 = 1,163장)을 사용해 앙상블 학습 및 보정
    fold_indices = [0, 1]

    # 1. MobileNetV3-Large (split_idx = 13)
    print("\n=== [Step 2] MobileNetV3-Large 특징 추출 및 2-Fold 미세조정 ===")
    t0 = time.time()
    base_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    H_large_orig, H_large_flip = extract_stem_features(base_large.features[:13], ds_orig, ds_flip)
    print(f"  Large Stem 특징 추출 완료 ({time.time() - t0:.1f}s) | shape={tuple(H_large_orig.shape)}")

    large_models = []
    large_va_orig, large_va_flip = {}, {}
    for f_idx in fold_indices:
        tr_idx, va_idx = splits[f_idx]
        print(f"  [MobileNetV3-Large Fold {f_idx}] (train={len(tr_idx)}, val={len(va_idx)})")
        m_l, best_l, lo, lf = train_fold_top(
            base_large,
            split_idx=13,
            H_orig=H_large_orig,
            H_flip=H_large_flip,
            y=y,
            tr_idx=tr_idx,
            va_idx=va_idx,
            epochs=7,
            batch_size=64,
            lr_backbone=3e-4,
            lr_head=1e-3,
            label_smoothing=0.02,
            mixup_prob=0.3,
            seed=SEED + f_idx * 10,
        )
        large_models.append(m_l)
        large_va_orig[f_idx] = lo
        large_va_flip[f_idx] = lf

    del H_large_orig, H_large_flip

    # 2. MobileNetV3-Small (split_idx = 9)
    print("\n=== [Step 3] MobileNetV3-Small 특징 추출 및 2-Fold 미세조정 ===")
    t0 = time.time()
    base_small = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    H_small_orig, H_small_flip = extract_stem_features(base_small.features[:9], ds_orig, ds_flip)
    print(f"  Small Stem 특징 추출 완료 ({time.time() - t0:.1f}s) | shape={tuple(H_small_orig.shape)}")

    small_models = []
    small_va_orig, small_va_flip = {}, {}
    for f_idx in fold_indices:
        tr_idx, va_idx = splits[f_idx]
        print(f"  [MobileNetV3-Small Fold {f_idx}] (train={len(tr_idx)}, val={len(va_idx)})")
        m_s, best_s, lo, lf = train_fold_top(
            base_small,
            split_idx=9,
            H_orig=H_small_orig,
            H_flip=H_small_flip,
            y=y,
            tr_idx=tr_idx,
            va_idx=va_idx,
            epochs=9,
            batch_size=64,
            lr_backbone=3e-4,
            lr_head=1e-3,
            label_smoothing=0.02,
            mixup_prob=0.3,
            seed=SEED + 100 + f_idx * 10,
        )
        small_models.append(m_s)
        small_va_orig[f_idx] = lo
        small_va_flip[f_idx] = lf

    del H_small_orig, H_small_flip

    # 3. OOF 기반 가중치 및 Temperature Calibration
    print("\n=== [Step 4] Out-of-Fold 검증 및 Temperature Calibration ===")
    oof_y = np.concatenate([y[splits[0][1]], y[splits[1][1]]])
    oof_l_orig = np.concatenate([large_va_orig[0], large_va_orig[1]])
    oof_l_flip = np.concatenate([large_va_flip[0], large_va_flip[1]])
    oof_s_orig = np.concatenate([small_va_orig[0], small_va_orig[1]])
    oof_s_flip = np.concatenate([small_va_flip[0], small_va_flip[1]])

    def sigmoid_np(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    p_l = 0.5 * (sigmoid_np(oof_l_orig) + sigmoid_np(oof_l_flip))
    p_s = 0.5 * (sigmoid_np(oof_s_orig) + sigmoid_np(oof_s_flip))
    print(f"  단일 아키텍처 OOF (1,163장) — Large LogLoss: {log_loss(oof_y, p_l):.4f} (Acc: {accuracy_score(oof_y, p_l>=0.5):.4f})")
    print(f"  단일 아키텍처 OOF (1,163장) — Small LogLoss: {log_loss(oof_y, p_s):.4f} (Acc: {accuracy_score(oof_y, p_s>=0.5):.4f})")

    # 최적 Large:Small 비중 및 Temperature 탐색
    def calib_obj(params):
        w_l, temp = params
        w_s = 1.0 - w_l
        lo = (w_l * oof_l_orig + w_s * oof_s_orig) / temp
        lf = (w_l * oof_l_flip + w_s * oof_s_flip) / temp
        p = 0.5 * (sigmoid_np(lo) + sigmoid_np(lf))
        p = np.clip(p, PROB_CLIP_MIN, PROB_CLIP_MAX)
        return log_loss(oof_y, p)

    res = minimize(calib_obj, [0.65, 1.0], bounds=[(0.4, 0.85), (0.5, 2.5)], method="Nelder-Mead")
    best_w_l, best_temp = float(res.x[0]), float(res.x[1])
    best_w_s = 1.0 - best_w_l

    lo_cal = (best_w_l * oof_l_orig + best_w_s * oof_s_orig) / best_temp
    lf_cal = (best_w_l * oof_l_flip + best_w_s * oof_s_flip) / best_temp
    p_cal = np.clip(0.5 * (sigmoid_np(lo_cal) + sigmoid_np(lf_cal)), PROB_CLIP_MIN, PROB_CLIP_MAX)
    print(f"  최적 앙상블 비중 (Large:Small = {best_w_l:.3f}:{best_w_s:.3f}), Temperature = {best_temp:.4f}")
    print(f"  보정 후 OOF LogLoss: {log_loss(oof_y, p_cal):.4f} | OOF Accuracy: {accuracy_score(oof_y, p_cal>=0.5):.4f}")

    # 4. 최종 CampusEnsemble 모델 조립 및 저장
    print("\n=== [Step 5] CampusEnsemble 가중치 통합 및 model.pt 저장 ===")
    ensemble = build_model()
    for i in range(2):
        ensemble.large_models[i].load_state_dict(large_models[i].state_dict())
        ensemble.small_models[i].load_state_dict(small_models[i].state_dict())

    w_tensor = torch.tensor([best_w_l / 2.0, best_w_l / 2.0, best_w_s / 2.0, best_w_s / 2.0], dtype=torch.float32)
    ensemble.ensemble_weights.copy_(w_tensor)
    ensemble.temperature.copy_(torch.tensor(best_temp, dtype=torch.float32))
    ensemble.eval()

    n_params = sum(p.numel() for p in ensemble.parameters())
    print(f"  최종 앙상블 총 파라미터 수: {n_params:,} / {PARAM_LIMIT:,} ({n_params / PARAM_LIMIT:.1%})")
    assert n_params <= PARAM_LIMIT, "파라미터 20M 초과!"

    torch.save(ensemble.state_dict(), "model.pt")
    pt_size_mb = os.path.getsize("model.pt") / (1024 * 1024)
    print(f"  model.pt 저장 완료 ({pt_size_mb:.2f} MB)")

    # 5. 테스트 500장 추론 및 submission.csv 생성
    print("\n=== [Step 6] 테스트 500장 추론 (submission.csv 생성) ===")
    t_inf = time.time()
    test_files = sorted(
        f for f in glob.glob("./test/*")
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    rows = ["id,prob"]
    with torch.no_grad():
        for i in range(0, len(test_files), 32):
            batch_files = test_files[i:i + 32]
            batch = torch.stack([
                tf_orig(Image.open(f).convert("RGB")) for f in batch_files
            ])
            batch_flip = torch.flip(batch, dims=[3])
            logits_orig = ensemble(batch)
            logits_flip = ensemble(batch_flip)
            probs = 0.5 * (torch.sigmoid(logits_orig) + torch.sigmoid(logits_flip)).squeeze(1)
            probs = torch.clamp(probs, min=PROB_CLIP_MIN, max=PROB_CLIP_MAX)
            for f, p in zip(batch_files, probs):
                rows.append(f"{os.path.splitext(os.path.basename(f))[0]},{p.item():.6f}")

    with open("submission.csv", "w") as fp:
        fp.write("\n".join(rows) + "\n")
    print(f"  submission.csv 생성 완료 (500장 추론 소요 시간: {time.time() - t_inf:.2f}초)")

    # 6. submission_package.zip 생성 및 CPU 재현 검증
    print("\n=== [Step 7] submission_package.zip 패키징 및 자가 검증 ===")
    if os.path.exists("package"):
        shutil.rmtree("package")
    os.makedirs("package")
    for fname in ["model.py", "predict.py", "model.pt"]:
        shutil.copy(fname, "package/")

    zip_path = shutil.make_archive("submission_package", "zip", "package")
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  패키지 압축 완료: {zip_path} ({zip_size_mb:.2f} MB / 200 MB 제한)")

    t_pkg = time.time()
    result = subprocess.run(
        [sys.executable, "predict.py", "--input", "../test", "--output", "../package_check.csv"],
        cwd="package",
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())
    if result.stderr.strip():
        print("STDERR:", result.stderr.strip())
    assert result.returncode == 0, "predict.py 실행 실패!"
    print(f"  패키지 단독 CPU 추론 소요 시간: {time.time() - t_pkg:.2f}초 (제한: 600초 이내)")

    def read_probs(path):
        with open(path) as f:
            next(f)
            return {line.split(",")[0]: float(line.split(",")[1]) for line in f if line.strip()}

    nb_probs = read_probs("submission.csv")
    pkg_probs = read_probs("package_check.csv")
    assert nb_probs.keys() == pkg_probs.keys(), "ID 목록 불일치!"
    max_diff = max(abs(nb_probs[k] - pkg_probs[k]) for k in nb_probs)
    print(f"  submission.csv vs package_check.csv 최대 확률 차이: {max_diff:.2e}")
    assert max_diff < 1e-5, "확률 재현 오차 초과!"
    print(f"\n모든 파이프라인 및 패키지 자가 검증 완료! (총 소요 시간: {(time.time() - t_start)/60:.1f}분)")


if __name__ == "__main__":
    main()
