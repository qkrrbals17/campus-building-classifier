# Campus Building Classifier

> 사진 한 장으로 **6공학관 vs IT관**을 구분하는 이미지 분류 모델
> 교내 이미지 분류 챌린지 · **장려상 (4위)** · Log Loss **0.0308**

## 대회 개요

| 항목 | 내용 |
|---|---|
| 과제 | 캠퍼스 건물 사진 이진 분류 (6공학관 = 0, IT관 = 1) |
| 기간 | 2026-11-07 ~ 11-08 (개인전) |
| 지표 | Log Loss (확률 제출, 낮을수록 좋음) |
| 제약 | 총 파라미터 20M 이하 · Colab CPU에서 테스트 500장 10분 이내 · 실행 중 다운로드 금지 · 외부 이미지와 테스트 이미지 학습 금지 |
| 최종 순위 | 공개되지 않은 Private 250장 점수로 산정 |

## 결과

| | |
|---|---|
| 최종 순위 | **4위 (장려상)** |
| Log Loss | **0.0308** |
| 모델 크기 | 약 11.4M 파라미터 (제한 20M의 57%) |

## 접근 방법

베이스라인(약 40만 파라미터 소형 CNN)에서 출발해, 제약 안에서 **정확하면서도 과확신하지 않는 확률**을 내는 데 집중했습니다. Log Loss는 한 장만 크게 틀려도 점수가 무너지기 때문입니다.

```
입력 이미지 ─┬─ MobileNetV3-Large (Fold 0) ─┐
             ├─ MobileNetV3-Large (Fold 1) ─┤  가중 평균 로짓
             ├─ MobileNetV3-Small (Fold 0) ─┼─ ÷ Temperature ─ sigmoid ─ 원본/좌우반전 평균 ─ 확률
             └─ MobileNetV3-Small (Fold 1) ─┘
```

1. **사전학습 백본 앙상블**: ImageNet 사전학습 MobileNetV3-Large 2개와 Small 2개를 서로 다른 Fold로 미세조정해 섞었습니다. 크기가 다른 두 계열을 섞어 오류가 겹치지 않게 했습니다.
2. **CPU 학습 가속**: 백본 앞단을 고정하고 그 출력(feature)을 한 번만 미리 계산해 캐시했습니다. 이후에는 뒷단만 학습해서 GPU 없이 노트북 CPU로 학습을 마쳤습니다.
3. **과적합 억제**: label smoothing (0.02), feature-space Mixup, feature noise, AdamW + Cosine 스케줄을 썼습니다.
4. **확률 보정**: 학습에 쓰지 않은 Out-of-Fold 예측 1,163장으로 Large:Small 비중과 **Temperature**를 함께 최적화하고, 확률을 [0.002, 0.998]로 잘랐습니다. 보정값은 모델 버퍼에 넣어 추론 코드가 그대로 사용합니다.
5. **TTA**: 원본과 좌우 반전 이미지의 예측 확률을 평균했습니다.
6. **제출 자동 검증**: 학습이 끝나면 제출 패키지를 만들고, 패키지만으로 CPU 추론을 다시 돌려 노트북 결과와 확률 차이가 1e-5 미만인지 확인합니다. 운영진 재현 검증에서 탈락하지 않기 위한 장치입니다.

## 파일

| 파일 | 설명 |
|---|---|
| `model.py` | 앙상블 모델 정의 (`build_model()`) |
| `train_local.py` | 학습 → 보정 → `model.pt` 저장 → 테스트 추론 → 패키지 생성·자가 검증까지 한 번에 실행 |
| `predict.py` | 제출용 추론 스크립트 (운영진 Colab CPU 재현용) |
| `requirements.txt` | 필요한 패키지 |

대회 데이터와 학습된 가중치(`model.pt`, 약 46MB)는 포함하지 않았습니다.

## 실행 방법

```bash
pip install -r requirements.txt

# 데이터 배치: ./train/6e, ./train/it, ./test
python train_local.py                                   # 학습 + model.pt + submission.csv

python predict.py --input ./test --output pred.csv      # 추론만
```

## 배운 점

- Log Loss 대회에서는 정확도보다 **확률 보정**이 순위를 가릅니다. Temperature Scaling과 클리핑이 과확신 오답 한 장의 피해를 막아 줍니다.
- 검증 점수를 믿으려면 검증 데이터가 학습에 새지 않아야 합니다. 보정도 반드시 Out-of-Fold 예측으로만 했습니다.
- 제출물이 **다른 환경에서 똑같이 재현되는지** 자동으로 확인하는 단계가 있으면 마감 직전에도 안심하고 제출할 수 있습니다.
