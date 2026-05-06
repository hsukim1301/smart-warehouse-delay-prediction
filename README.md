# 🏭 Smart Warehouse Delay Prediction

**Dacon 스마트 창고 출고 지연 예측 대회**

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-✓-green)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange)](https://xgboost.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-✓-yellow)](https://catboost.ai)
[![TabNet](https://img.shields.io/badge/TabNet-PyTorch-red)](https://github.com/dreamquark-ai/tabnet)

> 90개 실험을 통한 정형 데이터 모델링 방법론 연구. **누수 제거 → 피처 진화 → 모델 다양성**의 3단계 접근으로 LB 0.555 개선.

---

## 🏆 최종 성과

| 지표 | 값 |
|------|-----|
| **최종 LB 점수** | **10.0451** |
| **최종 등수** | **112 / 607팀 (상위 19.2%)** |
| **최고 OOF MAE** | 8.6104 (Exp 85 V4) |
| **시작 → 종료** | 10.6 → 10.0451 |
| **누적 개선** | **−0.555 LB** |

### Phase별 도약

```
Phase 1~3 (GBDT 기반):       10.6   → 10.156   (-0.444)
Phase 4 (Pseudo Labeling):   10.156 → 10.129   (-0.027)
Phase 5 (TabNet 5%):         10.129 → 10.128   (-0.001)
Phase 6 (Confidence PL):     10.128 → 10.1201  (-0.008)
Phase 7 (피처 재설계):       10.1201 → 10.0480  (-0.072) ⭐ 최대 도약
Phase 8 (NN 다양성):         10.0480 → 10.0451  (-0.003)
```

**Phase 7이 단일 phase 최대 도약** (-0.072). V2 → V3 → V4 피처 진화 사이클이 결정적.

---

## 📋 대회 개요

| 항목 | 내용 |
|------|------|
| **목표** | 향후 30분간 창고별 평균 출고 지연 시간(분) 예측 |
| **평가 지표** | MAE (Mean Absolute Error) |
| **Train** | 250,000행 × 94컬럼 |
| **Test** | 50,000행 × 93컬럼 |
| **레이아웃** | 300개 (Train: 250개, Test: 100개) |
| **핵심 난이도** | Test의 **40%가 Unseen Layout** (train에 없음) |

```
Train layout(250) ∩ Test layout(100) = Seen(50개)
Test에만 있는 Unseen layout = 50개

최종 LB = Seen(60%) × MAE_seen + Unseen(40%) × MAE_unseen
```

---

## 🔑 가장 중요한 교훈 5가지

### 1. 누수가 모든 것을 망친다

타겟 기반 layout 통계를 CV 루프 **밖**에서 계산하면 심각한 누수.

```python
# ❌ 누수
layout_agg = train.groupby('layout_id')['target'].mean()

# ✅ CV-safe (Smoothing k=30)
for fold, (tr_idx, val_idx) in enumerate(gkf.split(...)):
    layout_agg = train.loc[tr_idx].groupby('layout_id')['target'].mean()
    smoothed = (count * mean + k * global_mean) / (count + k)
```

**실제 영향**:
- Exp 29 (누수): OOF 8.72 → LB **17.8** ❌
- Exp 31 (제거): OOF 9.04 → LB **10.6** ✅

원인: Test의 40%가 Unseen layout. 누수 피처가 Unseen에서 NaN→0으로 폭발.

---

### 2. OOF MAE ≠ LB 점수 (Phase 6까지)

| 실험 | OOF | LB | 결과 |
|------|-----|-----|------|
| Exp 56 (피처 제거) | 8.7423 | 10.233 | ❌ 역전 |
| Exp 68 (혼합 변환) | 8.7234 | 10.189 | ❌ 역전 |

**그러나 Phase 7부터 정직한 전이가 시작**:
- V2: OOF -0.047 → LB -0.039 (전이율 0.83) ✅
- V4: OOF -0.029 → LB -0.026 (전이율 0.89) ✅

**원인**: Phase 7부터 *추가 + 제거 동시 진행* 패턴으로 noise도 함께 정리.

---

### 3. Percentile Rank — Phase 7의 결정적 단일 신호

V2 importance 분석에서 발견:

```
v2 신규 피처 43개 중 importance 상위:
  congestion_score_scen_pctrank   rank=10   ⭐ 1위
  pack_utilization_scen_pctrank   rank=45   3위
  (단 3개만 추가했는데 모두 상위 진입)
```

**가설 검증** (V4): 3개 → 38개 확장 시 LB **−0.0257** (전이율 0.89).

**왜 강력한가**:
- GBDT는 절대값으로 분기 → "scenario 내 상대 위치"는 직접 표현 못 함
- Percentile rank가 *상대* 정보를 명시화
- Unseen layout 일반화에 특히 강함

---

### 4. 피처 진화 사이클 — 검증된 LB 돌파 패턴

```
1. 현재 모델 importance 분석
2. 하위 80~150개 제거 (noise 감소)
3. 신규 피처 그룹 추가 (다른 차원)
4. 1-seed 비교 → 10-seed 학습
5. (반복)
```

**Phase 7 누적 효과**:
| 단계 | 피처 수 | LB | 변화 |
|------|--------|-----|------|
| 시작 | 446 | 10.1201 | — |
| V2 | 339 | 10.0813 | -0.039 |
| V3 | 277 | 10.0737 | -0.008 |
| V4 | 219 | 10.0480 | -0.026 ⭐ |

**한계** (Phase 8): V5(다차원 percentile), V4.5(정밀 정제) 모두 V4 못 이김. 같은 모델 클래스 내 ceiling 존재.

---

## 5. 모델 다양성 — 상관관계 0.97~0.98이 sweet spot

| 모델 | ↔ V4 상관 | 블렌딩 결과 |
|------|----------|------------|
| TabNet (R2 PL, Exp 80) | 0.981 | 효과 없음 |
| **TabNet (V4 PL, Exp 88)** | **0.9724** | **LB 10.0451** ⭐ |
| Multi-TabNet (Exp 89) | 0.9739 | 다양성 약화 |
| **MLP (V4 PL, Exp 90)** | **0.9545** | LB 10.0514 ❌ 충돌 |

**놀라운 발견**: 다양성이 너무 강하면 충돌. 적정 상관 **0.97~0.98**이 sweet spot.

블렌딩 효과의 진짜 조건:
1. 상관관계 0.97~0.98
2. 단독 OOF gap < 0.3
3. *잘못된 영역에서만* 다를 것

MLP는 (1)은 OK였지만 (2),(3) 미달 → V4 끌어내림.

---

## 🎯 가장 효과적인 피처 그룹

### Phase 1~2 핵심: 시나리오 상대 변화

```python
# vs_cummax/cummin (단일 도약 LB -0.15)
prev = df.groupby('scenario_id')[col].shift(1)
cum_max = prev.groupby(df['scenario_id']).cummax()
df[f'{col}_vs_cummax'] = df[col] / (cum_max + 1e-6)

# vs_start
first_val = df.groupby('scenario_id')[col].transform('first')
df[f'{col}_vs_start'] = df[col] / (first_val + 1e-6)
```

### Phase 7 핵심: Scenario Percentile Rank

```python
# Scenario 내 percentile (V4의 결정적 신호)
df[f'{col}_scen_pct'] = df.groupby('scenario_id')[col].rank(pct=True)

# Rolling 30 percentile
shifted = df.groupby('scenario_id')[col].shift(1)
df[f'{col}_roll30_pct'] = shifted.groupby(df['scenario_id']).transform(
    lambda x: x.rolling(30, min_periods=5).rank(pct=True)
)

# Rolling mean의 percentile (V4 importance 1, 3위)
df[f'{col}_roll10_mean_scen_pct'] = df.groupby('scenario_id')[
    f'{col}_roll10_mean'].rank(pct=True)
```

### Phase 8 핵심: V4 + TabNet 블렌딩

```python
# TabNet (Exp 75 검증된 sweet spot)
tabnet = TabNetRegressor(
    n_d=64, n_a=64, n_steps=5,
    gamma=1.5, lambda_sparse=1e-3,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax',
    device_name='cuda',
)

# 최종 PB 구성
final = 0.90 * pred_v4 + 0.10 * pred_tabnet
```

---

## 📊 전체 흐름 요약

```
Phase 1~2 (GBDT 기반): 누수 제거 + 피처 엔지니어링
  Baseline (9.24) → vs_cummax/cummin (LB 10.33) → Lag 확장 (LB 10.18)

Phase 3 (앙상블 고도화): Asymmetric Seed
  10-seed + subsample/colsample 최적화 → LB 10.156

Phase 4 (Pseudo Labeling): Seen pseudo + 갱신 사이클
  → LB 10.129

Phase 5 (TabNet): GBDT 95% + TabNet 5%
  → LB 10.128

Phase 6 (Confidence PL): 합의 PL + 분산 기반 weight
  → LB 10.1201

Phase 7 (피처 재설계): V2 → V3 → V4 진화 사이클 ⭐
  → LB 10.0480 (단일 phase 최대 도약 -0.072)

Phase 8 (NN 다양성): V4 + TabNet 블렌딩
  → LB 10.0451 (최종 PB)
```

상세 실험 기록은 [experiments_log.md](experiments_log.md) 참고.

---

## 🛠️ 환경

```
Python 3.x
lightgbm, xgboost==3.2.0, catboost
pytorch-tabnet, torch
scikit-learn, pandas, numpy
optuna, scipy
Google Colab (L4 GPU)
```

### Colab L4 GPU 최적화

| 모델 | 설정 |
|------|------|
| LightGBM | CPU (GPU 모드는 256 bin 제한) |
| XGBoost | `device='cuda'` |
| CatBoost | `task_type='GPU', devices='0'` |
| TabNet | `batch_size=4096, virtual_batch_size=256` |

CPU 대비 학습 시간 **5~6시간 → 2~3시간** 단축.

---

## 📁 파일 구조

```
├── README.md                        # 본 문서 (요약)
├── experiments_log.md               # 전체 실험 상세 기록 ⭐
│
├── notebooks/
│   ├── Phase 1~3 (GBDT 기반)
│   │   ├── experiment_52.py         # LB 최초 10.18
│   │   ├── experiment_57.py         # 10-seed 구조
│   │   ├── experiment_62.ipynb      # Colab GPU 이식
│   │   └── experiment_67.ipynb      # 타겟 변환 비교
│   │
│   ├── Phase 4 (Pseudo Labeling)
│   │   ├── experiment_69.ipynb      # PL 도입
│   │   ├── experiment_70b.ipynb     # LB 10.129
│   │   └── experiment_72.ipynb      # 예측 블렌딩
│   │
│   ├── Phase 5 (TabNet)
│   │   └── experiment_75.ipynb      # LB 10.128
│   │
│   ├── Phase 6 (교착 돌파)
│   │   └── experiment_79.ipynb      # 합의 PL + Confidence (LB 10.1201)
│   │
│   ├── Phase 7 (피처 재설계) ⭐
│   │   ├── experiment_82.ipynb      # V2 (LB 10.0813)
│   │   ├── experiment_84.ipynb      # V3 (LB 10.0737)
│   │   ├── experiment_85.ipynb      # V4 (LB 10.0480)
│   │   ├── experiment_86.ipynb      # V5 (실패 사례)
│   │   └── experiment_87.ipynb      # V4.5 (실패 사례)
│   │
│   └── Phase 8 (NN 다양성)
│       ├── experiment_88.ipynb      # V4+TabNet (LB 10.0451) ⭐ Final PB
│       ├── experiment_89.ipynb      # Multi-TabNet (실패)
│       └── experiment_90.ipynb      # V4+MLP (실패)
│
└── submissions/
    ├── submission_85_v4.csv          # V4 단독 (10.0480)
    └── submission_88_v4_tabnet_a10.csv  # 최종 PB (10.0451) ⭐
```

---

## 🎓 핵심 자산 (다음 대회에서 재활용)

### 검증된 방법론
1. **CV-safe Target Encoding** (Smoothing k=30, GroupKFold)
2. **피처 진화 사이클** (importance → 제거 + 신규 → 반복)
3. **Confidence-Weighted Pseudo Label** (분산 기반 weight)
4. **Inverse-LB 가중 합의 PL**
5. **GBDT + TabNet 블렌딩** (상관 0.97~0.98 sweet spot)

### 재활용 가능 코드
- 시나리오 상대 변화 (vs_cummax, vs_start)
- Scenario Percentile Rank (다중 시간 스케일)
- 누수 없는 PL 사이클 (eval_set에서 pseudo 제외)

### 검증된 hyperparameter
- **GBDT**: subsample=0.75, colsample=0.65, num_leaves=127
- **TabNet**: n_d=64, n_steps=5, lr=2e-2 (Exp 75 sweet spot)
- **Loss**: log1p 변환 + MAE/Huber

---

## 🚀 향후 개선 방향 (대회 후 회고)

대회 종료 시 시도해보지 못한 것들:

1. **PL refinement with 누수 회피** — V4 PL + eval_set에서 pseudo 제외 (Exp 83 패턴)
2. **Stacking 메타 모델** — V4, TabNet, GBDT 변형들을 메타 LGB로 묶기
3. **Self-training 사이클** — V4 예측을 다시 PL로 → V4' → 수렴까지

다만 모두 −0.003 이하 영역의 미세 개선 후보. 현실적 한계는 GBDT + TabNet 조합의 ceiling.

---

## 📜 라이센스

MIT License — 자유롭게 참고 및 수정 가능.

---

## 🙏 마치며

이 대회를 통해:
- **정형 데이터 모델링의 깊이와 한계 모두 경험**
- **메타 교훈** (OOF↔LB 역전, 다양성 sweet spot, 피처 진화 사이클) 정립
- **검증된 코드와 방법론**이 다음 대회의 자산으로 남음

**최종 PB**: 10.0451 (Exp 88, V4 + TabNet 10%)  
**117등 / 608팀 (상위 19.2%)** — 90개 실험으로 도달한 정직한 결과.
