# 📝 Experiments Log

**Smart Warehouse Delay Prediction — 전체 실험 기록**

> Phase 1~8 모든 실험의 상세 기록과 메타 분석. 각 단계의 설계 의도, 결과, 교훈 포함.

---

## 📑 목차

- [Phase 1: 기반 구축 (Exp 9~31)](#phase-1-기반-구축-exp-931)
- [Phase 2: 피처 고도화 (Exp 37~52)](#phase-2-피처-고도화-exp-3752)
- [Phase 3: 앙상블 고도화 (Exp 53~68)](#phase-3-앙상블-고도화-exp-5368)
- [Phase 4: Pseudo-Labeling (Exp 69~72)](#phase-4-pseudo-labeling-exp-6972)
- [Phase 5: TabNet 도입 (Exp 75~76)](#phase-5-tabnet-도입-exp-7576)
- [Phase 6: 교착 상태 돌파 (Exp 77~81)](#phase-6-교착-상태-돌파-exp-7781)
- [Phase 7: 피처 재설계 (Exp 82~87)](#phase-7-피처-재설계-exp-8287-)
- [Phase 8: 모델 다양성 (Exp 88~90)](#phase-8-모델-다양성-exp-8890)
- [전체 메타 교훈](#전체-메타-교훈)

---

## Phase 1: 기반 구축 (Exp 9~31)

**목표**: Baseline에서 기본 피처 + 누수 제거 검증

| 실험 | 핵심 시도 | OOF MAE | LB | 결과 |
|------|---------|---------|-----|------|
| Baseline | 초기 LightGBM | 9.2405 | - | - |
| Exp 9 | Sample Weight (q90/q95/q99) | 8.9723 | - | ✅ |
| Exp 10 | Inverse MAE 앙상블 가중치 | 8.9722 | - | ✅ |
| **Exp 11** | **Lag/Rolling 피처 548개 확장** | **8.8116** | **10.7** | ✅ **최대 단일 개선** |
| Exp 12 | LGB Optuna 튜닝 + 앙상블 | 8.7846 | - | ✅ |
| Exp 13 | Stacking (Ridge meta) | 9.3834 | - | ❌ +0.60 하락 |
| Exp 16 | 모멘텀 피처 + Huber Loss | 8.7301 | - | ✅ |
| Exp 24 | 구조적 피처 추가 | 8.7558 | - | ✅ |
| Exp 25 | Ultimate Integration (619 피처) | 8.7297 | **17.8** | ❌ 누수 |
| Exp 29 | 시나리오 통계 통합 (648 피처) | **8.7214** | **17.8** | ❌ 누수 최대 |
| Exp 30 | 누수 완전 제거 검증 | 9.6182 | 11.0 | ✅ 현실 파악 |
| Exp 31 | CV-safe TE (Smoothing k=30) | 9.0440 | **10.6** | ✅ 누수 제거 |

### 주요 발견

#### 🚨 누수 사례 (가장 중요한 교훈)

```
Exp 25 (누수): OOF 8.72 → LB 17.8  
Exp 29 (누수 최대): OOF 8.72 → LB 17.8
Exp 30 (누수 제거): OOF 9.62 → LB 11.0  
Exp 31 (CV-safe TE): OOF 9.04 → LB 10.6
```

**원인**: `layout_mean_delay` 등 layout 통계를 CV 루프 밖에서 계산.

**올바른 방법** (Exp 31에서 확립):
```python
for fold, (tr_idx, val_idx) in enumerate(gkf.split(...)):
    layout_agg = train.loc[tr_idx].groupby('layout_id')['target'].agg(['mean', 'std', 'count'])
    smoothed = (count * mean + k * global_mean) / (count + k)  # k=30
    # val fold에 merge → unseen layout은 global mean fallback
```

이 발견 이후 **모든 실험에서 CV-safe TE 사용**.

---

## Phase 2: 피처 고도화 (Exp 37~52)

**목표**: 시나리오 상대 변화 피처로 LB 도약

| 실험 | 핵심 시도 | OOF MAE | LB | 결과 |
|------|---------|---------|-----|------|
| Exp 37 fixed | 튜닝 CAT + 3-way 앙상블 + 클리핑 | 8.9951 | 10.5 | ✅ |
| Exp 41 | 시간 기반 Sample Weight 추가 | 8.9832 | - | ✅ |
| Exp 42~44 | vs_start, delta_start (8→22개 컬럼) | 8.9345 | 10.48 | ✅ |
| **Exp 45** | **vs_cummax, vs_cummin 추가** | **8.8337** | **10.33** | ✅ **최대 도약** |
| Exp 46 | position_in_range 추가 | 8.8296 | - | ✅ |
| Exp 47~51 | Lag(4~15) + Rolling(7~20) 확장 | 8.7452 | - | ✅ |
| **Exp 52** | **Lag(20,24) + Rolling 전체 확장** | **8.7493** | **10.181** | ✅ **LB 첫 10.18 돌파** |

### 핵심 코드 (Exp 45 — 최대 단일 도약)

```python
# vs_cummax/cummin: 시나리오 내 누적 최대/최솟값 대비 현재
prev    = df.groupby('scenario_id')[col].shift(1)
cum_max = prev.groupby(df['scenario_id']).cummax()
cum_min = prev.groupby(df['scenario_id']).cummin()
df[f'{col}_vs_cummax'] = df[col] / (cum_max + 1e-6)
df[f'{col}_vs_cummin'] = df[col] / (cum_min.abs() + 1e-6)

# position_in_range: 누적 범위 내 현재 위치 (0~1)
cum_range = cum_max - cum_min
df[f'{col}_position_in_range'] = ((df[col] - cum_min) / (cum_range + 1e-3)).clip(0, 2)
```

**효과**: Exp 44 (LB 10.48) → Exp 45 (LB 10.33), **단일 피처 그룹으로 0.15 향상**.

---

## Phase 3: 앙상블 고도화 (Exp 53~68)

**목표**: 10-seed 앙상블 + 환경 통일 + hyperparameter 미세 조정

| 실험 | 핵심 시도 | OOF | LB | 결과 |
|------|---------|-----|-----|------|
| Exp 54 | Cluster TE 추가 | 8.7428 | 10.188 | ❌ LB 역전 |
| Exp 55 | log1p 변환 제거 | 8.7800 | 10.240 | ❌ |
| Exp 56 | Zero-importance 107개 제거 | 8.7423 | 10.233 | ❌ 단독 역효과 |
| **Exp 57** | **Exp 52 + 107개 제거 + 10-seed** | **8.7465** | **10.162** | ✅ **구조 확립** |
| Exp 58 | Asymmetric 1+10 seed | 8.7513 | 10.167 | ✅ 효율화 |
| Exp 59 | CatBoost Optuna 재튜닝 (GPU) | - | 10.180 | ❌ 환경 분리 오염 |
| **Exp 62** | **Colab GPU 단일환경 이식** | **8.7380** | **10.163** | ✅ **환경 통일** |
| **Exp 63** | **subsample=0.85 + seed 연동** | **8.7313** | **10.160** | ✅ |
| Exp 64 | subsample=0.75 (탐색) | 8.7302 | 10.157 | ✅ |
| **Exp 65** | **colsample LGB=0.65, XGB=0.65** | **8.7286** | **10.156** | ✅ |
| Exp 66 | num_leaves=127 | 8.7250 | 10.156 | ✅ (LB 동일) |
| Exp 67 | 타겟 변환 비교 (log1p vs sqrt vs none) | 8.7247 | 10.156 | ➖ log1p 재확인 |
| Exp 68 | 모델별 혼합 변환 (LGB:sqrt, XGB/CAT:log1p) | **8.7234** | 10.189 | ❌ sqrt 역변환 오차 증폭 |

### Phase 3의 핵심 발견

#### 🔍 Asymmetric Seed Ensemble

LGB와 XGB는 `subsample`, `colsample_bytree` 없으면 seed를 바꿔도 결과 동일. 이 파라미터 추가 후 진짜 다양성 확보.

```python
# 변경 전: 10 seed 전체에서 LGB iteration 고정 (사실상 1 seed)
# 변경 후: seed마다 다른 iteration → 실제 다양성 확보
lgb_params = {
    ...
    'subsample': 0.75,
    'colsample_bytree': 0.65,
    'subsample_freq': 1,
    'random_state': seed   # ← seed 연동 핵심
}
```

#### 🔍 subsample 최적값 탐색

```
subsample=0.70 → OOF MAE: 8.7350
subsample=0.75 → OOF MAE: 8.7297 ← 최적
subsample=0.80 → OOF MAE: 8.7325
subsample=0.85 → OOF MAE: 8.7342
subsample=0.90 → OOF MAE: 8.7364
```

#### 🔍 타겟 변환 모델별 분석

```
LGB  → sqrt 가 개별 OOF 8.7376 (log1p 8.7541보다 좋음)
XGB  → log1p 가 개별 OOF 더 좋음
CAT  → log1p 필수 (sqrt 적용 시 8.8453으로 폭락)
```

**결론**: 단독으로는 sqrt가 LGB에 좋지만 **혼합 적용 시 sqrt 역변환 오차 증폭으로 LB 역효과** (Exp 68: OOF 8.72 → LB 10.19). **일관된 단일 변환(log1p) 사용**이 정답.

---

## Phase 4: Pseudo-Labeling (Exp 69~72)

**목표**: Test 예측값을 학습 데이터로 활용

| 실험 | 핵심 시도 | OOF | LB |
|------|---------|-----|-----|
| **Exp 69** | **PL 도입 (전략C: Seen pseudo only)** | - | **10.141** |
| **Exp 70b** | **갱신 pseudo + Unseen 소량 재도입** | - | **10.129** |
| **Exp 70-A** | **Seen pseudo only (참고)** | - | 10.133 |
| Exp 72 | 예측 블렌딩 | - | 10.1287 |

### 전략별 비교

| 전략 | 설명 | LB |
|------|------|-----|
| 전략 A | Pseudo 전체 사용 | 10.15+ |
| 전략 B | Pseudo Top 30% | 10.14 |
| **전략 C** | **Seen layout만 (가장 안전)** | **10.133** |
| 전략 D | Confidence weight (Exp 79에서 확장) | 10.121 |

#### 🔍 Pseudo Label 수렴

```
Round 1: 70b 예측을 PL로 사용 → 학습
Round 2: Round 1 결과를 PL로 사용 → 학습  
Round 3: Round 2 결과를 PL로 사용 → 수렴 (개선 없음)
```

PL은 빠르게 수렴. 무한 갱신은 의미 없음.

---

## Phase 5: TabNet 도입 (Exp 75~76)

**목표**: 정형 데이터에 NN 구조 추가로 다양성 확보

| 실험 | 핵심 시도 | OOF | LB |
|------|---------|-----|-----|
| **Exp 75** | **TabNet 5-seed (n_d=64, n_steps=5)** | **8.8798** | - |
| **Exp 75 blend** | **γ=0.05 (95% GBDT + 5% TabNet)** | - | **10.12838** ⭐ |
| Exp 76 | TabNet n_d=128 (표현력 확대) | 9.32+ | - ❌ 과적합 |
| Exp 76-B | TabNet 규제 강화 (n_steps=3, λ=1e-3) | 9.63+ | - ❌ 과도한 규제 |

### TabNet 검증된 구성

```python
TabNetRegressor(
    n_d=64, n_a=64,        # Exp 76(128)은 과적합
    n_steps=5,              # Exp 76-B(3)는 과도한 규제
    gamma=1.5,
    lambda_sparse=1e-3,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax',
)
```

#### 🔍 블렌딩 Sweet Spot

```
γ=0.0 (GBDT만):   LB 10.135
γ=0.05:           LB 10.128 ⭐
γ=0.10:           LB 10.130
γ=0.20:           LB 10.140
```

**5%가 최적**. TabNet OOF가 GBDT보다 나빠도 다양성으로 블렌딩 효과.

---

## Phase 6: 교착 상태 돌파 (Exp 77~81)

**목표**: LB 10.128 교착 상태 돌파

| 실험 | 핵심 시도 | OOF | LB | 결과 |
|------|---------|-----|-----|------|
| Exp 77 | Layout-Agnostic 조건부 블렌딩 | 8.7627 (agn) | 10.205 (s1) | ❌ |
| Exp 78 | MLP 제3모델군 도입 | 9.094 | - | ❌ 상관 0.977 |
| **Exp 79 R1** | **합의 PL + Confidence Weight** | **8.6942** | **10.1218** | ✅ |
| **Exp 79 R2** | **R1 PL 갱신 → 재학습** | **8.6906** | **10.1201** | ✅ **Phase 6 PB** |
| Exp 79 R3 | R2 PL 갱신 → 재학습 | 8.6894 | 10.1212 | ❌ 수렴 통과 |
| Exp 80 | TabNet 재학습 (R2 PL 기반) | - | 10.12 | ❌ 효과 없음 |
| Exp 81 | Loss Function 탐색 | - | - | ❌ 기존 최적 |

### Phase 6의 핵심 발견

#### 🔍 Confidence-Weighted Pseudo Label (Exp 79)

```python
# 모델 예측 분산 → confidence weight
pred_matrix = np.column_stack([preds[name] for name in preds])
pred_std = pred_matrix.std(axis=1)

# 합의 PL = LB inverse 가중 평균
weights = {name: (1.0 / lb_score) for name, lb_score in LB_SCORES.items()}
weights = {k: v / sum(weights.values()) for k, v in weights.items()}

consensus_pl = sum(w * preds[name] for name, w in weights.items())

# Sample weight = 모델 동의도 기반
def make_sample_weight_v2(df):
    base_weight = np.where(df['is_pseudo'].fillna(False), 0.6, 1.0)
    if 'pseudo_std' in df:
        confidence = 1.0 / (df['pseudo_std'].fillna(0) + 0.5)
        confidence_norm = confidence / confidence.max()
        pseudo_mask = df['is_pseudo'].fillna(False)
        base_weight = np.where(pseudo_mask, base_weight * confidence_norm, base_weight)
    return base_weight
```

#### 🔍 앙상블 다양성의 한계

GBDT 3종(LGB/XGB/CAT) 상관관계 0.999, TabNet 0.98, MLP 0.977. 정형 데이터에서 구조적으로 다른 모델을 써도 상관관계를 0.95 이하로 낮추기 극히 어려움. γ=0.05 블렌딩이 한계이며, 이마저도 base GBDT가 이미 최적점에 가까우면 효과 소멸.

#### 🔍 PL 수렴 패턴

```
Round 1 → Round 2: -0.0017 LB
Round 2 → Round 3: +0.0011 LB (오히려 악화)
```

PL refinement는 **2~3 라운드 후 수렴 또는 악화**. 무한 갱신 비효율.

---

## Phase 7: 피처 재설계 (Exp 82~87) ⭐

**목표**: 교착 돌파를 위한 피처 재설계 (단일 phase 최대 도약)

### 흐름

```
교착 상태 (10.1201)
    ↓ V2: 피처 재설계 (하위 제거 + 신규 추가)
LB 10.0813 (-0.039)
    ↓ V3: V2 importance 분석 → 진화
LB 10.0737 (-0.008)
    ↓ V4: Percentile Rank 대폭 확장 (3개 → 38개)
LB 10.0480 (-0.026) ⭐
    ↓ V5: 다차원 Percentile (실패)
    ↓ V4.5: 정밀 정제 (실패)
GBDT Ceiling 도달
```

### Exp 82 (V2): 피처 재설계 시작

| 항목 | 값 |
|------|-----|
| 피처 변화 | 446 → 339 (하위 150 제거 + 신규 43) |
| OOF | 8.7018 → 8.6438 (-0.058) |
| LB | 10.1201 → **10.0813** (-0.039) ⭐ |

**신규 피처 6그룹**:
- A. 2차 차분 (acceleration) — 6개
- B. 핵심 피처 비율/차이 — 16개
- C. 지수 가중 이동평균 (EWM) — 10개
- D. 현재값 vs EWM 비율 — 5개
- E. **Scenario percentile rank — 3개** ⭐ (importance 1, 3위)
- F. Rolling 변동계수 — 3개

**핵심 발견**: V2 importance 분석에서 `congestion_score_scen_pctrank`가 rank 10으로 압도적 1위. 단 3개 추가했는데 모두 상위 진입.

### Exp 84 (V3): 피처 진화

| 항목 | 값 |
|------|-----|
| 피처 변화 | 339 → 277 (V2 하위 100 제거 + 신규 38) |
| OOF | 8.6438 → 8.6392 (-0.005) |
| LB | 10.0813 → **10.0737** (-0.008, 전이율 1.65) |

**신규 피처 6그룹**:
- G. 다중 시간 스케일 비율
- H. Layout × 운영 추가 교호작용
- I. EWM 확장 (span=3, 20)
- J. 추세 일관성 (sign 비율)
- K. 3차 차분 (jerk)
- L. Cross-acceleration 비율

**전이율 1.65**: OOF보다 LB가 더 좋아짐. 정직한 신호.

### Exp 85 (V4): Percentile Rank 대폭 확장 ⭐

| 항목 | 값 |
|------|-----|
| 피처 변화 | 277 → 219 (V3 하위 100 + acceleration 모두 제거 + 신규 45) |
| OOF | 8.6392 → 8.6104 (-0.029) |
| LB | 10.0737 → **10.0480** (-0.026, 전이율 0.89) ⭐ |

**핵심 변경**:
1. **Acceleration 계열 모두 제거** — V2 분석에서 모두 하위 25%
2. **Percentile rank 3개 → 38개로 확장**:
   - M. Scenario 내 percentile (15개)
   - N. Rolling 30 percentile (12개)
   - O. Rolling 10 percentile (8개)
   - P. since_max/min (10개)

**가설 검증 완료**: Percentile rank가 GBDT의 missing 신호. 38개로 확장 시 LB **-0.026** 도약.

### Exp 86 (V5): 다차원 Percentile (실패)

| 항목 | 값 |
|------|-----|
| 시도 | V4 + 다차원 Percentile (Layout, Roll60, Velocity, Cumulative) |
| OOF (1-seed) | 8.6183 (V4 1-seed 8.6164 대비 +0.002) |
| 결과 | ❌ V4 못 이김 |

**V4 진단 결과** (V5 설계 단서):
```
M (scen_pct):     평균 rank 141.6 — Effective ✅
N (roll30_pct):   평균 rank 193.9 — Noise ❌
O (roll10_pct):   평균 rank 206.6 — Noise ❌
P (since_max/min):평균 rank 192.4 — Noise ❌
```

**교훈**: Percentile rank 자체가 강한 게 아니라 **specific한 형태(scenario 내 위치)가 핵심**. 다른 차원으로 확장은 효과 없음.

### Exp 87 (V4.5): 정밀 정제 (실패)

| 항목 | 값 |
|------|-----|
| 시도 | V4 importance 활용 — N, O, P 그룹 + M 하위 제거 (39개) + V4 하위 30개 |
| OOF (1-seed) | 8.6190 (V4 1-seed 8.6164 대비 +0.003) |
| 결과 | ❌ V4 못 이김 |

**충격적 발견**: V4 importance 하위 39개를 제거했더니 OOF 악화.

```
V4 신규 42개 중 importance:
  상위 25%: 3개
  하위 25%: 37개 (88%)
```

**교훈**: Tree 모델에서 **약한 피처 35개의 집단 효과**가 V4 성능을 받침. Importance가 낮아도 함부로 제거하면 안 됨. **Zero importance dead feature만 안전하게 제거 가능**.

### Exp 83 (Pseudo Label Refinement): 누수 발견

| 항목 | 값 |
|------|-----|
| 시도 | V2(LB 10.08) 중심 PL refinement |
| OOF | 7.4699 (비정상적으로 낮음) |
| LB | 10.1111 (gap +2.64) |
| 결과 | ❌ 누수 |

**누수 메커니즘 발견**:
- `train_gbdt_round`의 `eval_set`에 pseudo 행 포함
- early stopping이 **PL을 모방하도록 학습**
- PL 품질이 V4급으로 좋아지면 누수 확대

**교훈**:
- Exp 79까지 PL이 LB 10.13 수준이라 누수 영향 작았음
- PL이 V4급이 되면 누수 제거 필수
- 해결: eval_set에서 pseudo 행 완전 제외

---

## Phase 8: 모델 다양성 (Exp 88~90)

**목표**: GBDT ceiling을 NN 구조로 돌파

| 실험 | 시도 | OOF | ↔V4 상관 | LB | 결과 |
|------|------|-----|---------|-----|------|
| **Exp 88** | **V4 + TabNet 10% (n_d=64, steps=5)** | TabNet 8.7682 | **0.9724** | **10.0451** | ⭐ **최종 PB** |
| Exp 89 | Multi-TabNet 앙상블 (A+B+C) | 8.7417 | 0.9739 | - | ❌ 다양성 약화 |
| Exp 90 | V4 + MLP 10% | MLP 9.3715 | 0.9545 | 10.0514 | ❌ 충돌 |

### Exp 88: V4 + TabNet — 최종 PB ⭐

| Alpha | LB |
|-------|-----|
| 5% | 미제출 |
| **10%** | **10.0451** ⭐ |
| 15% | 10.0459 |

**Alpha 10%가 sweet spot**. 15%는 살짝 나쁨, 5%는 V2 사례에서 보수적.

```python
# 최종 PB 구성
pred_v4 = result_v4['test_pred']
pred_tabnet = result_tabnet['test_pred']
final = 0.90 * pred_v4 + 0.10 * pred_tabnet
final = np.maximum(final, 0)
```

### Exp 89: Multi-TabNet 앙상블 (실패)

3개 TabNet 다른 hyperparameter:
- TabNet_A: n_d=64, n_steps=5 (Exp 88)
- TabNet_B: n_d=48, n_steps=4 (작은 모델)
- TabNet_C: n_d=64, n_steps=6 (확장)

```
TabNet_A ↔ V4: 0.9724
TabNet_B ↔ V4: 0.9681 (가장 다양)
TabNet_C ↔ V4: 0.9745 (V4와 비슷)
A+B+C 앙상블 ↔ V4: 0.9739 (다양성 약화)
```

**교훈**: TabNet 끼리는 함수 공간 비슷. 같은 모델 클래스 내 hyperparameter 다양화는 한계.

### Exp 90: V4 + MLP (실패)

| 지표 | TabNet (Exp 88) | MLP (Exp 90) |
|------|----------------|-------------|
| OOF | 8.7682 | **9.3715** (V4 대비 +0.76) |
| ↔ V4 상관 | 0.9724 | **0.9545** |
| 블렌딩 LB | 10.0451 (-0.003) | **10.0514** (+0.003) ❌ |

**충격적 발견**: 다양성이 더 높아도 (0.9545 < 0.9724) 블렌딩 효과 ❌.

**원인 분석**:
1. MLP OOF 9.37로 V4 8.61보다 0.76 나쁨 (gap 너무 큼)
2. MLP가 *다른 영역*에서도 자주 틀림 → V4 끌어내림
3. 블렌딩 효과는 *V4가 못 맞추는 영역에서만* 다른 신호 줘야 함

**교훈**: **상관관계만 낮다고 능사 아님**. Sweet spot은 0.97~0.98.

---

## 전체 메타 교훈

### 1. 누수 (Data Leakage) — 가장 큰 적

```
Exp 25/29 (누수): OOF 8.72 → LB 17.8
Exp 31 (제거): OOF 9.04 → LB 10.6 (현실)
Exp 83 (PL 누수): OOF 7.47 → LB 10.11 (gap 2.64)
```

**누수의 다양한 형태**:
1. CV 밖 layout 통계 → 가장 흔함, Phase 1에서 발견
2. PL이 너무 정확해서 eval_set 누수 → Phase 7에서 발견
3. Pseudo target이 fold 간 정보 전이 → 이론적, 차단됨

### 2. OOF MAE ≠ LB 점수

Phase 6까지는 역전 빈번:
| 실험 | OOF | LB | 결과 |
|------|-----|-----|------|
| Exp 56 | 8.7423 | 10.233 | ❌ |
| Exp 68 | 8.7234 | 10.189 | ❌ |

**Phase 7부터 정직한 전이 시작** — *추가 + 제거 동시* 패턴 덕분.

### 3. 피처 진화 사이클이 가장 효과적

```
1. Importance 분석
2. 하위 80~150개 제거
3. 신규 피처 그룹 추가
4. 1-seed 비교 → 10-seed 학습
5. 반복
```

V2 → V3 → V4 사이클로 LB **-0.072**. 단일 phase 최대.

### 4. Specific 형태가 다양성보다 중요

**Percentile Rank 가설 검증**:
- V2: scen_pct 3개 → importance 1, 3위
- V4: scen_pct + roll10/30 + since_max 38개 → LB -0.026
- V5: 다른 차원의 Percentile 38개 → 실패

**교훈**: 차원이 아니라 **scenario 내 위치**라는 specific한 형태가 핵심.

### 5. 모델 다양성의 sweet spot

| 상관 ↔ V4 | 블렌딩 결과 |
|-----------|-----------|
| 0.999+ | 효과 0 (GBDT 끼리) |
| 0.98 | 효과 0~미세 (Exp 80) |
| **0.972** | **효과 -0.003** (Exp 88 ⭐) |
| 0.974 | 약화 (Exp 89) |
| 0.954 | 충돌 -0.003 (Exp 90 ❌) |

**Sweet spot 0.97~0.98**. 너무 다양하면 충돌.

블렌딩 조건:
1. 상관 0.97~0.98
2. 단독 OOF gap < 0.3
3. *잘못된 영역에서만* 다를 것

### 6. Importance 낮아도 함부로 제거 X

Tree 모델은 약한 피처들의 *집단 효과* 활용. V4의 신규 피처 88%가 하위 25%였지만 모두 제거하면 OOF 악화 (Exp 87).

**안전한 제거**:
- Zero importance dead feature만
- 또는 동시에 신규 추가 (V2~V4 패턴)

### 7. PL refinement는 빠르게 수렴

```
Round 1: -0.008 LB
Round 2: -0.0017 LB  
Round 3: +0.0011 LB (악화)
```

2~3 라운드 후 수렴 또는 악화. **무한 갱신 비효율**.

### 8. ROI 기반 의사결정의 가치

대회 후반에 모든 시도가 -0.001~0.005 영역. ROI 추정 필수:

| 시도 | 시간 | 새 PB 확률 |
|------|------|-----------|
| 피처 재설계 (V2~V4) | 6시간 | 80% |
| TabNet (V4 PL 기반) | 2시간 | 30% |
| Multi-TabNet | 3.5시간 | 20% |
| MLP | 2시간 | 30% (실제 0%) |
| PL refinement | 2.5시간 | 20% |

**판단 기준**: 시간 대비 기대 효과 비교. 안정적 마무리도 합리적 선택.

### 9. 환경 통일의 중요성

Exp 59 사례:
- Local CPU + Colab GPU 혼용
- 부동소수점 차이로 OOF 계산 오염
- LGBM MAE 18.11 같은 비정상 결과

**해결**: Exp 62 이후 Colab 단일 환경. CPU/GPU 분리:
- LGB CPU
- XGB GPU (`device='cuda'`)
- CAT GPU (`task_type='GPU'`)

학습 시간 5~6시간 → 2시간 단축.

### 10. 실패도 자산이다

이번 대회의 실패 사례:
- Stacking, Quantile Loss, Layout-Agnostic, Cluster TE
- V5 (다차원 Percentile), V4.5 (정밀 정제)
- Multi-TabNet, MLP

모두 **다음 대회의 시간 절약**. 같은 길을 다시 가지 않을 수 있음.

---

## 결론

90개 실험으로 LB **-0.555** 누적 개선. **117등 / 608팀 (상위 19.2%)**.

핵심 자산:
- **검증된 방법론**: CV-safe TE, 피처 진화 사이클, Confidence PL, V4+TabNet 블렌딩
- **재활용 코드**: vs_cummax, scen_pct, train_tabnet_round 등
- **메타 교훈**: 위 10가지 통찰

다음 정형 데이터 대회에서 시작점부터 다를 것이다.
