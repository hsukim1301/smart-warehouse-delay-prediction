# 🏭 Smart Warehouse Delay Prediction
**Dacon 스마트 창고 출고 지연 예측 대회**

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-✓-green)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange)](https://xgboost.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-✓-yellow)](https://catboost.ai)

---

## 📋 대회 개요

| 항목 | 내용 |
|------|------|
| **목표** | 향후 30분간 창고별 평균 출고 지연 시간(분) 예측 |
| **평가 지표** | MAE (Mean Absolute Error) |
| **Train** | 250,000행 × 94컬럼 |
| **Test** | 50,000행 × 93컬럼 |
| **레이아웃 구조** | 300개 레이아웃 (Train: 250개, Test: 100개) |
| **핵심 난이도** | Test의 40%가 Train에 없는 **Unseen Layout** |

### 데이터 구조 핵심

```
Train layout(250개) ∩ Test layout(100개) = Seen layout(50개)
Test에만 있는 Unseen layout = 50개 (Test의 40%)

최종 LB 점수 = Seen(60%) × MAE_seen + Unseen(40%) × MAE_unseen
```

---

## 🏆 최종 성과

| 지표 | 값 |
|------|-----|
| **최고 LB 점수** | **10.156** (Exp 65~66~67) |
| **최고 OOF MAE** | 8.7234 (Exp 68) |
| **Baseline → Final** | 9.2405 → 10.156 (LB 기준) |
| **현재 순위** | 103등 / 494팀 (상위 21%) |

---

## 🔑 핵심 인사이트

### 1. 누수(Data Leakage) — 가장 중요한 교훈

타겟 기반 레이아웃 통계를 CV 루프 **밖**에서 계산하면 심각한 누수가 발생한다.

```python
# ❌ 잘못된 방식 (누수 발생)
layout_agg = train.groupby('layout_id')['target'].mean()
train = train.merge(layout_agg, on='layout_id')

# ✅ 올바른 방식 (CV-safe)
for fold, (tr_idx, val_idx) in enumerate(gkf.split(...)):
    layout_agg = train.loc[tr_idx].groupby('layout_id')['target'].mean()
    smoothed = (count * mean + k * global_mean) / (count + k)  # Smoothing k=30
    # val fold에 merge → Unseen layout은 global mean fallback
```

**실제 영향**: 누수 있는 Exp 29 (OOF 8.72) → LB **17.8점**. 누수 제거 후 Exp 31 (OOF 9.04) → LB **10.6점**.

---

### 2. OOF MAE ≠ LB 점수

이 대회에서 가장 반복적으로 나타난 패턴은 **OOF가 개선되어도 LB가 나빠지는 역전 현상**이다.

| 실험 | OOF MAE | LB 점수 | 결과 |
|------|---------|---------|------|
| Exp 52 | 8.7493 | **10.181** | ✅ |
| Exp 54 (Cluster TE 추가) | 8.7428 | 10.188 | ❌ OOF↑ LB↓ |
| Exp 55 (log1p 제거) | 8.7800 | 10.240 | ❌ OOF↓ LB↓ |
| Exp 56 (피처 제거 단독) | 8.7423 | 10.233 | ❌ OOF↑ LB↓ |
| **Exp 57** | 8.7465 | **10.162** | ✅ |
| Exp 68 (혼합 변환) | **8.7234** | 10.189 | ❌ OOF↑↑ LB↓ |

**원인**: GroupKFold CV 구조가 실제 Test(Unseen Layout) 분포를 완전히 반영하지 못함. 특히 역변환(sqrt → 제곱)처럼 오차를 증폭하는 구조는 OOF에서는 잘 잡히지 않음.

---

### 3. 가장 효과적인 피처 그룹

#### ① 시나리오 상대 변화 피처 (최대 단일 도약)

```python
# vs_start: 시나리오 시작점 대비 현재 비율/차이
first_val = df.groupby('scenario_id')[col].transform('first')
df[f'{col}_vs_start']    = df[col] / (first_val + 1e-6)
df[f'{col}_delta_start'] = df[col] - first_val

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
→ Exp 44(LB 10.48) → **Exp 45(LB 10.33)**, 단일 피처 그룹으로 **0.15 향상**

#### ② Lag / Rolling 시계열 피처

```python
# Lag(1~24) + Rolling(3~20) mean/std — 22개 SEQ_COLS 전체 적용
# sort-based approach로 속도 최적화 (groupby.transform 대비 10배 이상 빠름)
for lag in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 24]:
    df[f'{col}_lag{lag}'] = df.groupby('scenario_id')[col].shift(lag)

prev = df.groupby('scenario_id')[col].shift(1)
grp  = prev.groupby(df['scenario_id'])
for window in [3, 5, 7, 10, 14, 20]:
    df[f'{col}_roll{window}_mean'] = grp.rolling(window, min_periods=1).mean()...
```

#### ③ 모멘텀 피처

```python
for col in ['congestion_score', 'low_battery_ratio', 'robot_active']:
    df[f'{col}_vel'] = df.groupby('scenario_id')[col].diff(1)
```

#### ④ 구조적 교호작용 피처

```python
df['robot_utilization']    = df['robot_active'] / (df['robot_total'] + 1e-6)
df['charger_utilization']  = df['robot_charging'] / (df['charger_count'] + 1e-6)
df['aisle_pressure']       = df['congestion_score'] / (df['aisle_width_avg'] + 1e-6)
df['bottleneck_risk']      = df['congestion_score'] * df['intersection_density'] / (df['aisle_width_avg'] + 1e-6)
df['pack_station_pressure']= df['order_inflow_15m'] / (df['pack_station_count'] + 1e-6)
```

---

## 🧪 실험 전체 이력

### Phase 1: 기반 구축 (Exp 9~31)

| 실험 | 핵심 시도 | OOF MAE | LB 점수 | 결과 |
|------|---------|---------|---------|------|
| Baseline | 초기 모델 | 9.2405 | - | - |
| Exp 9 | Sample Weight (q90/q95/q99) | 8.9723 | - | ✅ |
| Exp 10 | Inverse MAE 앙상블 가중치 | 8.9722 | - | ✅ |
| Exp 11 | Lag/Rolling 피처 548개 확장 | 8.8116 | 10.7 | ✅ 최대 단일 개선 |
| Exp 12 | LGB Optuna 튜닝 + 앙상블 | 8.7846 | - | ✅ |
| Exp 13 | Stacking (Ridge meta) | 9.3834 | - | ❌ +0.60 하락 |
| Exp 16 | 모멘텀 피처 + Huber Loss | 8.7301 | - | ✅ |
| Exp 24 | 구조적 피처 추가 | 8.7558 | - | ✅ |
| Exp 25 | Ultimate Integration (619 피처) | 8.7297 | 17.8 | ❌ 누수 |
| Exp 29 | 시나리오 통계 통합 (648 피처) | 8.7214 | 17.8 | ❌ 누수 |
| Exp 30 | 누수 완전 제거 검증 | 9.6182 | 11.0 | ✅ 현실 파악 |
| Exp 31 | CV-safe TE (Smoothing k=30) | 9.0440 | 10.6 | ✅ 누수 제거 |

### Phase 2: 피처 고도화 (Exp 37~52)

| 실험 | 핵심 시도 | OOF MAE | LB 점수 | 결과 |
|------|---------|---------|---------|------|
| Exp 37 fixed | 튜닝 CAT + 3-way 앙상블 + 클리핑 | 8.9951 | 10.5 | ✅ |
| Exp 41 | 시간 기반 Sample Weight 추가 | 8.9832 | - | ✅ |
| Exp 42~44 | vs_start, delta_start (8→22개 컬럼) | 8.9345 | 10.48 | ✅ |
| **Exp 45** | **vs_cummax, vs_cummin 추가** | **8.8337** | **10.33** | ✅ **최대 도약** |
| Exp 46 | position_in_range 추가 | 8.8296 | - | ✅ |
| Exp 47~51 | Lag(4~15) + Rolling(7~20) 확장 | 8.7452 | - | ✅ |
| Exp 52 | Lag(20,24) 추가 + Rolling 전체 확장 | 8.7493 | **10.181** | ✅ LB 최초 도약 |

### Phase 3: 앙상블 고도화 (Exp 53~68)

| 실험 | 핵심 시도 | OOF MAE | LB 점수 | 결과 |
|------|---------|---------|---------|------|
| Exp 54 | Cluster TE 추가 | 8.7428 | 10.188 | ❌ LB 역전 |
| Exp 55 | log1p 변환 제거 | 8.7800 | 10.240 | ❌ |
| Exp 56 | Zero-importance 107개 제거 | 8.7423 | 10.233 | ❌ 단독 역효과 |
| **Exp 57** | **Exp 52 + 107개 제거 + 10-seed** | **8.7465** | **10.162** | ✅ **구조 확립** |
| Exp 58 | Asymmetric 1+10 seed | 8.7513 | 10.167 | ✅ 효율화 |
| Exp 59 | CatBoost Optuna 재튜닝 (GPU) | - | 10.180 | ❌ 환경 분리 오염 |
| **Exp 62** | **Colab GPU 단일환경 이식** | **8.7380** | **10.163** | ✅ **환경 통일** |
| **Exp 63** | **subsample=0.85 + seed 연동** | **8.7313** | **10.160** | ✅ |
| **Exp 64** | **subsample=0.75 (탐색)** | **8.7302** | **10.157** | ✅ |
| **Exp 65** | **colsample_bytree LGB=0.65, XGB=0.65** | **8.7286** | **10.156** | ✅ |
| **Exp 66** | **num_leaves=127** | **8.7250** | **10.156** | ✅ (LB 동일) |
| Exp 67 | 타겟 변환 비교 (log1p vs sqrt vs none) | 8.7247 | 10.156 | ➖ log1p 재확인 |
| Exp 68 | 모델별 혼합 변환 (LGB:sqrt, XGB/CAT:log1p) | 8.7234 | 10.189 | ❌ sqrt 역변환 오차 증폭 |

---

## 🏗️ 최종 파이프라인 (Exp 65~67 기준)

```
1. 데이터 로드 및 layout_info merge

2. 피처 엔지니어링 (446개)
   ├── 구조적 교호: robot_utilization, aisle_pressure, bottleneck_risk 등
   ├── 모멘텀: velocity (congestion, low_battery, robot_active)
   ├── 시간 인덱스: time_idx, time_ratio, steps_remaining
   ├── Lag(1~24) + Rolling(3~20) mean/std — 22개 SEQ_COLS
   ├── Expanding mean — 7개 핵심 컬럼
   ├── vs_start, delta_start — 22개 SEQ_COLS
   ├── vs_cummax, vs_cummin — 22개 SEQ_COLS
   └── position_in_range — 22개 SEQ_COLS
   * Zero-importance 107개 제거 후 최종 446개 사용

3. CV-safe Target Encoding (GroupKFold, n_splits=5)
   └── layout_id 기준, Smoothing k=30, Unseen → global mean fallback

4. 10-Seed 앙상블 (seeds: 42, 123, 2026, 777, 1004, 314, 555, 888, 999, 1337)
   ├── LightGBM  (Huber Loss,  num_leaves=127, subsample=0.75, colsample=0.65)
   ├── XGBoost   (MAE Loss,    max_depth=7,    subsample=0.75, colsample=0.65, device=cuda)
   └── CatBoost  (MAE Loss,    depth=7,        Optuna 튜닝 파라미터, task_type=GPU)

5. Inverse MAE 가중 앙상블 (≈ LGB:0.33 / XGB:0.33 / CAT:0.33 균등)

6. 후처리: np.maximum(preds, 0)  # 음수 클리핑

7. 타겟 변환: np.log1p → np.expm1 역변환
```

---

## ❌ 실패한 시도들

| 시도 | 원인 | 교훈 |
|------|------|------|
| Stacking (Ridge meta) | 모델 간 상관관계 높아 이득 없음, 역변환 오차 | 단순 가중 평균이 더 안정적 |
| Quantile Loss + 강화 가중치 | MAE 최적화와 반대 방향으로 편향 | 평가지표와 손실함수 일관성 중요 |
| layout_info 구조 피처 대량 추가 | 기존 피처와 중복, 노이즈 | 피처 품질 > 피처 수량 |
| Seen/Unseen 분리 전문가 모델 | Seen 전용 학습 시 데이터 부족 | 데이터 분할은 신중하게 |
| Cluster TE (KMeans) | Unseen layout에서 의미 없는 노이즈로 작용 | layout 의존 피처는 항상 LB 역효과 위험 |
| log1p 제거 | 기존 파라미터가 log1p에 최적화된 상태에서 제거 | 변환과 파라미터는 함께 재튜닝해야 |
| sqrt 역변환 (Exp 68) | 제곱 연산으로 작은 오차가 증폭, Unseen에서 폭발 | 역변환 오차 증폭 위험 항상 체크 |
| CPU + GPU 환경 분리 | 부동소수점 차이로 OOF 계산 오염 (LGBM MAE 18.11 오류) | 반드시 단일 환경에서 전체 실험 |
| CatBoost Optuna 재튜닝 (Exp 59) | 환경 분리로 결과 오염 | 환경 통일 후 재시도 필요 |

---

## 💡 주요 발견사항

### Asymmetric Seed Ensemble
LGB와 XGB는 `subsample`, `colsample_bytree` 없이는 seed를 바꿔도 결과가 동일했다. 이 파라미터 추가 후 진짜 다양성이 생겼다.

```python
# 변경 전: 10 seed 전체에서 LGB iteration 고정 (사실상 1 seed)
# 변경 후: seed마다 다른 iteration → 실제 다양성 확보
lgb_params = {
    ...
    'subsample': 0.75,
    'colsample_bytree': 0.65,
    'subsample_freq': 1,
    'random_state': seed   # ← seed 연동
}
```

### Colab L4 GPU 환경 최적화
- **LightGBM**: CPU 유지 (GPU 모드는 256 bin 제한으로 결과 달라짐)
- **XGBoost**: `device='cuda'` (v3.2.0 기준)
- **CatBoost**: `task_type='GPU', devices='0'`

이 조합으로 로컬 CPU 대비 학습 시간 **5~6시간 → 2시간** 단축.

### subsample 최적값 탐색 결과

```
subsample=0.70 → OOF MAE: 8.7350
subsample=0.75 → OOF MAE: 8.7297 ← 최적
subsample=0.80 → OOF MAE: 8.7325
subsample=0.85 → OOF MAE: 8.7342
subsample=0.90 → OOF MAE: 8.7364
```

### 타겟 변환 모델별 최적값

```
LGB  → sqrt  가 개별 OOF 8.7376 (log1p 8.7541보다 좋음)
XGB  → log1p 가 개별 OOF 더 좋음
CAT  → log1p 필수 (sqrt 적용 시 8.8453으로 폭락)
```

단, 혼합 적용 시 sqrt 역변환 오차 증폭으로 LB 역효과. **일관된 단일 변환 사용 권장.**

---

## 🛠️ 환경

```
Python 3.x
lightgbm
xgboost==3.2.0
catboost
scikit-learn
pandas, numpy
optuna
Google Colab (L4 GPU)
```

---

## 📁 파일 구조

```
├── experiments_log.md     # 전체 실험 기록
├── README.md              # 본 문서
├── experiment_52.py       # LB 최초 10.18 돌파 코드
├── experiment_57.py       # 10-seed 앙상블 구조 확립
├── experiment_62.ipynb    # Colab GPU 환경 이식
├── experiment_63.ipynb    # subsample 다양성 추가
├── experiment_64.ipynb    # subsample=0.75 확정
├── experiment_65.ipynb    # colsample 독립 탐색
├── experiment_66.ipynb    # num_leaves=127 확정
└── experiment_67.ipynb    # 타겟 변환 비교 (최종 채택: log1p)
```

---

## 📊 성능 개선 흐름

```
Baseline (9.24) 
  → 피처 확장 (8.81, LB 10.7)
  → 누수 제거 + CV-safe TE (LB 10.6)
  → 튜닝 CAT + 3-way 앙상블 (LB 10.5)
  → vs_cummax/cummin 피처 (LB 10.33) ← 최대 단일 도약
  → Lag 확장 (LB 10.181)
  → Zero-imp 제거 + 10-seed (LB 10.162)
  → Colab GPU + seed 다양성 (LB 10.160)
  → subsample/colsample 최적화 (LB 10.156) ← 현재
```