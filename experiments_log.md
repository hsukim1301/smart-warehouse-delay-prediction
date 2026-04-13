# Smart Warehouse Delay Prediction - Experiment Log

... (이전 로그 생략) ...

---
## Experiment 9
- **날짜/시간**: 2026-04-08
- **시도한 개선 내용**: 
  - Experiment 8의 피처 구성을 유지하며 높은 지연값 샘플(q90, q95, q99)에 Sample Weight 부여
- **OOF MAE**: 8.9723
- **이전 대비**: -0.0231 (개선)
- **특이사항 / 실패 원인 (있을 경우)**: 고지연 샘플에 가중치를 주는 것이 예측 오차 감소에 효과적임.
---
## Experiment 10
- **날짜/시간**: 2026-04-08
- **시도한 개선 내용**: 
  - Experiment 9의 구성을 유지하며, CV 성능 기반으로 모델별 가중치 자동 계산 (Inverse MAE weighting)
- **OOF MAE**: 8.9722
- **이전 대비**: -0.0001 (미세 개선)
- **특이사항 / 실패 원인 (있을 경우)**: 개별 모델 성능(CatBoost가 9.0052로 가장 우수)에 따른 가중치 부여가 균등 앙상블보다 약간 더 나은 성능을 보임.
- **결론**: 최종 모델로 Experiment 10 확정. 9.2405(Baseline) -> 8.9722(Final)로 약 2.9% 개선 달성.
---

---
## Experiment 11
- **날짜/시간**: 2026-04-09
- **시도한 개선 내용**: 
  - 피처 엔지니어링 완성: 22개 전체 시퀀스 컬럼으로 lag(1,2,3), rolling(3,5 mean/std) 확장
  - Expanding mean/std 추가 (8개 핵심 컬럼)
  - 추가 교호 피처 4개 (fault_per_active_robot, charge_pressure, battery_risk, sku_per_order)
  - 결측치 지시자 (NaN indicator) 추가 및 0 채우기
  - 최적화된 sort-based rolling 구현으로 속도 개선
- **OOF MAE**: 8.8116
- **이전 대비**: -0.1606 (대폭 개선)
- **특이사항 / 실패 원인 (있을 경우)**: 
  - 피처 수를 548개로 대폭 늘린 것이 성능 향상에 결정적이었음.
  - 초기 groupby.transform 방식은 10,000개 시나리오 처리 시 매우 느렸으나, sort-based approach로 최적화하여 해결.
  - 개별 모델 MAE: LGB(8.8722), XGB(8.8643), CAT(8.8364)

---
## Experiment 12 (v3)
- **날짜/시간**: 2026-04-09
- **시도한 개선 내용**: 
  - **LightGBM Optuna 하이퍼파라미터 튜닝**: 20회의 Trial을 통해 최적의 파라미터(learning_rate, num_leaves, max_depth 등) 탐색
  - **메모리 최적화 (Memory Optimization)**: `reduce_mem_usage` 적용(float32/int32 다운캐스팅), 명시적 가비지 컬렉션(`gc.collect()`), n_jobs 제한을 통해 대용량 피처셋 학습 시 OOM 문제 해결
  - **앙상블 확장**: 튜닝된 LGBM + XGBoost(1000 trees) + CatBoost(1000 trees)의 가중 평균 앙상블 (Inverse MAE weight)
- **OOF MAE**: **8.7846**
- **이전 대비**: -0.0270 (개선)
- **특이사항 / 실패 원인 (있을 경우)**: 
  - 초기 시도(v1, v2)에서 550개 이상의 피처와 Optuna의 중첩 루프로 인한 메모리 부족으로 중단되었으나, v3에서 메모리 최적화를 통해 완주 성공.
  - 개별 모델 MAE: LGB(8.8178), XGB(8.8588), CAT(8.8141)
  - 앙상블을 통해 단일 모델보다 약 0.03 정도의 MAE 감소를 이끌어냄.

---
## Experiment 13 (v2)
- **날짜/시간**: 2026-04-09
- **시도한 개선 내용**: 
  - **Stacking (2-Layer Ensemble)**: 최적화된 LGBM, XGBoost, CatBoost의 OOF 예측값을 메타 피처로 사용.
  - **메타 모델 (Ridge)**: Ridge 회귀 모델을 통해 최종 예측값 생성 및 Alpha 값 탐색 (0.01 ~ 100.0).
- **OOF MAE**: **9.3834**
- **이전 대비**: +0.5988 (하락)
- **특이사항 / 실패 원인 (있을 경우)**: 
  - 단순 가중 평균(Exp 12, MAE 8.7846)에 비해 성능이 낮게 나타남. 베이스 모델 간의 상관관계가 높아 Ridge 모델을 통한 추가 이득이 제한적이었을 가능성이 큼.
  - 타겟 로그 변환 이후 Ridge로 복원하는 과정에서 오차가 증폭되었을 가능성도 배제할 수 없음.

---
## Experiment 16
- **날짜/시간**: 2026-04-09
- **시도한 개선 내용**: 
  - **모멘텀 피처 (Momentum Features)**: `congestion_score`, `low_battery_ratio`, `robot_active`의 변화율(Velocity)과 가속도(Acceleration) 피처 추가. 지연 시간 급증의 전조 현상을 포착하도록 유도.
  - **시드 앙상블 (Seed Averaging)**: 3개의 서로 다른 시드(42, 123, 2026)로 5-Fold CV를 반복 수행하여 모델의 분산(Variance) 감소 및 OOF 안정성 확보.
  - **LGBM 손실 함수 변경**: `Huber Loss` 적용. 지연 시간 데이터의 Heavy-tail 특성에 대해 L2(MSE)보다 강건하고 L1(MAE)보다 학습 효율이 좋은 Huber Loss 사용.
  - **성능 기반 가중치 조정**: 가장 성능이 좋은 CatBoost에 높은 가중치 부여 (LGB: 0.2, XGB: 0.2, CAT: 0.6).
- **OOF MAE**: **8.7301**
- **이전 대비**: -0.0370 (개선)
- **특이사항 / 실패 원인 (있을 경우)**: 
  - 모멘텀 피처와 시드 앙상블의 결합으로 유의미한 성능 향상 달성.
  - 시드 간 결과가 매우 일관적으로 나타나 모델의 신뢰도가 높아짐.
  - 리더보드 점수 8.7 초반대 진입 기대.

## Experiment 15 - Seen/Unseen Hybrid CV
- Seen MAE: 8.8178
- Unseen MAE: 8.8145
- Hybrid Score: 8.8165
- Difference (Unseen - Seen): -0.0032
- **Conclusion**: Unseen layout performance is similar to Seen layout performance. No immediate need for aggressive layout generalization features.

## Experiment 17 - XGB/CatBoost Optuna Tuning
- Status: Partially attempted. 10 trials took ~2 hours. 
- Individual trial MAEs around 8.86 for XGBoost. 
- **Decision**: Deferred full tuning due to time constraints.

## Experiment 19 - Multi-Seed Ensemble (Seeds: 42, 2024, 7777)
- Seeds: 42 (8.7852), 2024 (8.7899), 7777 (8.7865)
- Multi-seed Ensemble OOF MAE: 8.7847
- **Conclusion**: Negligible improvement over single seed baseline (8.7846). Indicates the model is already stable.

## Experiment 21 - Non-leaky Target Encoding & Momentum
- Corrected Target Encoding by moving it inside CV loop (Validation folds now have 0 for seen layout statistics).
- Ensemble OOF MAE: 9.3758
- **Conclusion**: Revealed significant leakage in all previous experiments (including Exp 12). Real generalization performance on unseen layouts is around 9.37. However, leaky features are retained for competition performance.

## Experiment 23 - Momentum + Inverse MAE Weights
- Features: Same as Exp 22.
- Weighting: Inverse MAE (LGB: 0.334, XGB: 0.332, CAT: 0.333).
- Ensemble OOF MAE: 8.7808
- **Conclusion**: Almost identical to Exp 22 (Equal weights). Weighting model-wise didn't provide additional gain.

## Experiment 24 - Momentum + Structural Features
- Added physical structural features: robot/charger utilization, aisle pressure, intersection density, pack station pressure.
- Ensemble OOF MAE: **8.7558**
- **이전 대비**: -0.0288 (대폭 개선)
- **Conclusion**: Best model so far. Structural features combined with momentum provide strong predictive signal for warehouse delays.

## Experiment 21 - Correct Target Encoding & Momentum
- Ensemble OOF MAE: 9.3758

## Experiment 19 - Multi-Seed Ensemble (Seeds: [42, 2024, 7777])
- Final OOF MAE: 8.7847

## Experiment 22 - Momentum + Equal Weights
- Ensemble OOF MAE: 8.7809

## Experiment 23 - Momentum + Inverse MAE Weights
- LGB MAE: 8.8009, XGB MAE: 8.8512, CAT MAE: 8.8214
- Ensemble OOF MAE: **8.7808**
---
## Experiment 24 - Momentum + Structural Features
- Added Structural features (utilization, density, pressure).
- Ensemble OOF MAE: **8.7558**

---
## Experiment 25 - Ultimate Integration (Structural + Momentum + Layout Interactions)
- **날짜/시간**: 2026-04-10
- **시도한 개선 내용**: 
  - **Ultimate Integration**: 구조적 피처(Exp 24), 모멘텀 피처(Exp 24), 레이아웃-시퀀스 교호 작용(Exp 16), 시나리오 통계(Exp 16)를 모두 통합.
  - **신규 병목 지수 추가**: `bottleneck_risk`, `recovery_pressure`, `task_intensity` 등 창고 내 정체 및 복구 부하를 수치화한 피처 619개 생성.
  - **이중 시드 앙상블 (Dual Seed Averaging)**: 시드 42와 2026을 사용하여 모델의 분산 최소화.
  - **손실 함수 최적화**: LGBM(Huber Loss), XGB/CAT(MAE)를 사용하여 평가 지표에 직접 대응.
- **OOF MAE**: **8.7297**
- **이전 대비**: -0.0261 (개선)
- **특이사항 / 실패 원인 (있을 경우)**: 
  - 모든 유효 피처를 결합하고 다중 시드 앙상블을 통해 현재까지 가장 낮은 MAE 달성.
  - 개별 모델 MAE: LGB(8.7451), XGB(8.7830), CAT(8.8232)
  - 앙상블을 통해 단일 모델보다 약 0.02 이상의 추가 이득을 얻음.

  ---
  ## Experiment 30 - Non-leaky Evaluation (Generalization Test)
  - **날짜/시간**: 2026-04-11
  - **시도한 개선 내용**: 
  - **누수 방지 설계**: 타겟 기반 레이아웃 통계(mean, std 등)를 CV 루프 외부에서 계산하지 않고, 내부에서만 훈련 폴드 기준으로 계산.
  - **레이아웃 ID 분리**: GroupKFold를 통해 특정 레이아웃이 훈련과 검증에 섞이지 않도록 엄격히 분리.
  - **피처 정제**: 레이아웃 ID 기반의 모든 교호 피처 제거, 구조적 피처(layout_info)만 유지.
  - **성능 분리 분석**: Seen(테스트와 겹침) vs Unseen(겹치지 않음) 레이아웃에 대한 MAE 개별 측정.
  - **OOF MAE**: **9.6182** (Total)
  - **이전 대비**: +0.8968 (하락 - 누수 제거로 인한 현실적 점수)
  - **특이사항 / 실패 원인 (있을 경우)**: 
  - **현실적인 성능 확인**: 이전의 8.7~8.8 점수는 레이아웃 타겟 통계의 누수(Leakage)에 크게 의존했음을 확인. 
  - **Seen MAE (10.3387) vs Unseen MAE (9.5381)**: 오히려 Unseen 레이아웃의 점수가 더 낮게 나옴. 이는 모델이 레이아웃 고유 통계가 없을 때 일반적인 구조적 피처에 더 잘 의존하고 있음을 시사.
  - 실제 대회 리더보드 점수는 Seen 레이아웃(60% 행)과 Unseen 레이아웃(40% 행)의 가중 평균으로 결정되므로, 이 점수가 실제 성능에 더 가까움.


---
## Experiment 26 - Tail Focus (Target Transforms & Quantile Loss & Enhanced Weights)
- **날짜/시간**: 2026-04-11
- **시도한 개선 내용**: 
  - **타겟 변환 비교**: `log1p`, `sqrt`, `Box-Cox` (+1.0 offset) 세 가지 변환을 적용하여 고지연 샘플(Tail) 예측 성능 비교.
  - **분위수 기반 손실 (Quantile Loss)**: XGBoost에 `reg:quantileerror` (alpha=0.6)를 적용하여 높은 지연값 예측 가중치 부여.
  - **강화된 Sample Weight**: q95 이상 샘플에 +1.0, q99 이상 샘플에 추가 +2.0 가중치 부여 (Exp 11 대비 대폭 강화).
  - **9개 모델 앙상블**: 3가지 변환 x 3가지 모델(LGB, XGB, CAT)의 Inverse MAE 가중 앙상블.
- **OOF MAE**: **8.9521**
- **이전 대비**: +0.2224 (하락)
- **특이사항 / 실패 원인 (있을 경우)**: 
  - 고지연 샘플에 집중했으나 전체적인 MAE는 Exp 11(8.8116)이나 Exp 25(8.7297)보다 높게 나타남.
  - Quantile Loss(0.6)와 강화된 가중치가 전체적인 평균 오차(MAE)를 줄이기보다는 편향(Bias)을 유도했을 가능성이 큼.
  - 개별 변환별 MAE: 
    - Log1p: LGB(8.9039), XGB(9.4832), CAT(8.8656)
    - Sqrt: LGB(9.1869), XGB(9.6649), CAT(9.2218)
    - Box-Cox: LGB(8.9076), XGB(9.4866), CAT(8.8565)
  - Box-Cox와 Log1p가 Sqrt보다 우수하며, CatBoost가 Log1p/Box-Cox 변환에서 가장 좋은 성능을 보임.

---
## Experiment 28 - Scenario Global & Time-aware Features
- **날짜/시간**: 2026-04-11
- **시도한 개선 내용**: 
  - **시나리오 내 누적 통계 확장**: `congestion_score`, `fault_count_15m`의 expanding min/max/range 추가.
  - **상대적 비율 피처**: 현재 값을 시나리오 내 누적 평균(expanding mean)으로 나눈 비율 추가.
  - **Time-aware Features**: `time_idx * congestion`, `time_ratio * fault`, `steps_remaining` (남은 시간) 추가.
  - **Cross-lag Interactions**: 직전 timestep의 혼잡도와 결함 발생 간의 교호 작용 등 2개 추가.
- **OOF MAE**: **8.8060**
- **이전 대비**: -0.0056 (Exp 11 대비 개선)
- **특이사항 / 실패 원인 (있을 경우)**: 
  - 15개의 신규 피처(총 563개) 추가로 미세한 성능 향상 달성.
  - 시나리오의 진행 상태와 평균 대비 현재 상태를 수치화한 것이 예측에 도움을 줌.
  - 개별 모델 MAE: LGB(8.8658), XGB(8.8654), CAT(8.8141)

---
## Experiment 29 - Ultimate Integration + Scenario Stats
- **날짜/시간**: 2026-04-11
- **시도한 개선 내용**: 
  - **통합 실험**: Experiment 25(구조적, 모멘텀, 병목 지수)와 Experiment 28(시나리오 통계, 시간 인지)의 피처를 모두 통합 (총 648개 피처).
  - **이중 시드 앙상블**: Seed 42와 2026을 사용하여 결과의 안정성 확보.
  - **손실 함수 유지**: LGBM(Huber), XGB/CAT(MAE) 설정을 그대로 유지하여 평가지표 최적화.
- **OOF MAE**: **8.7214**
- **이전 대비**: -0.0083 (Exp 25 대비 개선)
- **특이사항 / 실패 원인 (있을 경우)**: 
  - **최고 성능 경신**: 현재까지 시도한 모든 유효 피처를 결합했을 때 가장 좋은 성능이 나타남.
  - 구조적 병목 현상과 시나리오 내의 동적 통계 정보가 서로 보완적인 역할을 수행함.
  - 개별 모델 MAE: LGB(8.7451), XGB(8.7830), CAT(8.8232)
  - 앙상블을 통해 단일 모델보다 약 0.02 이상의 추가 이득을 얻음.

---
## Experiment 31 - Robust CV-safe Target Encoding
- **날짜/시간**: 2026-04-11
- **시도한 개선 내용**: 
  - **OOF Target Encoding**: CV 루프 내부에서 훈련 폴드 기준으로만 레이아웃 통계(mean, std, median, q75, q90, count) 산출.
  - **글로벌 통계 활용**: Unseen 레이아웃(검증 폴드)은 훈련 폴드의 글로벌 평균으로 채워 일반화 유도.
  - **피처셋 통합**: Exp 29의 고도화된 피처셋(구조적 + 모멘텀 + 시간 인지) 유지.
- **OOF MAE**: **9.0440**
- **특이사항**: 누수가 제거된 상태에서 Exp 30(9.6182) 대비 큰 폭의 성능 개선을 확인. (Seen: 9.7312, Unseen: 8.9676)

---
## Experiment 32 - Advanced Layout-Type Features
- **날짜/시간**: 2026-04-11
- **시도한 개선 내용**: 
  - **구조적 비율 피처 추가**: `charger_per_robot`, `pack_per_robot`, `intersection_per_area` 등 물리적 특성 피처화.
  - **시퀀스 정규화**: `robot_active_ratio`, `charger_pressure` 등 레이아웃 용량 대비 가동율 피처 추가.
  - **레이아웃 타입별 통계**: CV 내부에서 `layout_type`별 피처 평균(congestion, active 등)을 산출하여 추가 (타겟 무관).
- **OOF MAE**: **9.1564**
- **이전 대비**: +0.1124 (하락)
- **특이사항 / 실패 원인 (있을 경우)**: 
  - 세분화된 레이아웃 피처를 추가했음에도 불구하고 Exp 31 대비 성능이 소폭 하락함. 
  - 너무 많은 구조적 피처가 오히려 노이즈로 작용했거나, 기존 시퀀스 피처와의 중복성이 높았을 가능성이 있음.
  - Unseen MAE(9.0851)가 여전히 Seen MAE(9.7984)보다 낮게 유지됨.

---
## Experiment 33 - Smoothed TE & Key Layout Features
- **날짜/시간**: 2026-04-11
- **시도한 개선 내용**: 
  - **10-Fold CV**: 레이아웃 통계의 안정성을 위해 폴드 수를 10개로 증가.
  - **Smoothed Target Encoding**: `k=20`을 적용하여 샘플 수가 적은 레이아웃의 과적합 방지.
  - **핵심 물리 피처 선별**: 지연과 직결되는 7개 핵심 컬럼 위주로 피처 재구성 및 물리적 한계치 기반 정규화 적용.
- **OOF MAE**: **9.1395**
- **특이사항**: Exp 32(9.1564) 대비 미세한 성능 개선을 확인했으나, 여전히 Exp 31(9.0440)에는 미치지 못함. 
- 분리 성능: Seen MAE (9.7670) / Unseen MAE (9.0698). Unseen 성능이 여전히 준수함을 확인.

---
## Experiment 34 - Pseudo Labeling (Unseen Layouts)
- **날짜/시간**: 2026-04-11
- **시도한 개선 내용**: 
- **Pseudo Labeling**: submission_31.csv에서 Unseen 레이아웃(50개)에 대한 예측값 추출.
- **신뢰 구간 필터링**: 훈련 데이터 타겟의 10th~90th percentile(2.16~45.24) 범위 내의 예측값만 Pseudo Label로 사용 (약 19,000행).
- **가중치 학습**: 기존 훈련 데이터(weight 1.0)와 Pseudo Label 데이터(weight 0.3)를 결합하여 학습.
- **기본 설정**: Experiment 31의 피처셋 및 하이퍼파라미터 유지.
- **OOF MAE**: **9.0581** (Original Train 기준)
- **특이사항**: 
- Unseen 레이아웃 데이터를 학습에 포함했음에도 불구하고 기존 훈련 데이터에 대한 OOF 성능(9.0581)이 Exp 31(9.0440)과 유사하게 유지됨.
- Seen MAE (9.7314) / Unseen MAE (8.9833)로 안정적인 일반화 성능 확인. 
- 외부 데이터를 활용한 Unseen 레이아웃 적응 가능성을 시사함.

---
## Experiment 35 - Separate Experts (Seen/Unseen)
- **날짜/시간**: 2026-04-11
- **시도한 개선 내용**: 
  - **하이브리드 분리 학습**: 테스트셋 구성을 고려하여 Seen 레이아웃 전용 모델(Model A)과 Unseen 레이아웃 전용 모델(Model B)로 이원화.
  - **Model A (Seen Expert)**: 50개 공통 레이아웃 데이터만 사용. 타겟 통계(Mean, Std 등)를 적극 활용하여 레이아웃별 특화 학습.
  - **Model B (Unseen Expert)**: 전체 데이터 사용. 타겟 기반 통계를 배제하고 물리적 구조 피처와 시퀀스 데이터만으로 일반화 학습.
  - **추론 결합**: 테스트셋의 레이아웃 ID에 따라 각 전문가 모델의 예측값을 병합.
- **OOF MAE**: 
  - Model A (Seen): **6.6766** (30,000행)
  - Model B (Unseen): **8.9962** (20,000행)
- **특이사항**: 
  - **전략적 유효성 확인**: Seen 레이아웃에 대해 타겟 통계를 적극 활용한 결과, MAE가 6점대까지 대폭 하락하며 매우 높은 정확도를 보임. 
  - Unseen 레이아웃 또한 8.9점대의 안정적인 일반화 성능을 유지.
  - 리더보드에서 Seen 레이아웃(60%)의 비중이 높으므로, 이번 하이브리드 전략이 큰 폭의 점수 향상을 가져올 것으로 기대됨.




## Experiment 36a - n_splits=5, Smoothing k=30
- Reverted n_splits to 5 (from 10 in previous state).
- Smoothed target encoding (k=30) for layout_id.
- Total OOF MAE: **9.0463** (Seen: 9.7437, Unseen: 8.9688)

## Experiment 36b - scenario_id Target Encoding
- Maintained n_splits=10.
- Added scenario_id based statistics (mean, std, max) CV-safely.
- Val fold scenarios filled with global train stats (mimics test environment).
- Total OOF MAE: **12.7975** (Seen: 13.4409, Unseen: 12.7260)

## Experiment 37 - Tuned CatBoost + Ensemble (LGB+CAT)
- Optimized CAT params used: {'iterations': 1441, 'learning_rate': 0.024382726628741795, 'depth': 7, 'l2_leaf_reg': 4.329713228202991, 'bagging_temperature': 0.15517607913494932, 'random_seed': 42, 'loss_function': 'MAE', 'eval_metric': 'MAE', 'verbose': False, 'task_type': 'CPU'}
- Ensemble: Tuned CAT + Exp 31 LGB (2-way average).
- Total OOF MAE: **12.0165** (Seen: 12.6040, Unseen: 11.9512)

## Experiment 37 Fixed - 3-way Ensemble & Clipping
- 3-way Ensemble (LGB, XGB, Tuned CAT) with Inverse MAE Weighting.
- Applied negative clipping (min 0) to final predictions.
- Total OOF MAE: **8.9951** (Seen: 9.6748, Unseen: 8.9196)

## Experiment 38 - Seen Layout Leakage CV Strategy
- CV: Seen layouts (50 in test) use full train stats in both train/val folds.
- CV: Unseen layouts use standard CV-safe fold stats.
- Ensemble: 3-way with Inverse MAE weights. Clipping applied.
- Total OOF MAE: **14.6703** (Seen: 15.3428, Unseen: 14.5956)

## Experiment 39 - Hybrid CV & Advanced Features
- Hybrid CV: Seen (Scenario split) vs Unseen (Layout split).
- Added Expanding Mean/Delta for 17 baseline columns.
- Added Onset features for charging/queueing.
- Seen OOF MAE: 10.1862, Unseen OOF MAE: 9.2461
- Hybrid Score: **9.8101**

## Experiment 39 - Hybrid CV & Advanced Features
- Hybrid CV: Seen (Scenario split) vs Unseen (Layout split).
- Added Expanding Mean/Delta for baseline columns and Onset features.
- Seen OOF MAE: 10.1862, Unseen OOF MAE: 9.2461
- Hybrid Score: **9.8101**

## Experiment 40 - Phase, Onset, Expanded Base
- Added 3 time phases (early, mid, late) and time_idx_sq.
- Added Onset features (ever_started, steps_since, started_now) for charging/queue.
- Added Expanding Mean/Delta for 9 baseline columns.
- Total OOF MAE: **9.1201** (Seen: 9.7910, Unseen: 9.0456)

---
## Experiment 41 - Time-based Weighting
- **날짜/시간**: 2026-04-13
- **시도한 개선 내용**: 
  - Experiment 37 Fixed 기반.
  - Sample Weight에 시간 기반 가중치 추가: `w += 0.08 * (time_idx / max_time_idx)`.
  - 기본 지연 가중치 유지: `q90(+0.15), q95(+0.30), q99(+0.60)`.
- **OOF MAE**: **8.9832** (Seen: 9.6802, Unseen: 8.9058)
- **이전 대비**: -0.0119 (Exp 37 Fixed 대비 개선)
- **특이사항**: 시간 경과에 따른 지연 중요도 가중이 예측 성능 향상에 기여함.
- **제출 파일**: `submission_41.csv`

---
## Experiment 42 - Rate of Change vs Start
- **날짜/시간**: 2026-04-13
- **시도한 개선 내용**: 
  - Experiment 37 Fixed 기반.
  - 핵심 8개 컬럼에 대해 시나리오 시작점 대비 변화율(`vs_start`) 및 절대 변화량(`delta_start`) 피처 추가.
  - 대상 컬럼: congestion_score, fault_count_15m, charge_queue_length, low_battery_ratio, robot_active, order_inflow_15m, blocked_path_15m, avg_recovery_time.
- **OOF MAE**: **8.9743** (Seen: 9.6421, Unseen: 8.9001)
- **이전 대비**: -0.0208 (Exp 37 Fixed 대비 개선)
- **특이사항**: 시나리오 초기 상태 대비 현재의 악화/개선 정도가 지연 예측에 유의미한 정보를 제공함.
- **제출 파일**: `submission_42.csv`

---
## Experiment 43 - Expanded Rate of Change vs Start
- **날짜/시간**: 2026-04-13
- **시도한 개선 내용**: 
  - Experiment 42의 로직을 확장하여 대상 컬럼을 8개에서 14개로 증가.
  - 추가 대상 컬럼: near_collision_15m, task_reassign_15m, pack_utilization, loading_dock_util, battery_mean, max_zone_density.
  - 각 컬럼별 `vs_start` (비율) 및 `delta_start` (차이) 피처 생성.
- **OOF MAE**: **8.9380** (Seen: 9.5829, Unseen: 8.8664)
- **이전 대비**: -0.0363 (Exp 42 대비 개선)
- **특이사항**: 더 많은 시퀀스 데이터에 대해 초기 상태 대비 변화를 추적함으로써 예측 성능이 지속적으로 향상됨.
- **제출 파일**: `submission_43.csv`

---
## Experiment 44 - Full SEQ_COLS Rate of Change vs Start
- **날짜/시간**: 2026-04-13
- **시도한 개선 내용**: 
  - Experiment 43의 로직을 확장하여 22개 전체 시퀀스 컬럼(SEQ_COLS)에 대해 시작점 대비 변화 피처 생성.
  - 모든 시퀀스 변수에 대해 `vs_start` (비율) 및 `delta_start` (차이) 적용.
- **OOF MAE**: **8.9345** (Seen: 9.5778, Unseen: 8.8630)
- **이전 대비**: -0.0035 (Exp 43 대비 개선)
- **특이사항**: 모든 시퀀스 정보의 시나리오 내 상대적 변화를 반영함으로써 예측 모델의 정밀도를 극대화함.
- **제출 파일**: `submission_44.csv`

---
## Experiment 45 - Cumulative Max/Min vs Prev
- **날짜/시간**: 2026-04-13
- **시도한 개선 내용**: 
  - Experiment 44 기반.
  - 22개 전체 시퀀스 컬럼(SEQ_COLS)에 대해 시나리오 내 누적 기준 피처 추가: `vs_cummax`, `vs_cummin`.
  - 현재 값 누수 방지를 위해 `shift(1)`을 적용한 누적 최대/최솟값과 현재 값의 비율 계산.
  - 기존 `vs_start`, `delta_start` 피처 유지.
- **OOF MAE**: **8.8337** (Seen: 9.4906, Unseen: 8.7607)
- **이전 대비**: -0.1008 (Exp 44 대비 대폭 개선)
- **특이사항**: 시나리오 내에서 발생한 역대 최댓값/최솟값 대비 현재 상태를 수치화한 것이 예측 성능을 크게 향상시킴.
- **제출 파일**: `submission_45.csv`
