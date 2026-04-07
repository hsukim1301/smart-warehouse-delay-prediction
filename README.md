# 스마트 창고 출고 지연 예측 프로젝트 (v1)

이 프로젝트는 데이콘(Dacon)의 "스마트 창고 출고 지연 예측 AI 경진대회"를 위한 베이스라인 모델 구축 과정입니다. 창고의 다양한 운영 변수와 물리적 구조 정보를 결합하여 향후 30분간의 평균 출고 지연 시간을 예측합니다.

## 1. 프로젝트 개요
- **목표**: 창고 내 로봇 상태, 주문량, 환경 지표 등을 활용하여 출고 지연 시간(MAE) 최소화
- **데이터 구조**: 12,000개 시나리오 × 25개 타임슬롯 = 총 300,000행의 시계열성 정형 데이터
- **평가 지표**: Mean Absolute Error (MAE)

## 2. 주요 데이터 분석 결과 (EDA)
- **타겟 변수 (`avg_delay_minutes_next_30m`)**:
  - 왜도(Skewness): **5.68** (극심한 우측 꼬리 분포 → `log1p` 변환 적용)
  - 지연 없음(0) 비율: **2.73%**
- **상관관계 분석**:
  - **양의 상관관계**: 배터리 부족 비율(`low_battery_ratio`), 주문 유입량(`order_inflow_15m`), 충전 중 로봇 수(`robot_charging`)가 지연의 주요 원인
  - **음의 상관관계**: 평균 배터리 잔량(`battery_mean`), 대기 로봇 수(`robot_idle`)가 높을수록 지연 감소

## 3. 피처 엔지니어링 전략
- **레이아웃 정보 활용**: 창고 면적 대비 설비 밀도(`robot_density`, `pack_station_density`) 생성
- **시계열 래그(Lag) 피처**: 
  - 각 시나리오 내에서 직전 시점(T-1, T-2)의 상태값 반영
  - 최근 3개 타임슬롯의 이동 평균 및 최댓값(`roll3_mean`, `roll3_max`) 추가
- **교호작용 피처**: 운영 변수와 창고 구조의 결합(`battery_x_packdensity` 등)

## 4. 모델링 및 검증 전략
- **모델**: LightGBM Regressor
- **목적 함수**: `regression_l1` (평가지표인 MAE에 직접 최적화)
- **검증 방법**: **Scenario-based 5-Fold Cross Validation**
  - 동일한 시나리오의 데이터가 Train과 Validation에 섞이지 않도록 시나리오 아이디를 기준으로 분할하여 데이터 누수(Leakage) 방지
- **전처리**: 타겟 변수에 `np.log1p` 변환 적용 후, 예측 단계에서 `np.expm1`로 복원

## 5. 주요 파일 구성
- `main.py`: 전체 프로세스(전처리~제출파일 생성) 통합 파이썬 스크립트
- `Baseline_지연_예측.ipynb`: 가독성 및 시각화 중심의 Jupyter Notebook 베이스라인
- `analysis_report.txt`: 타겟 분포 및 상관관계 분석 결과 리포트

## 6. 실행 방법
```bash
python main.py
```
실행 완료 후 `submission.csv` 파일이 생성됩니다.
