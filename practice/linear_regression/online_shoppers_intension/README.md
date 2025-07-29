# 구매 전환 예측 실험 (Purchase Conversion Prediction)

## 📋 프로젝트 개요

온라인 쇼핑몰에서 사용자의 구매 전환 여부를 예측하는 머신러닝 실험입니다.

### 목표
- 사용자가 웹사이트를 방문했을 때 구매할지 안할지 예측
- 다양한 머신러닝 모델의 성능 비교
- 불균형 데이터 처리 방법 적용

### 타겟 변수
- `Revenue`: 구매 여부 (True/False)
- 구매 전환율: 약 15.47% (불균형 데이터)

## 🗂️ 데이터셋 정보

### Online Shoppers Purchasing Intention Dataset
- **총 샘플 수**: 12,330개 세션
- **특성 수**: 18개 (수치형 10개 + 범주형 8개)
- **타겟 변수**: Revenue (구매 여부)

### 주요 특성
#### 수치형 특성 (10개)
1. `Administrative` - 행정 관련 페이지 수
2. `Administrative_Duration` - 행정 페이지 체류 시간
3. `Informational` - 정보 제공 페이지 수
4. `Informational_Duration` - 정보 제공 페이지 체류 시간
5. `ProductRelated` - 상품 관련 페이지 수
6. `ProductRelated_Duration` - 상품 관련 페이지 체류 시간
7. `BounceRates` - 이탈률 (0~1)
8. `ExitRates` - 종료 페이지 비율 (0~1)
9. `PageValues` - 페이지 평균 기여도
10. `SpecialDay` - 기념일과의 근접성 (0~1)

#### 범주형 특성 (8개)
1. `Month` - 방문 월
2. `OperatingSystems` - 운영체제
3. `Browser` - 브라우저
4. `Region` - 지역
5. `TrafficType` - 유입 트래픽 유형
6. `VisitorType` - 방문자 유형
7. `Weekend` - 주말 방문 여부
8. `Revenue` - 구매 여부 (타겟 변수)

## 🚀 사용법

### 1. 환경 설정
```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비
- `online_shoppers_intention.csv` 파일을 프로젝트 디렉토리에 위치

### 3. 실험 실행

#### 방법 1: 전체 실험 실행
```bash
python run_experiment.py
```

#### 방법 2: 개별 실험 실행
```python
from purchase_conversion_experiment import PurchaseConversionExperiment

# 실험 객체 생성
experiment = PurchaseConversionExperiment("online_shoppers_intention.csv")

# 전체 실험 실행
results = experiment.run_experiment()

# 개별 단계 실행
experiment.load_data()
experiment.explore_data()
experiment.preprocess_data()
experiment.handle_imbalanced_data()
experiment.train_models()
experiment.plot_results()
```

## 🔬 실험 구성

### 1. 데이터 전처리
- 범주형 변수 인코딩 (LabelEncoder)
- 특성 스케일링 (StandardScaler)
- 데이터 분할 (train/test: 80/20)

### 2. 불균형 데이터 처리
- **SMOTE**: 소수 클래스 오버샘플링
- **RandomUnderSampler**: 다수 클래스 언더샘플링

### 3. 모델 비교
1. **LogisticRegression** - 기본 선형 모델
2. **RandomForest** - 앙상블 모델 (불균형 데이터에 강함)
3. **XGBoost** - 고성능 부스팅 모델
4. **SVM** - 비선형 관계 처리
5. **NeuralNetwork** - 복잡한 패턴 학습

### 4. 하이퍼파라미터 튜닝
- **GridSearchCV** 사용
- **RandomForest**: n_estimators, max_depth, min_samples_split, min_samples_leaf
- **XGBoost**: n_estimators, max_depth, learning_rate, subsample

### 5. 성능 평가 지표
- **Accuracy**: 전체 정확도
- **Precision**: 정밀도 (구매 예측 중 실제 구매 비율)
- **Recall**: 재현율 (실제 구매 중 예측 성공 비율)
- **F1-Score**: 정밀도와 재현율의 조화평균
- **ROC-AUC**: ROC 곡선 아래 면적

## 📊 결과 분석

### 시각화
- 상관관계 히트맵
- 타겟 변수 분포
- 특성별 구매 전환율
- ROC 곡선 비교
- Precision-Recall 곡선
- 특성 중요도 분석
- 혼동 행렬

### 성능 비교
- 다양한 모델의 성능 지표 비교
- 하이퍼파라미터 튜닝 전후 성능 비교
- 최적 모델 선정

## 📁 파일 구조

```
online_shoppers_intension/
├── purchase_conversion_experiment.py  # 메인 실험 클래스
├── run_experiment.py                  # 실행 스크립트
├── requirements.txt                   # 패키지 요구사항
├── README.md                         # 프로젝트 설명서
├── sol01.ipynb                      # 기존 노트북
├── sol01.py                         # 기존 파이썬 파일
└── online_shoppers_intention.csv    # 데이터 파일
```

## 🔧 주요 기능

### PurchaseConversionExperiment 클래스

#### 메서드
- `load_data()`: 데이터 로드 및 기본 정보 확인
- `explore_data()`: 데이터 탐색 및 시각화
- `preprocess_data()`: 데이터 전처리
- `handle_imbalanced_data()`: 불균형 데이터 처리
- `train_models()`: 다양한 모델 학습
- `hyperparameter_tuning()`: 하이퍼파라미터 튜닝
- `plot_results()`: 결과 시각화
- `feature_importance_analysis()`: 특성 중요도 분석
- `confusion_matrix_analysis()`: 혼동 행렬 분석
- `run_experiment()`: 전체 실험 실행

## 📈 예상 결과

### 모델 성능 예상 순위
1. **XGBoost** (튜닝 후) - 가장 높은 성능
2. **RandomForest** (튜닝 후) - 안정적인 성능
3. **NeuralNetwork** - 복잡한 패턴 학습
4. **SVM** - 비선형 관계 처리
5. **LogisticRegression** - 기본 성능

### 중요 특성 예상
1. `PageValues` - 페이지 기여도
2. `ProductRelated` - 상품 관련 페이지 수
3. `BounceRates` - 이탈률
4. `ExitRates` - 종료 페이지 비율
5. `Administrative` - 행정 페이지 수

## 🛠️ 문제 해결

### 일반적인 오류
1. **데이터 파일 없음**: `online_shoppers_intention.csv` 파일 확인
2. **패키지 설치 오류**: `pip install -r requirements.txt` 재실행
3. **메모리 부족**: 데이터 샘플링 또는 모델 파라미터 조정

### 성능 개선 팁
1. **특성 엔지니어링**: 새로운 특성 생성
2. **앙상블**: 여러 모델 조합
3. **데이터 증강**: 더 많은 샘플 수집
4. **하이퍼파라미터 최적화**: Bayesian Optimization 사용

## 📝 라이센스

이 프로젝트는 교육 목적으로 제작되었습니다.

## 🤝 기여

버그 리포트나 개선 제안은 언제든 환영합니다! 