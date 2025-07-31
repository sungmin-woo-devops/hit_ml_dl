"""
머신러닝 유틸리티 모듈
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import optuna
from scipy.stats import shapiro, jarque_bera
from typing import Dict, Tuple, List, Optional


def prepare_ml_data(df: pd.DataFrame, price_col: str) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """머신러닝을 위한 데이터 준비"""
    # 수치형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['Unnamed: 0', 'index']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols and col != price_col]
    
    if len(feature_cols) < 2:
        raise ValueError("최적화를 위한 충분한 피처가 없습니다.")
    
    # 타겟 변수 (다음 날 가격 예측)
    df_ml = df.copy()
    df_ml['target'] = df_ml[price_col].shift(-1)
    df_ml = df_ml.dropna()
    
    X = df_ml[feature_cols]
    y = df_ml['target']
    
    return df_ml, feature_cols, y


def create_objective_function(X_scaled: np.ndarray, y: pd.Series) -> callable:
    """Optuna 목적 함수 생성"""
    def objective(trial):
        # 하이퍼파라미터 탐색 범위
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        
        # 모델 생성
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        
        # 교차 검증으로 성능 평가
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        return scores.mean()
    
    return objective


def train_random_forest_model(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1
) -> Tuple[RandomForestRegressor, Dict[str, float], np.ndarray]:
    """랜덤 포레스트 모델 학습"""
    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 생성 및 학습
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # 성능 평가
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmse': rmse
    }
    
    return model, metrics, y_pred


def optimize_hyperparameters(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_trials: int = 50
) -> Tuple[Dict, float, pd.DataFrame]:
    """Optuna를 사용한 하이퍼파라미터 최적화"""
    # 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 목적 함수 생성
    objective = create_objective_function(X_scaled, y)
    
    # Optuna 스터디 생성 및 최적화 실행
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # 최적화 결과
    best_params = study.best_params
    best_score = study.best_value
    optimization_history = study.trials_dataframe()
    
    return best_params, best_score, optimization_history


def get_feature_importance(model: RandomForestRegressor, feature_cols: List[str]) -> pd.DataFrame:
    """특성 중요도 추출"""
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return feature_importance


def calculate_ensemble_predictions(
    model: RandomForestRegressor, 
    X_test_scaled: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """앙상블 예측의 평균과 표준편차 계산"""
    predictions = []
    for estimator in model.estimators_:
        pred = estimator.predict(X_test_scaled)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred


def calculate_learning_curve(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    n_estimators: int = 100
) -> Tuple[List[float], List[float], List[int]]:
    """학습 곡선 계산"""
    train_scores = []
    test_scores = []
    x_values = []
    
    for i in range(1, n_estimators, 10):  # 10개씩 샘플링
        partial_model = RandomForestRegressor(
            n_estimators=i,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        partial_model.fit(X_train_scaled, y_train)
        train_pred = partial_model.predict(X_train_scaled)
        test_pred = partial_model.predict(X_test_scaled)
        
        train_scores.append(r2_score(y_train, train_pred))
        test_scores.append(r2_score(y_test, test_pred))
        x_values.append(i)
    
    return train_scores, test_scores, x_values


def evaluate_model_performance(y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """모델 성능 평가"""
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmse': rmse
    }


def get_normality_test_results(returns: pd.Series) -> Dict[str, Dict[str, float]]:
    """정규성 검정 결과 계산"""
    # Shapiro-Wilk 검정
    shapiro_stat, shapiro_p = shapiro(returns)
    
    # Jarque-Bera 검정
    jb_stat, jb_p = jarque_bera(returns)
    
    return {
        'shapiro': {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'is_normal': shapiro_p > 0.05
        },
        'jarque_bera': {
            'statistic': jb_stat,
            'p_value': jb_p,
            'is_normal': jb_p > 0.05
        }
    }


def calculate_statistical_measures(returns: pd.Series, price_col: str) -> Dict[str, float]:
    """통계적 측정값 계산"""
    volatility = returns.std() * np.sqrt(252) * 100  # 연간 변동성
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    return {
        'volatility': volatility,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'var_95': var_95,
        'var_99': var_99
    }


def find_strong_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, any]]:
    """강한 상관관계 찾기"""
    strong_correlations = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                strong_correlations.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    return strong_correlations 