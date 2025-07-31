"""
머신러닝 모델 학습 모듈
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .utils import setup_environment

class ModelTrainer:
    """머신러닝 모델 학습 클래스"""
    
    def __init__(self):
        setup_environment()
    
    def train_prediction_model(self, df, target_column=None, test_size=0.2):
        """예측 모델 학습 및 평가"""
        try:
            # 데이터 준비
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['Unnamed: 0', 'index']
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # 타겟 컬럼이 지정되지 않은 경우 첫 번째 가격 컬럼 사용
            if target_column is None:
                price_cols = [col for col in feature_cols if any(x in col.lower() for x in ['btc', 'price'])]
                target_column = price_cols[0] if price_cols else feature_cols[0]
            
            # 타겟 컬럼이 피처에 포함되어 있으면 제거
            if target_column in feature_cols:
                feature_cols.remove(target_column)
            
            # 결측값 처리
            df_clean = df[feature_cols + [target_column]].dropna()
            
            if len(df_clean) < 50:
                print("모델 학습을 위한 충분한 데이터가 없습니다.")
                return None, None, None
            
            X = df_clean[feature_cols]
            y = df_clean[target_column]
            
            # 시계열 데이터 분할
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # 데이터 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 모델 학습
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            
            # 예측
            y_pred = model.predict(X_test_scaled)
            
            # 성능 평가
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            print(f"\n모델 성능:")
            print(f"MSE: {mse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"R²: {r2:.4f}")
            print(f"RMSE: {rmse:.6f}")
            
            # 특성 중요도
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print(f"\n특성 중요도:")
            for _, row in feature_importance.iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
            
            return model, scaler, feature_cols
            
        except Exception as e:
            print(f"모델 학습 오류: {e}")
            return None, None, None
    
    def evaluate_model(self, model, scaler, X_test, y_test, feature_cols):
        """모델 평가"""
        if model is None:
            print("모델이 없습니다.")
            return None
        
        try:
            # 예측
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            
            # 성능 지표
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # 결과 시각화
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 실제 vs 예측
            axes[0,0].scatter(y_test, y_pred, alpha=0.6)
            axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            axes[0,0].set_xlabel('실제값')
            axes[0,0].set_ylabel('예측값')
            axes[0,0].set_title('실제 vs 예측')
            
            # 시계열 예측 결과
            axes[0,1].plot(y_test.values, label='실제', alpha=0.7)
            axes[0,1].plot(y_pred, label='예측', alpha=0.7)
            axes[0,1].set_title('시계열 예측 결과')
            axes[0,1].legend()
            
            # 잔차 분포
            residuals = y_test.values - y_pred
            axes[1,0].hist(residuals, bins=30, alpha=0.7)
            axes[1,0].set_title('잔차 분포')
            axes[1,0].set_xlabel('잔차')
            
            # 잔차 시계열
            axes[1,1].plot(residuals)
            axes[1,1].axhline(y=0, color='r', linestyle='--')
            axes[1,1].set_title('잔차 시계열')
            
            plt.tight_layout()
            plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'residuals': residuals
            }
            
        except Exception as e:
            print(f"모델 평가 오류: {e}")
            return None
    
    def get_feature_importance(self, model, feature_cols):
        """특성 중요도 분석"""
        if model is None:
            return None
        
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def predict_future(self, model, scaler, df, feature_cols, steps=5):
        """미래 예측"""
        if model is None:
            print("모델이 없습니다.")
            return None
        
        try:
            # 최근 데이터로 예측
            recent_data = df[feature_cols].tail(1)
            recent_scaled = scaler.transform(recent_data)
            
            predictions = []
            current_data = recent_scaled.copy()
            
            for _ in range(steps):
                pred = model.predict(current_data)[0]
                predictions.append(pred)
                
                # 다음 예측을 위해 데이터 업데이트 (간단한 방법)
                # 실제로는 더 복잡한 시계열 예측 로직이 필요
                current_data = np.roll(current_data, -1, axis=1)
                current_data[0, -1] = pred
            
            return predictions
            
        except Exception as e:
            print(f"미래 예측 오류: {e}")
            return None