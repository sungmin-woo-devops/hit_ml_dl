#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
구매 전환 예측 실험 코드
Online Shoppers Purchasing Intention Dataset

목표: 사용자가 구매할지 안할지 예측 (Binary Classification)
타겟변수: Revenue (True/False)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class PurchaseConversionExperiment:
    """구매 전환 예측 실험 클래스"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: 데이터 파일 경로
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """데이터 로드 및 기본 정보 확인"""
        print("=== 데이터 로드 ===")
        self.df = pd.read_csv(self.data_path, encoding='utf-8')
        
        print(f"데이터 크기: {self.df.shape}")
        print(f"컬럼 수: {len(self.df.columns)}")
        print(f"타겟 변수 분포:")
        print(self.df['Revenue'].value_counts())
        print(f"구매 전환율: {self.df['Revenue'].mean():.4f} ({self.df['Revenue'].mean()*100:.2f}%)")
        
        return self.df
    
    def explore_data(self):
        """데이터 탐색 및 시각화"""
        print("\n=== 데이터 탐색 ===")
        
        # 수치형 변수 통계
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"수치형 변수({len(numeric_cols)}): {list(numeric_cols)}")
        
        # 범주형 변수 확인
        categorical_cols = self.df.select_dtypes(include=['object', 'bool']).columns
        print(f"범주형 변수({len(categorical_cols)}): {list(categorical_cols)}")
        
        # 상관관계 분석
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, fmt='.2f')
        plt.title('Feature Correlation Heatmap', fontsize=15)
        plt.tight_layout()
        plt.show()
        
        # 타겟 변수 분포 시각화
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 파이 차트
        revenue_counts = self.df['Revenue'].value_counts()
        axes[0].pie(revenue_counts.values, labels=['No Purchase', 'Purchase'], 
                   autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
        axes[0].set_title('Revenue Distribution')
        
        # 바 차트
        sns.countplot(data=self.df, x='Revenue', ax=axes[1])
        axes[1].set_title('Revenue Count')
        axes[1].set_xlabel('Revenue')
        axes[1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
        
        # 주요 특성과 타겟 변수의 관계
        important_features = ['Administrative', 'ProductRelated', 'BounceRates', 'ExitRates', 'PageValues']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(important_features):
            if i < len(axes):
                # Revenue별로 분리하여 히스토그램
                axes[i].hist(self.df[self.df['Revenue']==False][feature], alpha=0.7, 
                            label='No Purchase', bins=30, color='lightcoral')
                axes[i].hist(self.df[self.df['Revenue']==True][feature], alpha=0.7, 
                            label='Purchase', bins=30, color='lightblue')
                axes[i].set_title(f'{feature} vs Revenue')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_data(self):
        """데이터 전처리"""
        print("\n=== 데이터 전처리 ===")
        
        # 복사본 생성
        df_processed = self.df.copy()
        
        # 범주형 변수 인코딩
        le_month = LabelEncoder()
        le_visitor_type = LabelEncoder()
        
        df_processed['Month_encoded'] = le_month.fit_transform(df_processed['Month'])
        df_processed['VisitorType_encoded'] = le_visitor_type.fit_transform(df_processed['VisitorType'])
        df_processed['Weekend_encoded'] = df_processed['Weekend'].astype(int)
        df_processed['Revenue_encoded'] = df_processed['Revenue'].astype(int)
        
        # 특성 선택 (수치형 + 인코딩된 범주형)
        feature_columns = [
            'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
            'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 
            'PageValues', 'SpecialDay', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
            'Month_encoded', 'VisitorType_encoded', 'Weekend_encoded'
        ]
        
        X = df_processed[feature_columns]
        y = df_processed['Revenue_encoded']
        
        print(f"특성 수: {X.shape[1]}")
        print(f"샘플 수: {X.shape[0]}")
        
        # 데이터 분할
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 특성 스케일링
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"훈련 세트 크기: {self.X_train.shape}")
        print(f"테스트 세트 크기: {self.X_test.shape}")
        
        return X, y
    
    def handle_imbalanced_data(self, method='smote'):
        """불균형 데이터 처리"""
        print(f"\n=== 불균형 데이터 처리 ({method}) ===")
        
        if method == 'smote':
            smote = SMOTE(random_state=42)
            self.X_train_balanced, self.y_train_balanced = smote.fit_resample(
                self.X_train_scaled, self.y_train
            )
        elif method == 'undersample':
            rus = RandomUnderSampler(random_state=42)
            self.X_train_balanced, self.y_train_balanced = rus.fit_resample(
                self.X_train_scaled, self.y_train
            )
        else:
            self.X_train_balanced, self.y_train_balanced = self.X_train_scaled, self.y_train
        
        print(f"원본 훈련 세트 분포: {np.bincount(self.y_train)}")
        print(f"균형 조정 후 분포: {np.bincount(self.y_train_balanced)}")
        
        return self.X_train_balanced, self.y_train_balanced
    
    def train_models(self):
        """다양한 모델 학습"""
        print("\n=== 모델 학습 ===")
        
        # 모델 정의
        self.models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM': SVC(random_state=42, probability=True),
            'NeuralNetwork': MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 50))
        }
        
        # 모델 학습 및 평가
        for name, model in self.models.items():
            print(f"\n--- {name} 학습 중 ---")
            
            # 모델 학습
            if name in ['SVM', 'NeuralNetwork']:
                model.fit(self.X_train_balanced, self.y_train_balanced)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                model.fit(self.X_train_balanced, self.y_train_balanced)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # 성능 평가
            self.results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"Accuracy: {self.results[name]['accuracy']:.4f}")
            print(f"Precision: {self.results[name]['precision']:.4f}")
            print(f"Recall: {self.results[name]['recall']:.4f}")
            print(f"F1-Score: {self.results[name]['f1']:.4f}")
            print(f"ROC-AUC: {self.results[name]['roc_auc']:.4f}")
    
    def hyperparameter_tuning(self, model_name='RandomForest'):
        """하이퍼파라미터 튜닝"""
        print(f"\n=== {model_name} 하이퍼파라미터 튜닝 ===")
        
        if model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
        elif model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        else:
            print(f"{model_name}에 대한 튜닝은 구현되지 않았습니다.")
            return
        
        # Grid Search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train_balanced, self.y_train_balanced)
        
        print(f"최적 파라미터: {grid_search.best_params_}")
        print(f"최적 F1-Score: {grid_search.best_score_:.4f}")
        
        # 최적 모델로 예측
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test_scaled)
        y_pred_proba = best_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # 성능 평가
        self.results[f'{model_name}_Tuned'] = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"튜닝 후 성능:")
        print(f"Accuracy: {self.results[f'{model_name}_Tuned']['accuracy']:.4f}")
        print(f"Precision: {self.results[f'{model_name}_Tuned']['precision']:.4f}")
        print(f"Recall: {self.results[f'{model_name}_Tuned']['recall']:.4f}")
        print(f"F1-Score: {self.results[f'{model_name}_Tuned']['f1']:.4f}")
        print(f"ROC-AUC: {self.results[f'{model_name}_Tuned']['roc_auc']:.4f}")
    
    def plot_results(self):
        """결과 시각화"""
        print("\n=== 결과 시각화 ===")
        
        # 성능 비교
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        model_names = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in model_names]
            
            axes[i].bar(model_names, values, color='steelblue', alpha=0.7)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # 값 표시
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # ROC 곡선
        plt.figure(figsize=(10, 8))
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Precision-Recall 곡선
        plt.figure(figsize=(10, 8))
        for name, result in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, result['y_pred_proba'])
            plt.plot(recall, precision, label=f'{name}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def feature_importance_analysis(self):
        """특성 중요도 분석"""
        print("\n=== 특성 중요도 분석 ===")
        
        # RandomForest 특성 중요도
        if 'RandomForest' in self.models:
            rf_model = self.models['RandomForest']
            feature_names = self.X_train.columns
            
            importance = rf_model.feature_importances_
            indices = np.argsort(importance)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importance (Random Forest)')
            plt.bar(range(len(indices)), importance[indices])
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.show()
            
            print("상위 10개 중요 특성:")
            for i in range(min(10, len(indices))):
                print(f"{i+1}. {feature_names[indices[i]]}: {importance[indices[i]]:.4f}")
    
    def confusion_matrix_analysis(self):
        """혼동 행렬 분석"""
        print("\n=== 혼동 행렬 분석 ===")
        
        # 최고 성능 모델 선택
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['f1'])
        print(f"최고 F1-Score 모델: {best_model}")
        
        cm = confusion_matrix(self.y_test, self.results[best_model]['y_pred'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Purchase', 'Purchase'],
                   yticklabels=['No Purchase', 'Purchase'])
        plt.title(f'Confusion Matrix - {best_model}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # 분류 리포트
        print(f"\n{classification_report(self.y_test, self.results[best_model]['y_pred'])}")
    
    def run_experiment(self):
        """전체 실험 실행"""
        print("=== 구매 전환 예측 실험 시작 ===")
        
        # 1. 데이터 로드
        self.load_data()
        
        # 2. 데이터 탐색
        self.explore_data()
        
        # 3. 데이터 전처리
        self.preprocess_data()
        
        # 4. 불균형 데이터 처리
        self.handle_imbalanced_data(method='smote')
        
        # 5. 모델 학습
        self.train_models()
        
        # 6. 하이퍼파라미터 튜닝
        self.hyperparameter_tuning('RandomForest')
        self.hyperparameter_tuning('XGBoost')
        
        # 7. 결과 시각화
        self.plot_results()
        
        # 8. 특성 중요도 분석
        self.feature_importance_analysis()
        
        # 9. 혼동 행렬 분석
        self.confusion_matrix_analysis()
        
        print("\n=== 실험 완료 ===")
        
        return self.results

def main():
    """메인 함수"""
    # 데이터 경로 설정
    BASE_PATH = os.getcwd()
    CATEGORY_PATH = BASE_PATH + r'\practice\linear_regression'
    PROJECT_PATH = CATEGORY_PATH + r'\online_shoppers_intension'
    DATA_PATH = PROJECT_PATH + r'\online_shoppers_intention.csv'
    data_path = DATA_PATH
    
    # 실험 객체 생성 및 실행
    experiment = PurchaseConversionExperiment(data_path)
    results = experiment.run_experiment()
    
    # 최종 결과 요약
    print("\n=== 최종 결과 요약 ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    main() 