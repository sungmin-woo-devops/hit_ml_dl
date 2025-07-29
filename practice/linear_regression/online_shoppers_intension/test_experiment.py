#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
구매 전환 예측 실험 테스트 스크립트
각 단계별로 실험을 테스트할 수 있습니다.
"""

import os
import sys
from purchase_conversion_experiment import PurchaseConversionExperiment

def test_data_loading():
    """데이터 로딩 테스트"""
    print("=== 데이터 로딩 테스트 ===")
    
    BASE_PATH = os.getcwd()
    CATEGORY_PATH = BASE_PATH + r'\practice\linear_regression'
    PROJECT_PATH = CATEGORY_PATH + r'\online_shoppers_intension'
    DATA_PATH = PROJECT_PATH + r'\online_shoppers_intention.csv'
    data_file = DATA_PATH
    if not os.path.exists(data_file):
        print(f"오류: {data_file} 파일이 없습니다.")
        return False
    
    try:
        experiment = PurchaseConversionExperiment(data_file)
        df = experiment.load_data()
        print("✅ 데이터 로딩 성공")
        print(f"   데이터 크기: {df.shape}")
        return True
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return False

def test_data_exploration():
    """데이터 탐색 테스트"""
    print("\n=== 데이터 탐색 테스트 ===")
    
    try:
        experiment = PurchaseConversionExperiment("online_shoppers_intention.csv")
        experiment.load_data()
        experiment.explore_data()
        print("✅ 데이터 탐색 성공")
        return True
    except Exception as e:
        print(f"❌ 데이터 탐색 실패: {e}")
        return False

def test_preprocessing():
    """전처리 테스트"""
    print("\n=== 전처리 테스트 ===")
    
    try:
        experiment = PurchaseConversionExperiment("online_shoppers_intention.csv")
        experiment.load_data()
        X, y = experiment.preprocess_data()
        print("✅ 전처리 성공")
        print(f"   특성 수: {X.shape[1]}")
        print(f"   샘플 수: {X.shape[0]}")
        return True
    except Exception as e:
        print(f"❌ 전처리 실패: {e}")
        return False

def test_imbalanced_data_handling():
    """불균형 데이터 처리 테스트"""
    print("\n=== 불균형 데이터 처리 테스트 ===")
    
    try:
        experiment = PurchaseConversionExperiment("online_shoppers_intention.csv")
        experiment.load_data()
        experiment.preprocess_data()
        
        # SMOTE 테스트
        X_balanced, y_balanced = experiment.handle_imbalanced_data('smote')
        print("✅ SMOTE 처리 성공")
        
        # Undersample 테스트
        X_balanced2, y_balanced2 = experiment.handle_imbalanced_data('undersample')
        print("✅ Undersample 처리 성공")
        
        return True
    except Exception as e:
        print(f"❌ 불균형 데이터 처리 실패: {e}")
        return False

def test_model_training():
    """모델 학습 테스트"""
    print("\n=== 모델 학습 테스트 ===")
    
    try:
        experiment = PurchaseConversionExperiment("online_shoppers_intention.csv")
        experiment.load_data()
        experiment.preprocess_data()
        experiment.handle_imbalanced_data()
        experiment.train_models()
        
        print("✅ 모델 학습 성공")
        print(f"   학습된 모델 수: {len(experiment.results)}")
        
        # 결과 출력
        for model_name, metrics in experiment.results.items():
            print(f"   {model_name}: F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ 모델 학습 실패: {e}")
        return False

def test_hyperparameter_tuning():
    """하이퍼파라미터 튜닝 테스트"""
    print("\n=== 하이퍼파라미터 튜닝 테스트 ===")
    
    try:
        experiment = PurchaseConversionExperiment("online_shoppers_intention.csv")
        experiment.load_data()
        experiment.preprocess_data()
        experiment.handle_imbalanced_data()
        
        # RandomForest 튜닝 테스트
        experiment.hyperparameter_tuning('RandomForest')
        print("✅ RandomForest 튜닝 성공")
        
        return True
    except Exception as e:
        print(f"❌ 하이퍼파라미터 튜닝 실패: {e}")
        return False

def test_visualization():
    """시각화 테스트"""
    print("\n=== 시각화 테스트 ===")
    
    try:
        experiment = PurchaseConversionExperiment("online_shoppers_intention.csv")
        experiment.load_data()
        experiment.preprocess_data()
        experiment.handle_imbalanced_data()
        experiment.train_models()
        experiment.plot_results()
        
        print("✅ 시각화 성공")
        return True
    except Exception as e:
        print(f"❌ 시각화 실패: {e}")
        return False

def test_feature_importance():
    """특성 중요도 분석 테스트"""
    print("\n=== 특성 중요도 분석 테스트 ===")
    
    try:
        experiment = PurchaseConversionExperiment("online_shoppers_intention.csv")
        experiment.load_data()
        experiment.preprocess_data()
        experiment.handle_imbalanced_data()
        experiment.train_models()
        experiment.feature_importance_analysis()
        
        print("✅ 특성 중요도 분석 성공")
        return True
    except Exception as e:
        print(f"❌ 특성 중요도 분석 실패: {e}")
        return False

def test_confusion_matrix():
    """혼동 행렬 분석 테스트"""
    print("\n=== 혼동 행렬 분석 테스트 ===")
    
    try:
        experiment = PurchaseConversionExperiment("online_shoppers_intention.csv")
        experiment.load_data()
        experiment.preprocess_data()
        experiment.handle_imbalanced_data()
        experiment.train_models()
        experiment.confusion_matrix_analysis()
        
        print("✅ 혼동 행렬 분석 성공")
        return True
    except Exception as e:
        print(f"❌ 혼동 행렬 분석 실패: {e}")
        return False

def run_all_tests():
    """모든 테스트 실행"""
    print("구매 전환 예측 실험 테스트를 시작합니다...\n")
    
    tests = [
        ("데이터 로딩", test_data_loading),
        ("데이터 탐색", test_data_exploration),
        ("전처리", test_preprocessing),
        ("불균형 데이터 처리", test_imbalanced_data_handling),
        ("모델 학습", test_model_training),
        ("하이퍼파라미터 튜닝", test_hyperparameter_tuning),
        ("시각화", test_visualization),
        ("특성 중요도 분석", test_feature_importance),
        ("혼동 행렬 분석", test_confusion_matrix)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
            results[test_name] = False
    
    # 결과 요약
    print("\n" + "="*50)
    print("테스트 결과 요약")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ 통과" if success else "❌ 실패"
        print(f"{test_name}: {status}")
    
    print(f"\n전체 결과: {passed}/{total} 테스트 통과")
    
    if passed == total:
        print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("⚠️ 일부 테스트가 실패했습니다. 오류를 확인해주세요.")

def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        
        if test_name == "loading":
            test_data_loading()
        elif test_name == "exploration":
            test_data_exploration()
        elif test_name == "preprocessing":
            test_preprocessing()
        elif test_name == "imbalanced":
            test_imbalanced_data_handling()
        elif test_name == "training":
            test_model_training()
        elif test_name == "tuning":
            test_hyperparameter_tuning()
        elif test_name == "visualization":
            test_visualization()
        elif test_name == "importance":
            test_feature_importance()
        elif test_name == "confusion":
            test_confusion_matrix()
        else:
            print("사용법: python test_experiment.py [test_name]")
            print("사용 가능한 테스트: loading, exploration, preprocessing, imbalanced, training, tuning, visualization, importance, confusion")
    else:
        run_all_tests()

if __name__ == "__main__":
    main() 