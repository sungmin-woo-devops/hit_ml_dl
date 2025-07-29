#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
구매 전환 예측 실험 실행 스크립트
"""

import sys
import os
from purchase_conversion_experiment import PurchaseConversionExperiment

def main():
    """실험 실행"""
    print("구매 전환 예측 실험을 시작합니다...")
    
    # 데이터 파일 경로 확인
    BASE_PATH = os.getcwd()
    CATEGORY_PATH = BASE_PATH + r'\practice\linear_regression'
    PROJECT_PATH = CATEGORY_PATH + r'\online_shoppers_intension'
    data_file = PROJECT_PATH + r'\online_shoppers_intention.csv'
    
    if not os.path.exists(data_file):
        print(f"오류: 데이터 파일 '{data_file}'을 찾을 수 없습니다.")
        print("데이터 파일이 현재 디렉토리에 있는지 확인해주세요.")
        return
    
    try:
        # 실험 객체 생성
        experiment = PurchaseConversionExperiment(data_file)
        
        # 실험 실행
        results = experiment.run_experiment()
        
        # 최종 결과 출력
        print("\n" + "="*50)
        print("실험 완료!")
        print("="*50)
        
        # 최고 성능 모델 찾기
        best_model = max(results.keys(), key=lambda x: results[x]['f1'])
        best_f1 = results[best_model]['f1']
        
        print(f"\n최고 성능 모델: {best_model}")
        print(f"최고 F1-Score: {best_f1:.4f}")
        
        print(f"\n{best_model} 상세 성능:")
        for metric, value in results[best_model].items():
            if metric not in ['y_pred', 'y_pred_proba']:
                print(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        print(f"실험 실행 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 