"""
비트코인 데이터 분석 모듈 패키지

이 패키지는 비트코인 데이터 수집, 처리, 분석을 위한 모듈들을 포함합니다.
"""

from .data_collector import DataCollector
from .feature_engineering import FeatureEngineer
from .data_processor import DataProcessor
from .model_trainer import ModelTrainer
from .utils import setup_environment, get_data_paths

__version__ = "1.0.0"
__author__ = "Bitcoin Analysis Team"

__all__ = [
    'DataCollector',
    'FeatureEngineer', 
    'DataProcessor',
    'ModelTrainer',
    'setup_environment',
    'get_data_paths'
] 