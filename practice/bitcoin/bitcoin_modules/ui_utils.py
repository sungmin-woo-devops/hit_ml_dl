"""
UI 유틸리티 모듈
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Optional


def create_sidebar_settings(df: pd.DataFrame) -> Dict:
    """사이드바 설정 생성"""
    st.sidebar.header("설정")
    
    # 지표 선택
    st.sidebar.subheader("분석 지표 선택")
    
    # 사용 가능한 가격 지표들 찾기
    price_indicators = []
    for col in df.columns:
        if any(x in col.lower() for x in ['btc', 'price', 'usd_', 'eur', 'jpy', 'krw', 'cny', 'xau']):
            price_indicators.append(col)
    
    # 기본 선택값 설정
    default_indicator = 'USD_EUR' if 'USD_EUR' in price_indicators else price_indicators[0] if price_indicators else None
    
    selected_indicator = st.sidebar.selectbox(
        "분석할 지표 선택",
        price_indicators,
        index=price_indicators.index(default_indicator) if default_indicator else 0,
        key="indicator_selector"
    )
    
    # 기술지표 표시 여부
    show_technical_indicators = st.sidebar.checkbox(
        "기술지표 표시",
        value=True,
        key="show_tech_indicators"
    )
    
    # 차트 스타일 선택
    chart_style = st.sidebar.selectbox(
        "차트 스타일",
        ["라인 차트", "캔들스틱 차트", "영역 차트"],
        key="chart_style"
    )
    
    # 머신러닝 모델 설정
    st.sidebar.subheader("머신러닝 모델 설정")
    
    # 모델 활성화 여부
    enable_ml = st.sidebar.checkbox(
        "머신러닝 모델 활성화",
        value=False,
        key="enable_ml"
    )
    
    ml_settings = {}
    if enable_ml:
        # 랜덤 포레스트 하이퍼파라미터
        st.sidebar.write("**랜덤 포레스트 하이퍼파라미터**")
        
        ml_settings['n_estimators'] = st.sidebar.slider(
            "트리 개수 (n_estimators)",
            min_value=10,
            max_value=200,
            value=100,
            step=10,
            key="n_estimators"
        )
        
        ml_settings['max_depth'] = st.sidebar.slider(
            "최대 깊이 (max_depth)",
            min_value=3,
            max_value=20,
            value=10,
            step=1,
            key="max_depth"
        )
        
        ml_settings['min_samples_split'] = st.sidebar.slider(
            "분할 최소 샘플 수 (min_samples_split)",
            min_value=2,
            max_value=20,
            value=2,
            step=1,
            key="min_samples_split"
        )
        
        ml_settings['min_samples_leaf'] = st.sidebar.slider(
            "리프 최소 샘플 수 (min_samples_leaf)",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            key="min_samples_leaf"
        )
        
        ml_settings['test_size'] = st.sidebar.slider(
            "테스트 데이터 비율",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            key="test_size"
        )
        
        # 모델 학습 버튼
        ml_settings['train_model'] = st.sidebar.button(
            "모델 학습 시작",
            key="train_model"
        )
        
        # Optuna 하이퍼파라미터 최적화
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Optuna 자동 최적화")
        
        # 최적화 설정
        ml_settings['n_trials'] = st.sidebar.slider(
            "최적화 시도 횟수",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            key="n_trials"
        )
        
        ml_settings['optimize_hyperparams'] = st.sidebar.checkbox(
            "Optuna 최적화 실행",
            value=False,
            key="optimize_hyperparams"
        )
    
    return {
        'selected_indicator': selected_indicator,
        'show_technical_indicators': show_technical_indicators,
        'chart_style': chart_style,
        'enable_ml': enable_ml,
        'ml_settings': ml_settings
    }


def create_metrics_display(df: pd.DataFrame, price_col: str) -> None:
    """메트릭 표시"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_price = df[price_col].iloc[-1]
        price_change = df[price_col].pct_change().iloc[-1]
        st.metric(
            label=f"현재 {price_col}",
            value=f"{current_price:,.4f}",
            delta=f"{price_change*100:.2f}%"
        )
    
    with col2:
        max_price = df[price_col].max()
        max_change = (max_price - current_price) / current_price * 100
        st.metric(
            label=f"최고 {price_col}",
            value=f"{max_price:,.4f}",
            delta=f"{max_change:.2f}%"
        )
    
    with col3:
        min_price = df[price_col].min()
        min_change = (min_price - current_price) / current_price * 100
        st.metric(
            label=f"최저 {price_col}",
            value=f"{min_price:,.4f}",
            delta=f"{min_change:.2f}%"
        )


def display_model_results(results: Dict) -> None:
    """모델 결과 표시"""
    st.subheader("🤖 모델 예측 결과")
    
    y_test = results['y_test']
    y_pred = results['y_pred']
    r2 = results['r2']
    rmse = results['rmse']
    
    # 예측 성능 요약
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² 점수", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"{rmse:.6f}")
    with col3:
        st.metric("예측 정확도", f"{max(0, r2)*100:.1f}%")
    with col4:
        mean_error = abs(y_test - y_pred).mean()
        st.metric("평균 오차", f"{mean_error:.6f}")


def create_period_selector() -> str:
    """기간 선택기 생성"""
    return st.selectbox(
        "분석 기간 선택",
        ["전체", "1년", "6개월", "3개월", "1개월"],
        key="period_selector"
    )


def get_period_days(period: str) -> Optional[int]:
    """기간에 따른 일수 반환"""
    period_days = {
        "1년": 365,
        "6개월": 180,
        "3개월": 90,
        "1개월": 30
    }
    return period_days.get(period)


def display_technical_indicators_info() -> None:
    """기술지표 설명 표시"""
    with st.expander("기술지표 설명"):
        st.markdown("""
        - **RSI (Relative Strength Index)**: 과매수/과매도 지표
          - 70 이상: 과매수 구간
          - 30 이하: 과매도 구간
        - **볼린저 밴드 %B**: 가격의 상대적 위치
          - 1 이상: 상단 밴드 위
          - 0 이하: 하단 밴드 아래
        - **이동평균 비율**: MA20/MA50 비율
          - 1 이상: 상승 추세
          - 1 미만: 하락 추세
        """)


def display_ml_model_info() -> None:
    """머신러닝 모델 설명 표시"""
    with st.expander("머신러닝 모델 설명"):
        st.markdown("""
        **랜덤 포레스트 모델**
        
        - **트리 개수 (n_estimators)**: 앙상블에 사용할 의사결정 트리의 개수
        - **최대 깊이 (max_depth)**: 각 트리의 최대 깊이
        - **분할 최소 샘플 수 (min_samples_split)**: 노드를 분할하기 위한 최소 샘플 수
        - **리프 최소 샘플 수 (min_samples_leaf)**: 리프 노드에 필요한 최소 샘플 수
        
        **성능 지표**
        - **MSE**: 평균 제곱 오차
        - **MAE**: 평균 절대 오차
        - **R²**: 결정 계수 (1에 가까울수록 좋음)
        - **RMSE**: 평균 제곱근 오차
        """)


def create_data_download_section(df: pd.DataFrame) -> None:
    """데이터 다운로드 섹션 생성"""
    st.subheader("데이터 다운로드")
    csv = df.to_csv(index=False)
    st.download_button(
        label="CSV 다운로드",
        data=csv,
        file_name="bitcoin_analysis_data.csv",
        mime="text/csv"
    )


def display_system_info(df: pd.DataFrame, data_dir: str, price_col: str, enable_ml: bool) -> None:
    """시스템 정보 표시"""
    st.subheader("시스템 정보")
    st.write(f"데이터 폴더: {data_dir}")
    st.write(f"데이터 크기: {df.shape}")
    st.write(f"분석 지표: {price_col}")
    st.write(f"머신러닝 활성화: {'예' if enable_ml else '아니오'}")
    from datetime import datetime
    st.write(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def display_ml_settings(ml_settings: Dict) -> None:
    """머신러닝 설정 표시"""
    st.subheader("현재 머신러닝 설정")
    st.write(f"트리 개수: {ml_settings.get('n_estimators', 'N/A')}")
    st.write(f"최대 깊이: {ml_settings.get('max_depth', 'N/A')}")
    st.write(f"분할 최소 샘플 수: {ml_settings.get('min_samples_split', 'N/A')}")
    st.write(f"리프 최소 샘플 수: {ml_settings.get('min_samples_leaf', 'N/A')}")
    st.write(f"테스트 데이터 비율: {ml_settings.get('test_size', 'N/A'):.1%}")


def create_progress_display() -> tuple:
    """진행 상황 표시 생성"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    return progress_bar, status_text


def update_progress(progress_bar, status_text, progress: int, message: str) -> None:
    """진행 상황 업데이트"""
    progress_bar.progress(progress)
    status_text.text(message)


def clear_progress(progress_bar, status_text) -> None:
    """진행 상황 표시 제거"""
    progress_bar.empty()
    status_text.empty() 