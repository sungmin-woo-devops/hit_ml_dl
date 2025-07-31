import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="비트코인 데이터 분석 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 경로 설정
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "practice", "data")

@st.cache_data
def load_data():
    """데이터 로드 함수"""
    try:
        # 여러 데이터 파일 시도
        data_files = [
            os.path.join(DATA_DIR, "merged_data_features.csv"),
            os.path.join(DATA_DIR, "merged_data.csv"),
            os.path.join(DATA_DIR, "crypto_forex_data.csv")
        ]
        
        for file_path in data_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                st.success(f"데이터 로드 성공: {file_path}")
                return df
        
        st.error("사용 가능한 데이터 파일을 찾을 수 없습니다.")
        return None
        
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None

def create_technical_indicators(df):
    """기술지표 생성"""
    if df is None:
        return df
    
    # 가격 컬럼 찾기
    price_cols = [col for col in df.columns if any(x in col.lower() for x in ['btc', 'price', 'usd_eur'])]
    price_col = price_cols[0] if price_cols else None
    
    if price_col is None:
        st.warning("가격 데이터를 찾을 수 없습니다.")
        return df
    
    # 로그 수익률
    if 'log_returns' not in df.columns:
        df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
    
    # RSI
    if 'rsi' not in df.columns:
        delta = df[price_col].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    # 볼린저 밴드
    if 'bb_percent' not in df.columns:
        ma = df[price_col].rolling(20).mean()
        std = df[price_col].rolling(20).std()
        upper = ma + (2 * std)
        lower = ma - (2 * std)
        df['bb_percent'] = (df[price_col] - lower) / (upper - lower)
    
    # 이동평균 비율
    if 'ma_ratio' not in df.columns:
        ma_20 = df[price_col].rolling(20).mean()
        ma_50 = df[price_col].rolling(50).mean()
        df['ma_ratio'] = ma_20 / ma_50
    
    return df

def plot_price_chart(df, price_col):
    """가격 차트 생성"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[price_col],
        mode='lines',
        name='가격',
        line=dict(color='blue', width=2)
    ))
    
    # 이동평균 추가
    ma_20 = df[price_col].rolling(20).mean()
    ma_50 = df[price_col].rolling(50).mean()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=ma_20,
        mode='lines',
        name='MA20',
        line=dict(color='orange', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=ma_50,
        mode='lines',
        name='MA50',
        line=dict(color='red', width=1)
    ))
    
    fig.update_layout(
        title=f'{price_col} 가격 차트',
        xaxis_title='날짜',
        yaxis_title='가격',
        hovermode='x unified'
    )
    
    return fig

def plot_technical_indicators(df):
    """기술지표 차트 생성"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('RSI', '볼린저 밴드 %B', '이동평균 비율'),
        vertical_spacing=0.1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['rsi'], name='RSI', line=dict(color='purple')),
        row=1, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # 볼린저 밴드 %B
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['bb_percent'], name='BB %B', line=dict(color='orange')),
        row=2, col=1
    )
    fig.add_hline(y=1, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="green", row=2, col=1)
    
    # 이동평균 비율
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['ma_ratio'], name='MA Ratio', line=dict(color='blue')),
        row=3, col=1
    )
    fig.add_hline(y=1, line_dash="dash", line_color="black", row=3, col=1)
    
    fig.update_layout(height=800, showlegend=False)
    return fig

def plot_correlation_heatmap(df):
    """상관관계 히트맵 생성"""
    # 수치형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['Unnamed: 0', 'index']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title="상관관계 히트맵"
    )
    
    return fig

def main():
    st.title("📈 비트코인 데이터 분석 대시보드")
    st.markdown("---")
    
    # 사이드바
    st.sidebar.header("설정")
    
    # 데이터 로드
    with st.spinner("데이터를 로드하고 있습니다..."):
        df = load_data()
    
    if df is None:
        st.error("데이터를 로드할 수 없습니다. 데이터 파일이 있는지 확인해주세요.")
        return
    
    # 기술지표 생성
    df = create_technical_indicators(df)
    
    # 날짜 컬럼 처리
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        df['date'] = pd.to_datetime(df.iloc[:, 0])
    
    # 가격 컬럼 찾기
    price_cols = [col for col in df.columns if any(x in col.lower() for x in ['btc', 'price', 'usd_eur'])]
    price_col = price_cols[0] if price_cols else None
    
    if price_col is None:
        st.error("가격 데이터를 찾을 수 없습니다.")
        return
    
    # 메인 대시보드
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="현재 가격",
            value=f"${df[price_col].iloc[-1]:,.2f}",
            delta=f"{df[price_col].pct_change().iloc[-1]*100:.2f}%"
        )
    
    with col2:
        st.metric(
            label="최고가",
            value=f"${df[price_col].max():,.2f}",
            delta=f"{(df[price_col].max() - df[price_col].iloc[-1])/df[price_col].iloc[-1]*100:.2f}%"
        )
    
    with col3:
        st.metric(
            label="최저가",
            value=f"${df[price_col].min():,.2f}",
            delta=f"{(df[price_col].min() - df[price_col].iloc[-1])/df[price_col].iloc[-1]*100:.2f}%"
        )
    
    st.markdown("---")
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 가격 차트", 
        "📈 기술지표", 
        "🔥 상관관계", 
        "📋 데이터 통계",
        "⚙️ 설정"
    ])
    
    with tab1:
        st.header("가격 차트")
        price_chart = plot_price_chart(df, price_col)
        st.plotly_chart(price_chart, use_container_width=True)
        
        # 기간 선택
        st.subheader("기간별 분석")
        periods = st.selectbox(
            "분석 기간 선택",
            ["전체", "1년", "6개월", "3개월", "1개월"]
        )
        
        if periods != "전체":
            period_days = {
                "1년": 365,
                "6개월": 180,
                "3개월": 90,
                "1개월": 30
            }
            days = period_days[periods]
            recent_df = df.tail(days)
            recent_chart = plot_price_chart(recent_df, price_col)
            st.plotly_chart(recent_chart, use_container_width=True)
    
    with tab2:
        st.header("기술지표")
        tech_chart = plot_technical_indicators(df)
        st.plotly_chart(tech_chart, use_container_width=True)
        
        # 기술지표 설명
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
    
    with tab3:
        st.header("상관관계 분석")
        corr_chart = plot_correlation_heatmap(df)
        st.plotly_chart(corr_chart, use_container_width=True)
        
        # 강한 상관관계 표시
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Unnamed: 0', 'index']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        corr_matrix = df[numeric_cols].corr()
        
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= 0.7:
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        if strong_correlations:
            st.subheader("강한 상관관계 (|상관계수| >= 0.7)")
            for i, pair in enumerate(strong_correlations[:10], 1):
                direction = "양의" if pair['correlation'] > 0 else "음의"
                st.write(f"{i}. {direction} {pair['var1']} ↔ {pair['var2']}: {pair['correlation']:.3f}")
    
    with tab4:
        st.header("데이터 통계")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("기본 통계")
            stats_df = df[price_col].describe()
            st.dataframe(stats_df)
        
        with col2:
            st.subheader("기술지표 통계")
            tech_stats = df[['rsi', 'bb_percent', 'ma_ratio']].describe()
            st.dataframe(tech_stats)
        
        # 데이터 정보
        st.subheader("데이터 정보")
        info_data = {
            "총 데이터 수": len(df),
            "분석 기간": f"{df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}",
            "컬럼 수": len(df.columns),
            "결측값": df.isnull().sum().sum()
        }
        
        for key, value in info_data.items():
            st.metric(key, value)
    
    with tab5:
        st.header("설정")
        
        # 데이터 새로고침
        if st.button("데이터 새로고침"):
            st.cache_data.clear()
            st.rerun()
        
        # 데이터 다운로드
        st.subheader("데이터 다운로드")
        csv = df.to_csv(index=False)
        st.download_button(
            label="CSV 다운로드",
            data=csv,
            file_name="bitcoin_analysis_data.csv",
            mime="text/csv"
        )
        
        # 설정 정보
        st.subheader("시스템 정보")
        st.write(f"데이터 폴더: {DATA_DIR}")
        st.write(f"데이터 크기: {df.shape}")
        st.write(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()