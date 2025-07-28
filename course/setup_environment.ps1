# 가상환경 설정 스크립트
# 호환성 문제 해결을 위한 환경 구성

Write-Host "=== ML/DL 프로젝트 환경 설정 ===" -ForegroundColor Green

# 1. 기존 가상환경 정리
Write-Host "1. 기존 가상환경 정리 중..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "기존 venv 폴더를 삭제합니다..." -ForegroundColor Yellow
    try {
        Remove-Item -Recurse -Force "venv" -ErrorAction Stop
        Write-Host "기존 가상환경 삭제 완료" -ForegroundColor Green
    }
    catch {
        Write-Host "가상환경 삭제 중 오류 발생. 수동으로 삭제해주세요." -ForegroundColor Red
        Write-Host "관리자 권한으로 PowerShell을 실행하거나 Jupyter를 종료한 후 다시 시도하세요." -ForegroundColor Red
        exit 1
    }
}

# 2. 새 가상환경 생성
Write-Host "2. 새 가상환경 생성 중..." -ForegroundColor Yellow
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "가상환경 생성 실패" -ForegroundColor Red
    exit 1
}
Write-Host "가상환경 생성 완료" -ForegroundColor Green

# 3. 가상환경 활성화
Write-Host "3. 가상환경 활성화 중..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "가상환경 활성화 실패" -ForegroundColor Red
    exit 1
}
Write-Host "가상환경 활성화 완료" -ForegroundColor Green

# 4. pip 업그레이드
Write-Host "4. pip 업그레이드 중..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "pip 업그레이드 완료" -ForegroundColor Green

# 5. 기본 패키지 설치 (호환성 순서대로)
Write-Host "5. 기본 패키지 설치 중..." -ForegroundColor Yellow

# NumPy 먼저 설치 (다른 패키지들의 기반)
Write-Host "  - NumPy 설치 중..." -ForegroundColor Cyan
pip install "numpy==1.24.3"
if ($LASTEXITCODE -ne 0) {
    Write-Host "NumPy 설치 실패" -ForegroundColor Red
    exit 1
}

# 핵심 데이터 사이언스 패키지들
Write-Host "  - 핵심 데이터 사이언스 패키지 설치 중..." -ForegroundColor Cyan
pip install "pandas==2.0.3" "matplotlib==3.7.2" "seaborn==0.12.2" "scipy==1.11.1"

# 머신러닝 패키지들
Write-Host "  - 머신러닝 패키지 설치 중..." -ForegroundColor Cyan
pip install "scikit-learn==1.3.0" "tensorflow==2.13.0" "keras==2.13.1"

# 데이터 분석 패키지들
Write-Host "  - 데이터 분석 패키지 설치 중..." -ForegroundColor Cyan
pip install "ydata-profiling==4.6.3" "sweetviz==2.2.1" "autoviz==0.1.905"

# 추가 패키지들
Write-Host "  - 추가 패키지 설치 중..." -ForegroundColor Cyan
pip install "dataprep==0.4.4" "dtale==3.9.0" "lux==0.5.1" "talib-binary==0.4.26" "requests==2.31.0"

# Jupyter 환경
Write-Host "  - Jupyter 환경 설치 중..." -ForegroundColor Cyan
pip install "jupyter==1.0.0" "ipykernel==6.25.2" "notebook==7.0.2"

# 기타 의존성 패키지들
Write-Host "  - 기타 의존성 패키지 설치 중..." -ForegroundColor Cyan
pip install "Pillow==10.0.0" "wordcloud==1.9.2" "statsmodels==0.14.0" "phik==0.12.5" "visions==0.7.6" "pydantic==2.4.2" "typeguard==4.0.0" "imagehash==4.3.1" "PyYAML==6.0.1" "tqdm==4.66.1"

Write-Host "모든 패키지 설치 완료!" -ForegroundColor Green

# 6. 설치 확인
Write-Host "6. 설치 확인 중..." -ForegroundColor Yellow
python -c "import numpy; print(f'NumPy 버전: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas 버전: {pandas.__version__}')"
python -c "import matplotlib; print(f'Matplotlib 버전: {matplotlib.__version__}')"
python -c "import ydata_profiling; print('ydata-profiling 설치 확인 완료')"

Write-Host "=== 환경 설정 완료! ===" -ForegroundColor Green
Write-Host "이제 Jupyter Notebook을 실행할 수 있습니다:" -ForegroundColor Cyan
Write-Host "jupyter notebook" -ForegroundColor White 