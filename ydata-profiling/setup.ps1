# PowerShell 스크립트 - ydata-profiling 환경 설정

Write-Host "=== ydata-profiling 환경 설정 ===" -ForegroundColor Green

# 1. 필요한 패키지 설치 (NumPy 호환성 문제 해결)
Write-Host "1. NumPy 호환성 문제 해결 중..." -ForegroundColor Yellow
pip uninstall numpy -y
pip install "numpy<2.0" --force-reinstall

Write-Host "2. ydata-profiling 설치 중..." -ForegroundColor Yellow
pip install ydata-profiling==4.12.2 pandas==2.2.3

# 3. llvmlite와 numba 제거 (충돌 방지)
Write-Host "3. 충돌 방지를 위한 패키지 제거 중..." -ForegroundColor Yellow
pip uninstall llvmlite numba -y

# 4. report 폴더 생성
Write-Host "4. report 폴더 생성 중..." -ForegroundColor Yellow
if (-not (Test-Path -Path "report")) {
    New-Item -Path "report" -ItemType Directory
    Write-Host "report 폴더가 생성되었습니다." -ForegroundColor Green
}

# 5. Python 스크립트 실행을 위한 코드 작성
$pythonScript = @"
import os
import pandas as pd
from ydata_profiling import ProfileReport

# Current working directory
print(f"Current working directory: {os.getcwd()}")

# Data file path check (go up one directory to find dataset)
file_path = '../dataset/diabetes.csv'
if os.path.exists(file_path):
    print(f"Data file found: {file_path}")
else:
    print(f"Data file not found: {file_path}")
    print("Available files:")
    if os.path.exists('../dataset'):
        for file in os.listdir('../dataset'):
            print(f"  - {file}")
    exit(1)

# Load data
try:
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"Data loading failed: {e}")
    exit(1)

# Data information
print("\n=== Data Information ===")
print(df.info())

print("\n=== Data Preview ===")
print(df.head())

# Generate profiling report
print("\n=== Generating Profiling Report ===")
try:
    profile = ProfileReport(df, title='Diabetes Dataset Profiling Report', explorative=True)
    
    # Save report as HTML file
    output_file = 'report/diabetes_profiling_report.html'
    profile.to_file(output_file)
    print(f'Profiling report generated: {output_file}')
    
    # Check file size
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f'Report file size: {file_size:.2f} MB')
    
except Exception as e:
    print(f"Profiling report generation failed: {e}")
    exit(1)

print("\n=== Process Completed ===")
"@

# 6. Python 스크립트 파일로 저장 및 실행
$scriptPath = "generate_profile.py"
$pythonScript | Out-File -FilePath $scriptPath -Encoding UTF8
Write-Host "5. Python 스크립트 실행 중..." -ForegroundColor Yellow
python $scriptPath

# 7. 실행 완료 메시지
Write-Host "6. 프로세스 완료!" -ForegroundColor Green
Write-Host "report/diabetes_profiling_report.html 파일을 확인하세요." -ForegroundColor Cyan