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
