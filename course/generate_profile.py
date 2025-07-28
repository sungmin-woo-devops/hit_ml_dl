import os
import pandas as pd
from ydata_profiling import ProfileReport

# ?곗씠??遺덈윭?ㅺ린
file_path = '../dataset/diabetes.csv'
df = pd.read_csv(file_path)

# ?곗씠???뺣낫 異쒕젰
print(df.info())

# YData Profiling?쇰줈 ?곗씠???꾨줈?뚯씪留?蹂닿퀬???앹꽦
profile = ProfileReport(df, title='Diabetes Dataset Profiling Report', explorative=True)

# 蹂닿퀬?쒕? HTML ?뚯씪濡????
output_file = '../report/diabetes_profiling_report.html'
profile.to_file(output_file)
print(f'?꾨줈?뚯씪留?蹂닿퀬?쒓? ?앹꽦?섏뿀?듬땲?? {output_file}')
