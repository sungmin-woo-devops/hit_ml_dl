[문제]
광고 예산(TV, Radio, Newspaper)과 판매량(Sales) 데이터를 사용하여 선형 회귀 모델을 학습시키고, 새로운 광고 예산이 주어졌을 때 판매량을 예측하는 프로그램을 작성하세요. 또한, 예측 결과와 실제 데이터를 시각화하여 모델의 성능을 직관적으로 확인하세요.

[세부 요구사항]
1. 데이터 로드
 - 캐글(https://www.kaggle.com/)에서 구글 계정으로 가입합니다.
 - 캐글 메인에서 좌측 Datasets 메뉴 선택하고, 다음 화면 상단 검색창에서 Advertising을 검색합니다.
 - 다음 화면에서 상단 Datasets를 클릭하고 조회된 결과에서 Advertising Dataset 선택한 후 Download 버튼 클릭하여 다운받습니다.
 - 다운받은 파일은 로컬 PC dataset로 이동하여 저장합니다.
 - dataset/Advertising.csv 경로에서 데이터를 로드합니다. (pandas 라이브러리 사용)

2. 데이터 전처리
 - 불필요한 'Unnamed: 0' 컬럼을 제거합니다.
 - 독립 변수(X)는 'TV', 'Radio', 'Newspaper' 컬럼으로 설정합니다.
 - 종속 변수(y)는 'Sales' 컬럼으로 설정합니다.

3. 데이터 분할
 - 전체 데이터를 학습 데이터와 테스트 데이터로 8:2 비율로 분할합니다. (scikit-learn의 train_test_split 함수 사용)

4. 모델 학습
선형 회귀 모델을 학습 데이터로 학습시킵니다. (scikit-learn의 LinearRegression 모델 사용)

5. 예측
 - 테스트 데이터에 대한 판매량을 예측합니다.

6. 모델 평가
 - 테스트 데이터에 대한 R-squared (결정 계수) 값을 계산하여 모델 성능을 평가합니다. (scikit-learn의 r2_score 함수 사용)

7. 새로운 데이터 예측
 - TV=200, Radio=50, Newspaper=30 일 때 판매량을 예측합니다.

8. 결과 출력
 - 테스트 데이터에 대한 R-squared 값을 출력합니다.
 - 새로운 데이터에 대한 예측 판매량을 출력합니다.

9. 시각화
 - 실제 판매량 vs 예측 판매량 산점도: 테스트 데이터의 실제 판매량과 예측 판매량을 산점도로 시각화합니다. x축은 실제 판매량, y축은 예측 판매량으로 설정하고, 제목과 축 레이블을 명확하게 표시합니다.
 - 각 독립변수와 판매량과의 관계 시각화: 각 독립 변수('TV', 'Radio', 'Newspaper')와 'Sales' 간의 산점도를 그리고, 회귀선을 추가하여 시각화합니다.


https://classroom.google.com/u/1/c/NzcwNzMzMzg2NjIy/a/Nzk0NDk3ODE0MTkw/details