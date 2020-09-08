import tensorflow as tf
#import tensorflow.compat.v1 as tfc
#tfc.disable_v2_behavior()

x_data = [1,2,3]
y_data = [1,2,3]

## -1부터 1사이의 균등분포를 가진 무작위 값으로 초기화
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # 가중치
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # 편향

## 자료를 입력받을 placeholder 설정
X = tf.placeholder(tf.float32, name = "X")
Y = tf.placeholder(tf.float32, name = "Y")

## 상관관계(선형관계) 분석을 위한 수식
hypothesis = W * X + b

## 비용 = 모든 데이터의 대한 손실값의 평균 (손실함수)
cost = tf.reduce_mean(tf.square(hypothesis - Y))

### 손실함수 :  한 쌍의 데이터에 대한 손실값을 계산하는 함수
### 손실값 : 실제값과 모델로 예측한 값이 얼마나 차이가 나는가를 나타내는 값, 손실값이 작을수록 X값에 대한 Y값ㅇ르 정확하게 예측할 수 있다는 뜻
       ### 그리고 이 손실을 전체 데이터에 대해 구한 경우 이를 비용이라고 함.

### 학습 : 변수들의 값을 다양하게 넣어 계산해보면서 이 손실값을 최소화하는 W와 b의 값을 구하는 것

### https://www.tensorflow.org/api_docs/ 에서 어떤 함수가 있는지 살펴볼 수 있다.

## 경사하강법 최적화 함수 사용 (손실값을 최소화하는 연산 그래프 생성)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train_op = optimizer.minimize(cost)

### 경사하강법 : 최적화 방법 중 가장 기본적인 알고리즘으로, 함수의 기울기를 구하고 기울기가 낮은 쪽으로 계속 이동시키면서 최적의 값을 찾는 방법

## with를 이용해 세션 블록을 만들고 세션 종료를 자동으로 처리하도록 만듬
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    ## train_op를 실행하고, 실행 시마다 변화하는 손실값을 출력하는 코드  (range(x) : (x-1)번 학습을 수행 후 상관관계 데이터를 입력)
    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict = {X: x_data, Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))

    ## 학습에 사용되지 않은 5와 1.5를 X값으로 넣고 결과 확인
    print("\n=== Test ===")
    print("X : 5, Y : ", sess.run(hypothesis, feed_dict = {X : 5}))
    print("X : 2.5, Y : ", sess.run(hypothesis, feed_dict={X : 2.5}))
