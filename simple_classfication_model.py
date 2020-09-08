import tensorflow as tf
#import tensorflow.compat.v1 as tfc
#tfc.disable_v2_behavior()

import numpy as np

## 학습에 사용할 데이터 정의  ([털, 날개] 유무)
x_data = np.array([[0,0], [1,0], [1,1], [0,0], [0,0], [0,1]])

### 레이블 데이터는 one_hot encoding이라는 특수한 형태로 구성
   ### one_hot_encoding : 데이터가 가질 수 있는 값들을 일렬로 나열한 배열을 만들고, 그중 표현하려는 값을 뜻한는 인덱스의 원소만 1로
                    ### 표기하고, 나머지 원소는 모두 0으로 채우는 표기법

y_data = np.array([
    [1,0,0], # 기타
    [0,1,0], # 포유류
    [0,0,1], # 조류
    [1,0,0],
    [1,0,0],
    [0,0,1]
])

## 신경망 모델 구성
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

## 신경망을 결정하는 가중치과 편향 변수 설정
W = tf.Variable(tf.random_uniform([2,3], -1., 1.))
b = tf.Variable(tf.zeros([3]))  #레이블 수인 3개의 요소를 가진 변수로 설정

## 가중치를 곱하고 편향을 더한 결과를 활성화 함수인 ReLU에 적용해 신경망 구성
L = tf.add(tf.matmul(X,W), b)
L = tf.nn.relu(L)

## 신경망을 통해 나온 출력값을 softmax 함수를 이용해 사용하기 쉽게 다듬기
model = tf.nn.softmax(L)   # softmax 함수는 배열 내의 결괏값들을 전체 합이 1이 되도록 만들어 준다.

## 손실함수 작성 (교차 엔트로피 함수 : 예측값과 실제값 사이의 확률 분포 차이를 계산한 값)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis = 1))

## 경사하강법으로 최적화 진행
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train_op = optimizer.minimize(cost)

## 텐서플로 세션 초기화
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

## 특징과 레이블 데이터를 이용해 학습을 진행
for step in range(1000):
    sess.run(train_op, feed_dict = {X: x_data, Y: y_data})

    ## 학습 도중 10번에 한 번씩 손실값을 출력
    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict = {X: x_data, Y: y_data}))

## 학습된 결과를 확인
prediction = tf.argmax(model, axis = 1)
target = tf.argmax(Y, axis = 1)  #argmax : 가장 큰 값의 인덱스를 찾아주는 argmax함수를 이용해 레이블 값 출력
print('예측값 : ', sess.run(prediction, feed_dict = {X: x_data}))
print('실제값 : ', sess.run(target, feed_dict = {Y: y_data}))

## 정확도 출력
is_correct = tf.equal(prediction, target)  #equal : 예측값과 실측값을 비교, 값은 true/false
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))  #tf.cast : 0,과 1로 바꿔 평균을 냄
print('정확도 : %.2f' % sess.run(accuracy * 100, feed_dict = {X: x_data, Y: y_data}))