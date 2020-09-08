## 신경망의 층을 둘 이상으로 구성한 심층 신경망(딥러닝) 구현

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
W1 = tf.Variable(tf.random_uniform([2,10], -1., 1.))  #[특징, 은닉층의 뉴런 수]
W2 = tf.Variable(tf.random_uniform([10,3], -1., 1.))  #[은닉층의 뉴런 수, 분류 수]

b1 = tf.Variable(tf.zeros([10])) #은닉층의 뉴런 수
b2 = tf.Variable(tf.zeros([3]))  #분류 수

## 특징 입력값에 첫 번쨰 가중치와 편향, 그리고 활성화 함수(ReLU) 적용
L1 = tf.add(tf.matmul(X,W1), b1)
L1 = tf.nn.relu(L1)

## 출력층을 만들기 위해 두 번째 가중치와 편향을 적용해 최종 모델 작성 => 출력값 : 3개
model = tf.add(tf.matmul(L1, W2), b2)

## 손실함수 작성 (교차 엔트로피 함수 : 예측값과 실제값 사이의 확률 분포 차이를 계산한 값)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = model))

## AdamOptimizer 사용 : GradientDescentOptimizer보다 보편적으로 성능이 좋다고 알려져 있음.
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
train_op = optimizer.minimize(cost)

## 텐서플로 세션 초기화
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

## 특징과 레이블 데이터를 이용해 학습을 진행
for step in range(100):
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