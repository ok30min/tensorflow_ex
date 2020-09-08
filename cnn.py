import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot = True)

### 신경망 모델 구성
## CNN 모델은 2차원 평면으로 구성
X = tf.placeholder(tf.float32, [None, 28,28,1]) # None = 입력 데이터의 갯수, 마지막 차원인 1은 특징의 개수(색상이 1개이므로 1)
Y = tf.placeholder(tf.float32, [None,10])       # 출력값 : 10
keep_prob = tf.placeholder(tf.float32)          # dropout을 위한 keep_prob 정의

## 3x3 크기의 커널을 가진 컨볼루션 계층 만들기 (32개)
# 오른쪽과 아래쪽으로 한 칸씩 움직이는 32개의 커널을 가진 컨볼루션 계층을 만드는 코드
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01)) # 가중치
L1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME') # padding = 'SAME' : 커널 슬라이딩 시 이미지의 가장 외곽에서 한 칸 밖으로 움직이는 옵션 (테두리까지 정확히 평가)
L1 = tf.nn.relu(L1)
# 2x2 크기의 커널을 가진 풀링 계층 만들기
L1 = tf.nn.max_pool(L1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME') # strides=[1,2,2,1] : 슬라이딩 시 두 칸씩 움직이는 옵션

## 3x3 크기의 커널을 가진 컨볼루션 계층 만들기 (64개)
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01)) # 32 = 첫 번째 컨볼루션 계층의 커널 갯수 = 출력층의 개수 = 첫 번째 컨볼루션 계층이 찾아낸 이미지의 특징 개수
L2 = tf.nn.conv2d(L1, W2, strides = [1,1,1,1], padding = 'SAME')
L2 = tf.nn.relu(L2)
# 2x2 크기의 커널을 가진 풀링 계층 만들기
L2 = tf.nn.max_pool(L2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

## 추출한 특징들을 이용해 10개의 분류를 만들어 내는 계층 구성
# 10개의 분류는 1차원 배열이므로 차원을 줄이는 단계를 거쳐야 함
W3 = tf.Variable(tf.random_normal([7*7*64,256], stddev=0.01)) # 7*7*64 = 직전의 풀링 계층의 크기
L3 = tf.reshape(L2, [-1,7*7*64]) # reshape를 이용해 7*7*64 크기의 1차원 계층으로 만들기
L3 = tf.matmul(L3,W3) # 256개의 뉴런으로 연결하는 신경망 생성
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob) # dropout을 사용하는 이유 : 과적합을 막아주기 위해 (추후에 다른 계층에도 dropout을 적용해 결과 확인)

## 은닉층인 L3의 출력값 256개를 받아 최종 출력값인 0~9 레이블을 갖는 10개의 출력값을 만든다.
W4 = tf.Variable(tf.random_normal([256,10], stddev = 0.01))
model = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = model, labels = Y)) # 손실함수
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost) # 최적화 함수 (RMSPropOptimizer로 바꿔서 결과 비교해보기)

###신경망 모델 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1,28,28,1)

        _, cost_val = sess.run([optimizer, cost], feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7})
        total_cost += cost_val

    print('Epoch:', '%.04d' % (epoch + 1), 'Avg. cost = ', '{:,3f}'.format(total_cost / total_batch))

print('최적화 완료!')

###결과 확인
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : ', sess.run(accuracy, feed_dict = {X: mnist.test.images.reshape(-1,28,28,1), Y: mnist.test.labels, keep_prob: 1}))