## dropout 이란?
### 학습 시 전체 신경망 중 일부만을 사용하도록하는 것. (단, 제거하지 않고)
### 즉, 학습 단계마다 일부 뉴런을 사용하지 않음으로써, 일부 특징이 특정 뉴런들에 고정되는 것을 막아 가중치의 균형을 잡도록 해 과적합을 방지
### 신경망이 충분히 학습되기까지 시간은 좀 더 오래 걸림.

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot = True)

## 신경망 모델 구성
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32) # 학습 시에 얼마 만큼의 뉴런을 사용할 것인지를 정해준다.

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X,W1))
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1,W2))
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256,10], stddev = 0.01))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = model, labels = Y)) # 손실값
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

## 신경망 모델 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(30):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost], feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.8})

        total_cost += cost_val

    print('Epoch : ', '%04d' % (epoch + 1),
          'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

## 결과 확인
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : ', sess.run(accuracy, feed_dict = {X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))