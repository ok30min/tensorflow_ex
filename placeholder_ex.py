import tensorflow as tf

## placeholder = 입력값을 나중에 받기 위해 사용하는 매개변수
X = tf.placeholder(tf.float32, [None, 3])
print(X)

x_data = [[1,2,3], [4,5,6]]

## random_normal = 정규분포의 무작위 값으로 초기화
W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))

## 수식 작성  (행렬의 곱이기 때문에 matmul함수 사용)
expr = tf.matmul(X,W) + b

sess = tf.Session()
sess.run(tf.initialize_all_variables())  ## initailize_all_variables : 앞에서 정의한 변수들을 초기화

print("===x_data===")
print(x_data)
print("===W===")
print(sess.run(W))
print("===b===")
print(sess.run(b))

print("===expr===")
print(sess.run(expr, feed_dict = {X : x_data}))

sess.close
