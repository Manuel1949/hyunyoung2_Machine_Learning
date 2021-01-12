
import tensorflow as tf 


input_text = tf.constant([[[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]]], dtype=tf.float32)

#conv_w = tf.get_variable("CONV_W", shape=[2, 3, 3], dtype=tf.float32)

conv_w = tf.constant([[[1],[2],[3]],[[3],[2],[1]]], dtype=tf.float32)

#conv_w = tf.constant([[1,1,1],[2,2,2],[3,3,3]], dtype=tf.float32)

conv_bias = tf.constant(value=1, shape=[1], dtype=tf.float32)

result = tf.nn.conv1d(input_text, conv_w, 1, "VALID")

result_with_bias = tf.add(result, conv_bias)
#result = tf.nn.conv2d(input_text, conv_w, strides=[1,1,1,1], padding="VALID", name="Conv")

#init_op = tf.initialize_all_variables()

sess = tf.Session()

#sess.run(init_op)

print("INPUT TEXT")
print(sess.run(input_text))

print("Convolution's weight")
print(sess.run(conv_w))

print("Convolution's bias")
print(sess.run(conv_bias))

print("Result")
print(sess.run(result))

print("Result with bias")
print(sess.run(result_with_bias))

