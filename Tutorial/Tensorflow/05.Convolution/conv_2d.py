
"""
refer to http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

         https://cezannec.github.io/CNN_Text_Classification/
"""

import tensorflow as tf 


input_text = tf.constant([[[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]]], dtype=tf.float32)

input_text_expanded = tf.expand_dims(input_text, -1)

#conv_w = tf.get_variable("CONV_W", shape=[2, 3, 3], dtype=tf.float32)

#conv_w = tf.constant([[[1,1,1],[2,2,2],[3,3,3]],[[1,1,1],[2,2,2],[3,3,3]]], dtype=tf.float32)

# filter size, embedding_size, 1, num_filters
filter_shape = [2, 3, 1, 1]

#conv_w = tf.constant(value=1, shape=filter_shape, dtype=tf.float32)

conv_w = tf.constant([[[[1,1]],[[2,1]],[[3,1]]],[[[3,1]],[[2,1]],[[1,1]]]], dtype=tf.float32)

conv_bias = tf.constant([1,2], dtype=tf.float32)

result = tf.nn.conv2d(input_text_expanded, conv_w, strides=[1,1,1,1], padding="SAME", name="Conv")

result_with_bias = tf.add(result, conv_bias)

#ksize = [1, sequence_length - filter_size+1, 1,1]

pooled = tf.nn.max_pool(result_with_bias, ksize=[1, 5-2+1, 1, 1], strides=[1,1,1,1],  padding="VALID")


# concat test

a = tf.constant([[[[ 55. , 29.]]]], dtype=tf.float32)
b = tf.constant([[[[ 55. , 29.]]]], dtype=tf.float32)

c = [a,b]

h_pool = tf.concat(c, 3)


sess = tf.Session()

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

print("POOLED")
print(sess.run(pooled))


print("a")
print(sess.run(a))


print("b")
print(sess.run(b))

print("c")
print(sess.run(c))

print("h_pool")
print(sess.run(h_pool))
