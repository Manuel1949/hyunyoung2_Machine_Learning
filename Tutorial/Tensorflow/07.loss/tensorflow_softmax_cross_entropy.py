import numpy as np
import tensorflow as tf


### tensorflow basic batch size 

x_1d = [[1,2,3]]

y_1d = [[0,1,0]]

s_y_1d = [1]

x_2d_1 = [[4,5,6]]

y_2d_1 = [[1,0,0]]

s_y_2d_1 = [1]

x_2d_2 = [[1,2,3],[4,5,6]]

y_2d_2 = [[0,1,0],[1,0,0]]

s_y_2d_2 = [1,0]

def numpy_version():
    def stable_softmax(x):
        """Numeriacal stable version

        reference: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
        """
        e_x = np.exp(x-np.max(x))
        sum_e_x = e_x.sum(axis=-1, keepdims=True)
        print("=== e_x: {} ===".format(e_x))
        #print("=== sum_e_x: {} ===".format(sum_e_x))
        print("=== sum_e_x with axis=-1 and keepdims=True: {} ===".format(e_x.sum(axis=-1, keepdims=True)))
        div = np.divide(e_x, sum_e_x)
        print("=== div: {} ===".format(div))
        return div

    def softmax(x):
        """ normal version 
        """
  
        e_x = np.exp(x)
        sum_e_x = e_x.sum(axis=-1, keepdims=True)
        print("=== e_x:  {} ===".format(e_x))
        #print("=== sum_e_x: {} ===".format(sum_e_x))
        print("=== sum_e_x with axis=-1 and keepdims=True: {}".format(e_x.sum(axis=-1, keepdims=True)))
        div = np.divide(e_x, sum_e_x)
        print("=== div: {} ===".format(div))
        return div

    def cross_entropy(x, y):
        if len(y.shape) == 1:
           m = 1
        else:
           m = y.shape[0]
        print("=== y shape: {} ===".format(y.shape))
        # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing
        log_likelihood = -np.log(x)
        print("=== log_likelihood: {} ===".format(log_likelihood))
        
        negative_log = log_likelihood * y

        print("=== negative log: {} ===".format(negative_log))

        p_sum = np.sum(negative_log)
   
        print("=== p sum: {} ===".format(p_sum))

        loss = p_sum / m
        print("=== loss: {} ===".format(loss))
        return loss 

    x_1 = np.array(x_1d, dtype=np.float32)
    y_1 = np.array(y_1d, dtype=np.int32)
   
    x_2_1 = np.array(x_2d_1, dtype=np.float32)
    y_2_1 = np.array(y_2d_1, dtype=np.int32)

    x_2_2 = np.array(x_2d_2, dtype=np.float32)
    y_2_2 = np.array(y_2d_2, dtype=np.int32)

    print("\n=== softmax test ===")

    print("\n=== x_1d: {} ===".format(x_1d))

    print("\n=== Stable version ===")
    x_1_stable = stable_softmax(x_1)
    print("=== stabler ver: {} ===".format(x_1_stable))

    print("\n=== y_1d: {} ===".format(y_1d))
    cross_entropy(x_1_stable, y_1)

    print("\n=== Normal version ===")
    x_1_normal = softmax(x_1)
    print("=== normal ver: {} ===".format(x_1_normal))

    print("\n=== y_1d: {} ===".format(y_1d))
    cross_entropy(x_1_normal, y_1)  

    print("\n=== x_2d_1: {} ===".format(x_2d_1))
  
    print("\n=== Stable version ===")
    x_2_1_stable = stable_softmax(x_2_1)
    print("=== stabler ver: {} ===".format(x_2_1_stable))

    print("\n=== y_2d_1: {} ===".format(y_2d_1))
    cross_entropy(x_2_1_stable, y_2_1)

    print("\n=== Normal version ===")
    x_2_1_normal = softmax(x_2_1)
    print("=== normal ver:{}".format(x_2_1_normal))

    print("\n=== y_2d_1: {} ===".format(y_2d_1))
    cross_entropy(x_2_1_normal, y_2_1)

    print("\n=== x_2d_2:\n{} ===".format(x_2d_2))

    print("\n=== Stable version ===")
    x_2_2_stable = stable_softmax(x_2_2)
    print("=== stabler ver:\n {} ===\n".format(x_2_2_stable))

    print("\n=== y_2d_2: {} ===".format(y_2d_2))
    cross_entropy(x_2_2_stable, y_2_2)

    print("\n=== Normal version ===")
    x_2_2_normal = softmax(x_2_2)
    print("=== normal ver:\n{} ===".format(x_2_2_normal))

    print("\n=== y_2d_2: {} ===".format(y_2d_2))
    cross_entropy(x_2_2_normal, y_2_2)
  

def tensorflow_version():
    sess = tf.Session()

    input_x_1 = tf.constant(x_1d, dtype=tf.float32)
    label_y_1 = tf.constant(y_1d, dtype=tf.int32)
    s_label_y_1 = tf.constant(s_y_1d, dtype=tf.int32)
    y_1 = tf.constant(y_1d, dtype=tf.float32)

    s_y_1 = tf.constant(s_y_1d, dtype=tf.float32)

    input_x_2_1 = tf.constant(x_2d_1, dtype=tf.float32)
    label_y_2_1 = tf.constant(y_2d_1, dtype=tf.int32)
    s_label_y_2_1 = tf.constant(s_y_2d_1, dtype=tf.int32)
    y_2_1 = tf.constant(y_2d_1, dtype=tf.float32)
   
    input_x_2_2 = tf.constant(x_2d_2, dtype=tf.float32)
    label_y_2_2 = tf.constant(y_2d_2, dtype=tf.int32)
    s_label_y_2_2 = tf.constant(s_y_2d_2, dtype=tf.int32)
    y_2_2 = tf.constant(y_2d_2, dtype=tf.float32)



    softmax_cross_entropy_x_1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_y_1, logits=input_x_1)
    softmax_cross_entropy_x_2_1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_y_2_1, logits=input_x_2_1)
    softmax_cross_entropy_x_2_2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_y_2_2, logits=input_x_2_2)

    s_softmax_cross_entropy_x_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=s_label_y_1, logits=input_x_1)
    s_softmax_cross_entropy_x_2_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=s_label_y_2_1, logits=input_x_2_1)
    s_softmax_cross_entropy_x_2_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=s_label_y_2_2, logits=input_x_2_2)

    def _softmax(_x):
        # axis is by defalt -1 which indicate the last dimension 
        softmax = tf.nn.softmax(logits=_x, axis=None, name=None)

        raw_ver_softmax = tf.div(tf.exp(_x), tf.reduce_sum(tf.exp(_x), axis=-1, keepdims=True))
        return softmax, raw_ver_softmax

    def _cross_entropy(_x, _y):
        neg_log = -tf.log(_x)
        mul = tf.multiply(neg_log, _y)
        reduced_sum = tf.reduce_sum(mul, axis=-1)

        return neg_log, mul, reduced_sum 


    softmax_x_1, raw_ver_softmax_x_1 = _softmax(input_x_1)
    neg_log_x_1, mul_x_1, reduced_sum_x_1 = _cross_entropy(softmax_x_1, y_1)

    softmax_x_2_1, raw_ver_softmax_x_2_1 = _softmax(input_x_2_1)
    neg_log_x_2_1, mul_x_2_1, reduced_sum_x_2_1 = _cross_entropy(softmax_x_2_1, y_2_1)

    softmax_x_2_2, raw_ver_softmax_x_2_2 = _softmax(input_x_2_2)
    neg_log_x_2_2, mul_x_2_2, reduced_sum_x_2_2 = _cross_entropy(softmax_x_2_2, y_2_2)



    print("\n=== input_x_1 ===")
    print(sess.run(input_x_1))
    print("=== label_y_1 ===")
    print(sess.run(label_y_1))
    print("=== softmax ===")
    print(sess.run(softmax_x_1))
    print("=== raw softmax ===")
    print(sess.run(raw_ver_softmax_x_1))
    print("=== neg log ===")
    print(sess.run(neg_log_x_1))
    print("=== multiplye(_x, _y)")
    print(sess.run(mul_x_1))
    print("=== reduce_sum(multiply(_x,_y))")
    print(sess.run(reduced_sum_x_1))
    print("=== tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_y_1, logits=input_x_1) ===")
    print(sess.run(softmax_cross_entropy_x_1))

    print("\n=== input_x_2_1 ===")
    print(sess.run(input_x_2_1))
    print("=== label_y_2_1 ===")
    print(sess.run(label_y_2_1))  
    print("=== softmax ===")
    print(sess.run(softmax_x_2_1))
    print("=== raw softmax ===")
    print(sess.run(raw_ver_softmax_x_2_1))
    print("=== neg log ===")
    print(sess.run(neg_log_x_2_1))
    print("=== multiplye(_x, _y)")
    print(sess.run(mul_x_2_1))
    print("=== reduce_sum(multiply(_x,_y))")
    print(sess.run(reduced_sum_x_2_1))
    print("=== tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_y_2_1, logits=input_x_2_1) ===")
    print(sess.run(softmax_cross_entropy_x_2_1)) 


    print("\n=== input_x_2_2 ===")
    print(sess.run(input_x_2_2))
    print("=== label_y_2_1 ===")
    print(sess.run(label_y_2_2))
    print("=== softmax ===")
    print(sess.run(softmax_x_2_2))
    print("=== raw softmax ===")
    print(sess.run(raw_ver_softmax_x_2_2))
    print("=== neg log ===")
    print(sess.run(neg_log_x_2_2))
    print("=== multiplye(_x, _y)")
    print(sess.run(mul_x_2_2))
    print("=== reduce_sum(multiply(_x,_y))")
    print(sess.run(reduced_sum_x_2_2))
    print("=== tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_y_2_2, logits=input_x_2_2) ===")
    print(sess.run(softmax_cross_entropy_x_2_2))

    print("\n=== input_x_1 ===")
    print(sess.run(input_x_1))
    print("=== s_label_y_1 ===")
    print(sess.run(s_label_y_1))
    print("=== tf.nn.sparse_softmax_cross_entropy_with_logits_v2(labels=s_label_y_1, logits=input_x_1) ===")
    print(sess.run(s_softmax_cross_entropy_x_1))

    print("\n=== input_x_2_1 ===")
    print(sess.run(input_x_2_1))
    print("=== s_label_y_2_1 ===")
    print(sess.run(s_label_y_2_1))  
    print("=== tf.nn.sparse_softmax_cross_entropy_with_logits_v2(labels=s_label_y_2_1, logits=input_x_2_1) ===")
    print(sess.run(s_softmax_cross_entropy_x_2_1)) 


    print("\n=== input_x_2_2 ===")
    print(sess.run(input_x_2_2))
    print("=== s_label_y_2_2 ===")
    print(sess.run(s_label_y_2_2))
    print("=== tf.nn.softmax_cross_entropy_with_logits_v2(labels=s_label_y_2_2, logits=input_x_2_2) ===")
    print(sess.run(s_softmax_cross_entropy_x_2_2))


    sess.close()










if __name__=="__main__":
   numpy_version()

   tensorflow_version()
