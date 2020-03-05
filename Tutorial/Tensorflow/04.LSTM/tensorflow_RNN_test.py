#-*- codding: utf-8 -*-


import tensorflow as tf
import numpy as np


def test_RNN_layer1():
    forward_rnn_layers = [tf.nn.rnn_cell.BasicRNNCell(size, activation=tf.keras.activations.linear, dtype=tf.float32) for size in [2]]#,2]]
    #backward_rnn_layers = [tf.nn.rnn_cell.BasicRNNCell(size, activation=tf.keras.activations.linear, dtype=tf.float32) for size in [2,2]]
 
    forward_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(forward_rnn_layers)
    #backward_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(backward_rnn_layers)

    x = tf.constant([[[1,2],[3,4]]], dtype=tf.float32)

    uni_output, uni_state = tf.nn.dynamic_rnn(cell=forward_multi_rnn_cell, inputs=x, sequence_length=[1], dtype=tf.float32)
    bi_output, bi_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = forward_multi_rnn_cell,
                                                          cell_bw = forward_multi_rnn_cell,#backward_multi_rnn_cell, 
                                                          inputs=x,
                                                          sequence_length=[1],
                                                          dtype=tf.float32)

    init_op = tf.global_variables_initializer()

    tf_global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    sess = tf.Session()

    sess.run(init_op)

    print("\n=== Trainable Variables ===")
    trainable_vars = sess.run(tf_global_variables)
    print(trainable_vars)

    print("\n=== Trainable Variable name ===")
    for idx, val in enumerate(tf_global_variables):
       print("{}-{}: {}".format(idx, val.name, val))

    ## RNN setting
    fw_rnn_w1, fw_rnn_b1 = trainable_vars 

    print("\n=== RNN layer1 ===")
    print("\n=== fw rnn weight1 ===")
    print(fw_rnn_w1)
   
    print("\n=== fw rnn bias ===")
    print(fw_rnn_b1)

    print("\n=== x ===\n{}".format(sess.run(x)))

    x1 = np.array([1.,2.], dtype=np.float32)
    print("\n=== x1 ===\n{}".format(x1))

    x2 = np.array([3.,4.], dtype=np.float32)
    print("\n=== x2 ===\n{}".format(x2))

    h0 = np.array([0.,0.], dtype=np.float32)
    print("\n===  h0 in layer1 ===\n{}".format(h0))

    input1_x1 = np.concatenate((x1, h0), axis=-1)
    print("\n=== real input1 x1: concat(x1, h0) === \n{}".format(input1_x1))

    layer1_h1 = np.add(input1_x1.dot(fw_rnn_w1), fw_rnn_b1)
    print("\n=== h1 in layer1 (np.add(input1_x1.dot(fw_rnn_w1), fw_rnn_b1)) === \n{}".format(layer1_h1))

    input1_x2 = np.concatenate((x2, layer1_h1),  axis=-1)
    print("\n=== real input1 x2 (concat(x2, layer1_h1) ===\n{}".format(input1_x2))

    layer1_h2 = np.add(input1_x2.dot(fw_rnn_w1), fw_rnn_b1)
    print("\n=== h2 in layer1 (np.add(input1_x2.dot(fw_rnn_w1), fw_rnn_b1)) ===\n{}".format(layer1_h2))

    print("\n=== rnn output ===\n{}".format(sess.run(uni_output)))

    print("\n=== rnn state ===\n{}".format(sess.run(uni_state)))

    print("\n=== x ===\n{}".format(sess.run(x)))

    x1 = np.array([3.,4.], dtype=np.float32)
    print("\n=== x1 ===\n{}".format(x1))

    x2 = np.array([1.,2.], dtype=np.float32)
    print("\n=== x2 ===\n{}".format(x2))

    h0 = np.array([0.,0.], dtype=np.float32)
    print("\n===  h0 in layer1 ===\n{}".format(h0))

    input1_x1 = np.concatenate((x1, h0), axis=-1)
    print("\n=== real input1 x1: concat(x1, h0) === \n{}".format(input1_x1))

    layer1_h1 = np.add(input1_x1.dot(fw_rnn_w1), fw_rnn_b1)
    print("\n=== h1 in layer1 (np.add(input1_x1.dot(fw_rnn_w1), fw_rnn_b1)) === \n{}".format(layer1_h1))

    input1_x2 = np.concatenate((x2, layer1_h1),  axis=-1)
    print("\n=== real input1 x2 (concat(x2, layer1_h1) ===\n{}".format(input1_x2))

    layer1_h2 = np.add(input1_x2.dot(fw_rnn_w1), fw_rnn_b1)
    print("\n=== h2 in layer1 (np.add(input1_x2.dot(fw_rnn_w1), fw_rnn_b1)) ===\n{}".format(layer1_h2))

    print("\n=== bi rnn output ===\n{}".format(sess.run(bi_output)))

    print("\n=== bi state output ===\n{}".format(sess.run(bi_state)))

    sess.close()

def test_RNN_layer2():
    forward_rnn_layers = [tf.nn.rnn_cell.BasicRNNCell(size, activation=tf.keras.activations.linear, dtype=tf.float32) for size in [2,2]]
    #backward_rnn_layers = [tf.nn.rnn_cell.BasicRNNCell(size, activation=tf.keras.activations.linear, dtype=tf.float32) for size in [2,2]]

    forward_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(forward_rnn_layers)
    #backward_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(backward_rnn_layers)

    x = tf.constant([[[1,2],[3,4]]], dtype=tf.float32)

    uni_output, uni_state = tf.nn.dynamic_rnn(cell=forward_multi_rnn_cell, sequenc_length=[1], inputs=x, dtype=tf.float32)
    bi_output, bi_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = forward_multi_rnn_cell,
                                                          cell_bw = forward_multi_rnn_cell,#backward_multi_rnn_cell, 
                                                          inputs=x,
                                                          sequence_length=[2],
                                                          dtype=tf.float32)

    init_op = tf.global_variables_initializer()

    tf_global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    sess = tf.Session()

    sess.run(init_op)

    print("\n=== Trainable Variables ===")
    trainable_vars = sess.run(tf_global_variables)
    print(trainable_vars)

    print("\n=== Trainable Variable name ===")
    for idx, val in enumerate(tf_global_variables):
       print("{}-{}: {}".format(idx, val.name, val))

    print("\n=== RNN layer 2 ===")
    ## RNN setting
    fw_rnn_w1, fw_rnn_b1, fw_rnn_w2, fw_rnn_b2 = trainable_vars #, bw_rnn_w1,  bw_rnn_b1  bw_rnn_w2, bw_rnn_b2 = trainable_vars

    print("=== fw_rnn_weight ===")
    print(fw_rnn_w1)
    print("=== fw_rnn_bias ===")
    print(fw_rnn_b1)


    print("\n=== x ===\n{}".format(sess.run(x)))

    x1 = np.array([1.,2.], dtype=np.float32)
    print("\n=== x1 ===\n{}".format(x1))

    x2 = np.array([3.,4.], dtype=np.float32)
    print("\n=== x2 ===\n{}".format(x2))

    h0 = np.array([0.,0.], dtype=np.float32)
    print("\n===  h0 in layer1 ===\n{}".format(h0))

    input1_x1 = np.concatenate((x1, h0), axis=-1)
    print("\n=== real input1 x1: concat(x1, h0) === \n{}".format(input1_x1))

    layer1_h1 = np.add(input1_x1.dot(fw_rnn_w1), fw_rnn_b1)
    print("\n=== h1 in layer1 (np.add(input1_x1.dot(fw_rnn_w1), fw_rnn_b1)) === \n{}".format(layer1_h1))

    input1_x2 = np.concatenate((x2, layer1_h1),  axis=-1)
    print("\n=== real input1 x2 (concat(x2, layer1_h1) ===\n{}".format(input1_x2))

    layer1_h2 = np.add(input1_x2.dot(fw_rnn_w1), fw_rnn_b1)
    print("\n=== h2 in layer1 (np.add(input1_x2.dot(fw_rnn_w1), fw_rnn_b1)) ===\n{}".format(layer1_h2))

    layer2_h0 = h0
    print("\n=== h0 in layer2 ===\n{}".format(layer2_h0))

    input2_x1 = np.concatenate((layer1_h1, layer2_h0), axis=-1)
    print("\n=== real input2 x1: concat(layer1_h1, layer2_h0) ===\n{}".format(input2_x1))

    layer2_h1 = np.add(input2_x1.dot(fw_rnn_w2), fw_rnn_b2)
    print("\n=== h1 in layer 2 (np.add(input2_x1.dot(fw_rnn_w2), fw_rnn_b2)) ===\n{}".format(layer2_h1))

    input2_x2 = np.concatenate((layer1_h2, layer2_h1))
    print("\n=== reali input2 x2: concat(layer1_h2, layer2_h1) ===\n{}".format(input2_x2))

    layer2_h2 = np.add(input2_x2.dot(fw_rnn_w2), fw_rnn_b2)
    print("\n=== h2 in layer 2: (np.add(input2_x2.dot(fw_rnn_w2), fw_rnn_b2)) ===\n{}".format(layer2_h2))

    print("\n=== rnn output ===\n{}".format(sess.run(uni_output)))

    print("\n=== rnn state ===\n{}".format(sess.run(uni_state)))

    print("\n=== x ===\n{}".format(sess.run(x)))

    x1 = np.array([3.,4.], dtype=np.float32)
    print("\n=== x1 ===\n{}".format(x1))

    x2 = np.array([1.,2.], dtype=np.float32)
    print("\n=== x2 ===\n{}".format(x2))

    h0 = np.array([0.,0.], dtype=np.float32)
    print("\n===  h0 in layer1 ===\n{}".format(h0))

    input1_x1 = np.concatenate((x1, h0), axis=-1)
    print("\n=== real input1 x1: concat(x1, h0) === \n{}".format(input1_x1))

    layer1_h1 = np.add(input1_x1.dot(fw_rnn_w1), fw_rnn_b1)
    print("\n=== h1 in layer1 (np.add(input1_x1.dot(fw_rnn_w2), fw_rnn_b2)) === \n{}".format(layer1_h1))

    input1_x2 = np.concatenate((x2, layer1_h1),  axis=-1)
    print("\n=== real input1 x2 (concat(x2, layer1_h1) ===\n{}".format(input1_x2))

    layer1_h2 = np.add(input1_x2.dot(fw_rnn_w1), fw_rnn_b1)
    print("\n=== h2 in layer1 (np.add(input1_x2.dot(fw_rnn_w1), fw_rnn_b1)) ===\n{}".format(layer1_h2))

    layer2_h0 = h0
    print("\n=== h0 in layer2 ===\n{}".format(layer2_h0))

    input2_x1 = np.concatenate((layer1_h1, layer2_h0), axis=-1)
    print("\n=== real input2 x1: concat(layer1_h1, layer2_h0) ===\n{}".format(input2_x1))

    layer2_h1 = np.add(input2_x1.dot(fw_rnn_w2), fw_rnn_b2)
    print("\n=== h1 in layer 2 (np.add(input2_x1.dot(fw_rnn_w2), fw_rnn_b2)) ===\n{}".format(layer2_h1))

    input2_x2 = np.concatenate((layer1_h2, layer2_h1))
    print("\n=== reali input2 x2: concat(layer1_h2, layer2_h1) ===\n{}".format(input2_x2))

    layer2_h2 = np.add(input2_x2.dot(fw_rnn_w2), fw_rnn_b2)
    print("\n=== h2 in layer 2: (np.add(input2_x2.dot(fw_rnn_w2), fw_rnn_b2)) ===\n{}".format(layer2_h2))

    print("\n=== bi rnn output ===\n{}".format(sess.run(bi_output)))

    print("\n=== bi state output ===\n{}".format(sess.run(bi_state)))

    sess.close()

def test_LSTM_layer1():
    forward_rnn_layers = [tf.nn.rnn_cell.BasicLSTMCell(size, activation=tf.keras.activations.linear, state_is_tuple=True, dtype=tf.float32) for size in [2]]#,2]]
    #backward_rnn_layers = [tf.nn.rnn_cell.BasicRNNCell(size, activation=tf.keras.activations.linear, dtype=tf.float32) for size in [2,2]]

    forward_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(forward_rnn_layers)
    #backward_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(backward_rnn_layers)

    x = tf.constant([[[1,2],[3,4]]], dtype=tf.float32)

    uni_output, uni_state = tf.nn.dynamic_rnn(cell=forward_multi_rnn_cell, sequence_length=[2], inputs=x, dtype=tf.float32)
    bi_output, bi_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = forward_multi_rnn_cell,
                                                          cell_bw = forward_multi_rnn_cell,#backward_multi_rnn_cell, 
                                                          inputs=x,
                                                          sequence_length=[2],
                                                          dtype=tf.float32)

    init_op = tf.global_variables_initializer()

    tf_global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    sess = tf.Session()

    sess.run(init_op)

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    print("\n=== Trainable Variables ===")
    trainable_vars = sess.run(tf_global_variables)
    print(trainable_vars)

    print("\n=== Trainable Variable name ===")
    for idx, val in enumerate(tf_global_variables):
       print("{}-{}: {}".format(idx, val.name, val))

    print("\n=== RNN layer 2 ===")

    ## LSTM setting
    fw_rnn_w1, fw_rnn_b1 = trainable_vars #fw_rnn_w2, fw_rnn_b2 = trainable_vars #, bw_rnn_w1,  bw_rnn_b1  bw_rnn_w2, bw_rnn_b2 = trainable_vars

    print("=== fw_rnn_weight ===")
    print(fw_rnn_w1)
    print("=== fw_rnn_bias ===")
    print(fw_rnn_b1)


    fw_rnn_w1_input = fw_rnn_w1[:,0:2]
    fw_rnn_w1_new_input = fw_rnn_w1[:,2:4]
    fw_rnn_w1_forget = fw_rnn_w1[:, 4:6]
    fw_rnn_w1_output = fw_rnn_w1[:,6:8]

    print("\n=== w1 input gate ===")
    print(fw_rnn_w1_input)
    print("\n=== w1 new input ===")
    print(fw_rnn_w1_new_input)
    print("\n=== w1 forget gate ===")
    print(fw_rnn_w1_forget)
    print("\n=== w1 ouput gate ===")
    print(fw_rnn_w1_output)
    print("\n=== b1 ===")
    print(fw_rnn_b1)
    print("\n==== forger bias in tensorflow ====")    
    forget_bias=1.0
    print(forget_bias)
  
    print("\n=== x ===\n{}".format(sess.run(x)))

    x1 = np.array([1.,2.], dtype=np.float32)
    print("\n=== x1 ===\n{}".format(x1))

    x2 = np.array([3.,4.], dtype=np.float32)
    print("\n=== x2 ===\n{}".format(x2))

    h0 = np.array([0., 0.], dtype=np.float32)
    print("\n===  h0 in layer1 ===\n{}".format(h0))

    c0_layer1 = np.array([0., 0.], dtype=np.float32)
    print("\n=== c0 in layer 1 ===\n{}".format(c0_layer1))

    input1_x1 = np.concatenate((x1, h0), axis=-1)
    print("\n=== real input1 x1: concat(x1, h0) === \n{}".format(input1_x1))

    gate1_h1 = np.add(input1_x1.dot(fw_rnn_w1), fw_rnn_b1)
    print("\n=== gete1_h1 in layer1 (np.add(input1_x1.dot(fw_rnn_w1), fw_rnn_b1)) === \n{}".format(gate1_h1))

    c1_layer1 = np.add(np.multiply(c0_layer1, sigmoid(np.add(gate1_h1[4:6],forget_bias))),
                       np.multiply(gate1_h1[2:4], sigmoid(gate1_h1[0:2])))

    print("\n=== c1_layer1 (np.add(np.multiply(c1_layer1, sigmoid(np.add(gate1_h1[4:6], foget_bias))), np.multiply(gate1_h1[2:4], sigmoid(gate1_h1[0:2]))) ===\n{}".format(c1_layer1))

    layer1_h1 = np.multiply(sigmoid(gate1_h1[6:8]), c1_layer1)
     
    print("\n=== layer1_h1 ===\n{}".format(layer1_h1)) 

    input1_x2 = np.concatenate((x2, layer1_h1),  axis=-1)
    print("\n=== real input1 x2 (concat(x2, layer1_h1) ===\n{}".format(input1_x2))

    gate1_h2 = np.add(input1_x2.dot(fw_rnn_w1), fw_rnn_b1)
    print("\n=== gate1_h2 in layer1 (np.add(input1_x2.dot(fw_rnn_w1), fw_rnn_b1)) ===\n{}".format(gate1_h2))

    c2_layer1 = np.add(np.multiply(c1_layer1, sigmoid(np.add(gate1_h2[4:6], forget_bias))),
                       np.multiply(gate1_h2[2:4], sigmoid(gate1_h2[0:2])))

    print("\n=== c2_layer1 (np.add(np.multiply(c1_layer1, sigmoid(np.add(gate1_h2[4:6], forget_bias))), np.multiply(gate1_h2[2:4], sigmoid(gate1_h2[0:2]))) ===\n{}".format(c2_layer1))

    layer1_h2 = np.multiply(sigmoid(gate1_h2[6:8]), c2_layer1)
    print("\n=== layer1_h2 ===\n{}".format(layer1_h2))
     
    print("\n=== rnn output ===\n{}".format(sess.run(uni_output)))

    print("\n=== rnn state ===\n{}".format(sess.run(uni_state)))

    print("\n=== x ===\n{}".format(sess.run(x)))

    x1 = np.array([3.,4.], dtype=np.float32)
    print("\n=== x1 ===\n{}".format(x1))

    x2 = np.array([1.,2.], dtype=np.float32)
    print("\n=== x2 ===\n{}".format(x2))

    h0 = np.array([0.,0.], dtype=np.float32)
    print("\n===  h0 in layer1 ===\n{}".format(h0))

    c0_layer1 = np.array([0., 0.], dtype=np.float32)
    print("\n=== c0 in layer 1 ===\n{}".format(c0_layer1))

    input1_x1 = np.concatenate((x1, h0), axis=-1)
    print("\n=== real input1 x1: concat(x1, h0) === \n{}".format(input1_x1))

    gate1_h1 = np.add(input1_x1.dot(fw_rnn_w1), fw_rnn_b1)
    print("\n=== gete1_h1 in layer1 (np.add(input1_x1.dot(fw_rnn_w1), fw_rnn_b1)) === \n{}".format(gate1_h1))

    c1_layer1 = np.add(np.multiply(c0_layer1, sigmoid(np.add(gate1_h1[4:6],forget_bias))),
                       np.multiply(gate1_h1[2:4], sigmoid(gate1_h1[0:2])))

    print("\n=== c1_layer1 (np.add(np.multiply(c1_layer1, sigmoid(np.add(gate1_h1[4:6], foget_bias))), np.multiply(gate1_h1[2:4], sigmoid(gate1_h1[0:2]))) ===\n{}".format(c1_layer1))

    layer1_h1 = np.multiply(sigmoid(gate1_h1[6:8]), c1_layer1)
     
    print("\n=== layer1_h1 ===\n{}".format(layer1_h1)) 

    input1_x2 = np.concatenate((x2, layer1_h1),  axis=-1)
    print("\n=== real input1 x2 (concat(x2, layer1_h1) ===\n{}".format(input1_x2))

    gate1_h2 = np.add(input1_x2.dot(fw_rnn_w1), fw_rnn_b1)
    print("\n=== gate1_h2 in layer1 (np.add(input1_x2.dot(fw_rnn_w1), fw_rnn_b1)) ===\n{}".format(gate1_h2))

    c2_layer1 = np.add(np.multiply(c1_layer1, sigmoid(np.add(gate1_h2[4:6], forget_bias))),
                       np.multiply(gate1_h2[2:4], sigmoid(gate1_h2[0:2])))

    print("\n=== c2_layer1 (np.add(np.multiply(c1_layer1, sigmoid(np.add(gate1_h2[4:6], forget_bias))), np.multiply(gate1_h2[2:4], sigmoid(gate1_h2[0:2]))) ===\n{}".format(c2_layer1))

    layer1_h2 = np.multiply(sigmoid(gate1_h2[6:8]), c2_layer1)
    print("\n=== layer1_h2 ===\n{}".format(layer1_h2))

    print("\n=== bi rnn output ===\n{}".format(sess.run(bi_output)))

    print("\n=== bi state output ===\n{}".format(sess.run(bi_state)))




def test_LSTM_layer2():
    forward_rnn_layers = [tf.nn.rnn_cell.BasicLSTMCell(size, activation=tf.keras.activations.linear, state_is_tuple=True, dtype=tf.float32) for size in [2,2]]
    #backward_rnn_layers = [tf.nn.rnn_cell.BasicRNNCell(size, activation=tf.keras.activations.linear, dtype=tf.float32) for size in [2,2]]

    forward_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(forward_rnn_layers)
    #backward_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(backward_rnn_layers)

    x = tf.constant([[[1,2],[3,4]]], dtype=tf.float32)

    uni_output, uni_state = tf.nn.dynamic_rnn(cell=forward_multi_rnn_cell, inputs=x, dtype=tf.float32)
    bi_output, bi_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = forward_multi_rnn_cell,
                                                          cell_bw = forward_multi_rnn_cell,#backward_multi_rnn_cell, 
                                                          inputs=x,
                                                          sequence_length=[2],
                                                          dtype=tf.float32)

    init_op = tf.global_variables_initializer()

    tf_global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    sess = tf.Session()

    sess.run(init_op)

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    print("\n=== Trainable Variables ===")
    trainable_vars = sess.run(tf_global_variables)
    print(trainable_vars)

    print("\n=== Trainable Variable name ===")
    for idx, val in enumerate(tf_global_variables):
       print("{}-{}: {}".format(idx, val.name, val))

    print("\n=== RNN layer 2 ===")

    ## LSTM setting
    fw_rnn_w1, fw_rnn_b1, fw_rnn_w2, fw_rnn_b2 = trainable_vars #, bw_rnn_w1,  bw_rnn_b1  bw_rnn_w2, bw_rnn_b2 = trainable_vars

    print("=== fw_rnn_weight ===")
    print(fw_rnn_w1)
    print("=== fw_rnn_bias ===")
    print(fw_rnn_b1)


    fw_rnn_w1_input = fw_rnn_w1[:,0:2]
    fw_rnn_w1_new_input = fw_rnn_w1[:,2:4]
    fw_rnn_w1_forget = fw_rnn_w1[:, 4:6]
    fw_rnn_w1_output = fw_rnn_w1[:,6:8]

    print("\n=== w1 input gate ===")
    print(fw_rnn_w1_input)
    print("\n=== w1 new input ===")
    print(fw_rnn_w1_new_input)
    print("\n=== w1 forget gate ===")
    print(fw_rnn_w1_forget)
    print("\n=== w1 ouput gate ===")
    print(fw_rnn_w1_output)
    print("\n=== b1 ===")
    print(fw_rnn_b1)
    print("\n==== forger bias in tensorflow ====")    
    forget_bias=1.0   
    print(forget_bias)

    print("\n=== x ===\n{}".format(sess.run(x)))

    x1 = np.array([1.,2.], dtype=np.float32)
    print("\n=== x1 ===\n{}".format(x1))

    x2 = np.array([3.,4.], dtype=np.float32)
    print("\n=== x2 ===\n{}".format(x2))

    h0 = np.array([0., 0.], dtype=np.float32)
    print("\n===  h0 in layer1 ===\n{}".format(h0))

    c0_layer1 = np.array([0., 0.], dtype=np.float32)
    print("\n=== c0 in layer 1 ===\n{}".format(c0_layer1))

    input1_x1 = np.concatenate((x1, h0), axis=-1)
    print("\n=== real input1 x1: concat(x1, h0) === \n{}".format(input1_x1))

    gate1_h1 = np.add(input1_x1.dot(fw_rnn_w1), fw_rnn_b1)
    print("\n=== gete1_h1 in layer1 (np.add(input1_x1.dot(fw_rnn_w1), fw_rnn_b1)) === \n{}".format(gate1_h1))

    c1_layer1 = np.add(np.multiply(c0_layer1, sigmoid(np.add(gate1_h1[4:6], forget_bias))),
                       np.multiply(gate1_h1[2:4], sigmoid(gate1_h1[0:2])))

    print("\n=== c1_layer1 (np.add(np.multiply(c1_layer1, sigmoid(np.add(gate1_h1[4:6],forget_bias))), np.multiply(gate1_h1[2:4], sigmoid(gate1_h1[0:2]))) ===\n{}".format(c1_layer1))

    layer1_h1 = np.multiply(sigmoid(gate1_h1[6:8]), c1_layer1)
     
    print("\n=== layer1_h1 ===\n{}".format(layer1_h1)) 

    input1_x2 = np.concatenate((x2, layer1_h1),  axis=-1)
    print("\n=== real input1 x2 (concat(x2, layer1_h1) ===\n{}".format(input1_x2))

    gate1_h2 = np.add(input1_x2.dot(fw_rnn_w1), fw_rnn_b1)
    print("\n=== gate1_h2 in layer1 (np.add(input1_x2.dot(fw_rnn_w1), fw_rnn_b1)) ===\n{}".format(gate1_h2))

    c2_layer1 = np.add(np.multiply(c1_layer1, sigmoid(np.add(gate1_h2[4:6], forget_bias))),
                       np.multiply(gate1_h2[2:4], sigmoid(gate1_h2[0:2])))

    print("\n=== c2_layer1 (np.add(np.multiply(c1_layer1, sigmoid(np.add(gate1_h2[4:6],forget_bias))), np.multiply(gate1_h2[2:4], sigmoid(gate1_h2[0:2]))) ===\n{}".format(c2_layer1))

    layer1_h2 = np.multiply(sigmoid(gate1_h2[6:8]), c2_layer1)
     
    print("\n=== layer1_h2 ===\n{}".format(layer1_h2)) 

    layer2_h0 = h0
    print("\n=== h0 in layer2 ===\n{}".format(layer2_h0))

    c0_layer2 = c0_layer1
    print("\n=== c0 in layer2 ===\n{}".format(c0_layer2))

    input2_x1 = np.concatenate((layer1_h1, layer2_h0), axis=-1)
    print("\n=== real input2 x1: concat(layer1_h1, layer2_h0) ===\n{}".format(input2_x1))

    gate2_h1 = np.add(input2_x1.dot(fw_rnn_w2), fw_rnn_b2)
    print("\n=== gate2_h1 in layer 2 (np.add(input2_x1.dot(fw_rnn_w2), fw_rnn_b2)) ===\n{}".format(gate2_h1))

    c1_layer2 = np.add(np.multiply(c0_layer2, sigmoid(np.add(gate2_h1[4:6], forget_bias))),
                       np.multiply(gate2_h1[2:4], sigmoid(gate2_h1[0:2])))

    print("\n=== c1_layer2 (np.add(np.multiply(c0_layer2, sigmoid(gate2_h1[4:6])), np.multiply(gate2_h1[2:4], sigmoid(gate2_h1[0:2]))) ===\n{}".format(c1_layer2))

    layer2_h1 = np.multiply(sigmoid(gate2_h1[6:8]), c1_layer2)
     
    print("\n=== layer2_h1 ===\n{}".format(layer2_h1)) 

    input2_x2 = np.concatenate((layer1_h2, layer2_h1))
    print("\n=== real input2 x2: concat(layer1_h2, layer2_h1) ===\n{}".format(input2_x2))


    gate2_h2 = np.add(input2_x2.dot(fw_rnn_w2), fw_rnn_b2)
    print("\n=== gate_h2 in layer 2: (np.add(input2_x2.dot(fw_rnn_w2), fw_rnn_b2)) ===\n{}".format(gate2_h2))

    c2_layer2 = np.add(np.multiply(c1_layer2, sigmoid(np.add(gate2_h2[4:6],forget_bias))),
                       np.multiply(gate2_h2[2:4], sigmoid(gate2_h2[0:2])))

    print("\n=== c2_layer2 (np.add(np.multiply(c1_layer2, sigmoid(gate2_h2[4:6])), np.multiply(gate2_h2[2:4], sigmoid(gate2_h2[0:2]))) ===\n{}".format(c2_layer2))

    layer2_h2 = np.multiply(sigmoid(gate2_h2[6:8]), c2_layer2)
     
    print("\n=== layer2_h2 ===\n{}".format(layer2_h2)) 

    print("\n=== rnn output ===\n{}".format(sess.run(uni_output)))

    print("\n=== rnn state ===\n{}".format(sess.run(uni_state)))

    print("\n=== x ===\n{}".format(sess.run(x)))

    x1 = np.array([3.,4.], dtype=np.float32)
    print("\n=== x1 ===\n{}".format(x1))

    x2 = np.array([1.,2.], dtype=np.float32)
    print("\n=== x2 ===\n{}".format(x2))

    h0 = np.array([0.,0.], dtype=np.float32)
    print("\n===  h0 in layer1 ===\n{}".format(h0))

    c0_layer1 = np.array([0., 0.], dtype=np.float32)
    print("\n=== c0 in layer 1 ===\n{}".format(c0_layer1))

    input1_x1 = np.concatenate((x1, h0), axis=-1)
    print("\n=== real input1 x1: concat(x1, h0) === \n{}".format(input1_x1))

    gate1_h1 = np.add(input1_x1.dot(fw_rnn_w1), fw_rnn_b1)
    print("\n=== gete1_h1 in layer1 (np.add(input1_x1.dot(fw_rnn_w1), fw_rnn_b1)) === \n{}".format(gate1_h1))

    c1_layer1 = np.add(np.multiply(c0_layer1, sigmoid(np.add(gate1_h1[4:6], forget_bias))),
                       np.multiply(gate1_h1[2:4], sigmoid(gate1_h1[0:2])))

    print("\n=== c1_layer1 (np.add(np.multiply(c1_layer1, sigmoid(np.add(gate1_h1[4:6],forget_bias))), np.multiply(gate1_h1[2:4], sigmoid(gate1_h1[0:2]))) ===\n{}".format(c1_layer1))

    layer1_h1 = np.multiply(sigmoid(gate1_h1[6:8]), c1_layer1)
     
    print("\n=== layer1_h1 ===\n{}".format(layer1_h1)) 

    input1_x2 = np.concatenate((x2, layer1_h1),  axis=-1)
    print("\n=== real input1 x2 (concat(x2, layer1_h1) ===\n{}".format(input1_x2))

    gate1_h2 = np.add(input1_x2.dot(fw_rnn_w1), fw_rnn_b1)
    print("\n=== gate1_h2 in layer1 (np.add(input1_x2.dot(fw_rnn_w1), fw_rnn_b1)) ===\n{}".format(gate1_h2))

    c2_layer1 = np.add(np.multiply(c1_layer1, sigmoid(np.add(gate1_h2[4:6], forget_bias))),
                       np.multiply(gate1_h2[2:4], sigmoid(gate1_h2[0:2])))

    print("\n=== c2_layer1 (np.add(np.multiply(c1_layer1, sigmoid(np.add(gate1_h2[4:6],forget_bias))), np.multiply(gate1_h2[2:4], sigmoid(gate1_h2[0:2]))) ===\n{}".format(c2_layer1))

    layer1_h2 = np.multiply(sigmoid(gate1_h2[6:8]), c2_layer1)
     
    print("\n=== layer1_h2 ===\n{}".format(layer1_h2)) 

    layer2_h0 = h0
    print("\n=== h0 in layer2 ===\n{}".format(layer2_h0))

    c0_layer2 = c0_layer1
    print("\n=== c0 in layer2 ===\n{}".format(c0_layer2))

    input2_x1 = np.concatenate((layer1_h1, layer2_h0), axis=-1)
    print("\n=== real input2 x1: concat(layer1_h1, layer2_h0) ===\n{}".format(input2_x1))

    gate2_h1 = np.add(input2_x1.dot(fw_rnn_w2), fw_rnn_b2)
    print("\n=== gate2_h1 in layer 2 (np.add(input2_x1.dot(fw_rnn_w2), fw_rnn_b2)) ===\n{}".format(gate2_h1))

    c1_layer2 = np.add(np.multiply(c0_layer2, sigmoid(np.add(gate2_h1[4:6], forget_bias))),
                       np.multiply(gate2_h1[2:4], sigmoid(gate2_h1[0:2])))

    print("\n=== c1_layer2 (np.add(np.multiply(c0_layer2, sigmoid(gate2_h1[4:6])), np.multiply(gate2_h1[2:4], sigmoid(gate2_h1[0:2]))) ===\n{}".format(c1_layer2))

    layer2_h1 = np.multiply(sigmoid(gate2_h1[6:8]), c1_layer2)
     
    print("\n=== layer2_h1 ===\n{}".format(layer2_h1)) 

    input2_x2 = np.concatenate((layer1_h2, layer2_h1))
    print("\n=== real input2 x2: concat(layer1_h2, layer2_h1) ===\n{}".format(input2_x2))


    gate2_h2 = np.add(input2_x2.dot(fw_rnn_w2), fw_rnn_b2)
    print("\n=== gate_h2 in layer 2: (np.add(input2_x2.dot(fw_rnn_w2), fw_rnn_b2)) ===\n{}".format(gate2_h2))

    c2_layer2 = np.add(np.multiply(c1_layer2, sigmoid(np.add(gate2_h2[4:6],forget_bias))),
                       np.multiply(gate2_h2[2:4], sigmoid(gate2_h2[0:2])))

    print("\n=== c2_layer2 (np.add(np.multiply(c1_layer2, sigmoid(gate2_h2[4:6])), np.multiply(gate2_h2[2:4], sigmoid(gate2_h2[0:2]))) ===\n{}".format(c2_layer2))

    layer2_h2 = np.multiply(sigmoid(gate2_h2[6:8]), c2_layer2)
     
    print("\n=== layer2_h2 ===\n{}".format(layer2_h2)) 


    print("\n=== bi rnn output ===\n{}".format(sess.run(bi_output)))

    print("\n=== bi state output ===\n{}".format(sess.run(bi_state)))

    sess.close()


def test_GRU_layer1():
    forward_rnn_layers = [tf.nn.rnn_cell.GRUCell(size, activation=tf.keras.activations.linear, dtype=tf.float32 ) for size in [2,2]]#,2]]
    #backward_rnn_layers = [tf.nn.rnn_cell.BasicRNNCell(size, activation=tf.keras.activations.linear, dtype=tf.float32) for size in [2,2]]

    forward_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(forward_rnn_layers)
    #backward_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(backward_rnn_layers)

    x = tf.constant([[[1,2],[3,4]]], dtype=tf.float32)

    uni_output, uni_state = tf.nn.dynamic_rnn(cell=forward_multi_rnn_cell, sequence_length=[2], inputs=x, dtype=tf.float32)
    bi_output, bi_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = forward_multi_rnn_cell,
                                                          cell_bw = forward_multi_rnn_cell,#backward_multi_rnn_cell, 
                                                          inputs=x,
                                                          sequence_length=[2],
                                                          dtype=tf.float32)


    config = forward_rnn_layers[0].get_config()
    config['trainable'] = False

    config1 = forward_rnn_layers[0].get_config()

    init_op = tf.global_variables_initializer()

    tf_global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    sess = tf.Session()

    sess.run(init_op)

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    print("\n=== Trainable Variables ===")
    trainable_vars = sess.run(tf_global_variables)
    print(trainable_vars)

    print("\n=== Trainable Variable name ===")
    for idx, val in enumerate(tf_global_variables):
       print("{}-{}: {}".format(idx, val.name, val))

    print("\n=== RNN layer 2 ===")

    ## LSTM setting
    gate_w1, gate_b1, candidate_w1, candidate_b1, _,_,_,_  = trainable_vars #fw_rnn_w2, fw_rnn_b2 = trainable_vars #, bw_rnn_w1,  bw_rnn_b1  bw_rnn_w2, bw_rnn_b2 = trainable_vars

    print("\n=== gate weight ===")
    print(gate_w1)
    print("\n=== gate bias ===")
    print(gate_b1)

  
    print("\n=== x ===\n{}".format(sess.run(x)))

    x1 = np.array([1.,2.], dtype=np.float32)
    print("\n=== x1 ===\n{}".format(x1))

    x2 = np.array([3.,4.], dtype=np.float32)
    print("\n=== x2 ===\n{}".format(x2))

    h0 = np.array([0., 0.], dtype=np.float32)
    print("\n===  h0 in layer1 ===\n{}".format(h0))

    input1_x1 = np.concatenate((x1, h0), axis=-1)
    print("\n=== real input1 x1: concat(x1, h0) === \n{}".format(input1_x1))

    gate1_h1 = sigmoid(np.add(input1_x1.dot(gate_w1), gate_b1))
    print("\n=== gate1_h1 in layer1 (sigmoid(np.add(input1_x1.dot(gate_w1), gate_b1))) === \n{}".format(gate1_h1))

    r1_layer1 = gate1_h1[0:2]

    z1_layer1 = gate1_h1[2:4]

    reset_h0 = np.multiply(r1_layer1, h0)
   
    print("\n=== reset_h0 ===\n{}".format(reset_h0))

    new_input1_x1 = np.concatenate((x1, reset_h0), axis=-1)
    print("\n=== new_input_x1 ===\n{}".format(new_input1_x1))

    new_candidate1 = np.add(new_input1_x1.dot(candidate_w1), candidate_b1)
    print("\n=== new_candidate1 in layer1 ===\n{}".format(new_candidate1))
    
    layer1_h1 = np.add(np.multiply(z1_layer1, h0), np.multiply((1-z1_layer1), new_candidate1))

    print("\n=== h1 in layer 1 ===\n{}".format(layer1_h1))

    input1_x2 = np.concatenate((x2, layer1_h1), axis=-1)
    print("\n=== real input1 x2: concat(x2, laye1_h1) === \n{}".format(input1_x2))

    gate1_h2 = sigmoid(np.add(input1_x2.dot(gate_w1), gate_b1))
    print("\n=== gate1_h2 in layer1 (sigmoid(np.add(input1_x2.dot(gate_w1), gate_b1))) === \n{}".format(gate1_h2))

    r2_layer1 = gate1_h2[0:2]

    z2_layer1 = gate1_h2[2:4]

    reset_h1 = np.multiply(r2_layer1, layer1_h1)
   
    print("\n=== reset_h1 ===\n{}".format(reset_h1))

    new_input1_x2 = np.concatenate((x2, reset_h1), axis=-1)
    print("\n=== new_input_x2 ===\n{}".format(new_input1_x2))

    new_candidate2 = np.add(new_input1_x2.dot(candidate_w1), candidate_b1)
    print("\n=== new_candidate2 in layer1 ===\n{}".format(new_candidate2))
    
    layer1_h2 = np.add(np.multiply(z2_layer1, layer1_h1), np.multiply((1-z2_layer1), new_candidate2))

    print("\n=== h2 in layer 1 ===\n{}".format(layer1_h2))

    print("\n=== rnn output ===\n{}".format(sess.run(uni_output)))

    print("\n=== rnn state ===\n{}".format(sess.run(uni_state)))

    print("\n=== x ===\n{}".format(sess.run(x)))

    x1 = np.array([3.,4.], dtype=np.float32)
    print("\n=== x1 ===\n{}".format(x1))

    x2 = np.array([1.,2.], dtype=np.float32)
    print("\n=== x2 ===\n{}".format(x2))

    h0 = np.array([0.,0.], dtype=np.float32)
    print("\n===  h0 in layer1 ===\n{}".format(h0))

    input1_x1 = np.concatenate((x1, h0), axis=-1)
    print("\n=== real input1 x1: concat(x1, h0) === \n{}".format(input1_x1))

    gate1_h1 = sigmoid(np.add(input1_x1.dot(gate_w1), gate_b1))
    print("\n=== gate1_h1 in layer1 (sigmoid(np.add(input1_x1.dot(gate_w1), gate_b1))) === \n{}".format(gate1_h1))

    r1_layer1 = gate1_h1[0:2]

    z1_layer1 = gate1_h1[2:4]

    reset_h0 = np.multiply(r1_layer1, h0)
   
    print("\n=== reset_h0 ===\n{}".format(reset_h0))

    new_input1_x1 = np.concatenate((x1, reset_h0), axis=-1)
    print("\n=== new_input_x1 ===\n{}".format(new_input1_x1))

    new_candidate1 = np.add(new_input1_x1.dot(candidate_w1), candidate_b1)
    print("\n=== new_candidate1 in layer1 ===\n{}".format(new_candidate1))
    
    layer1_h1 = np.add(np.multiply(z1_layer1, h0), np.multiply((1-z1_layer1), new_candidate1))

    print("\n=== h1 in layer 1 ===\n{}".format(layer1_h1))

    input1_x2 = np.concatenate((x2, layer1_h1), axis=-1)
    print("\n=== real input1 x2: concat(x2, laye1_h1) === \n{}".format(input1_x2))

    gate1_h2 = sigmoid(np.add(input1_x2.dot(gate_w1), gate_b1))
    print("\n=== gate1_h2 in layer1 (sigmoid(np.add(input1_x2.dot(gate_w1), gate_b1))) === \n{}".format(gate1_h2))

    r2_layer1 = gate1_h2[0:2]

    z2_layer1 = gate1_h2[2:4]

    reset_h1 = np.multiply(r2_layer1, layer1_h1)
   
    print("\n=== reset_h1 ===\n{}".format(reset_h1))

    new_input1_x2 = np.concatenate((x2, reset_h1), axis=-1)
    print("\n=== new_input_x2 ===\n{}".format(new_input1_x2))

    new_candidate2 = np.add(new_input1_x2.dot(candidate_w1), candidate_b1)
    print("\n=== new_candidate2 in layer1 ===\n{}".format(new_candidate2))
    
    layer1_h2 = np.add(np.multiply(z2_layer1, layer1_h1), np.multiply((1-z2_layer1), new_candidate2))

    print("\n=== h2 in layer 1 ===\n{}".format(layer1_h2))


    print("\n=== bi rnn output ===\n{}".format(sess.run(bi_output)))

    print("\n=== bi state output ===\n{}".format(sess.run(bi_state)))



    print(config)
    print(config1)
  #  print(sess.run(config))
    sess.close()



if __name__ == "__main__":
   #test_RNN_layer1()

   #test_RNN_layer2()

   #test_LSTM_layer1()

   #test_LSTM_layer2()
   
   test_GRU_layer1()
