# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 22:50:13 2018

@author: chint
"""

import tensorflow as tf

def activation(x, use_activation=None):
    assert use_activation in ['sigmoid', 'relu', 'tanh', 'leaky_relu', None]
    
    if use_activation == 'sigmoid':
        out = tf.sigmoid(x)
    
    elif use_activation == 'relu':
        out = tf.nn.relu(x)
    
    elif use_activation == 'tanh':
        out = tf.tanh(x)
    
    elif use_activation == 'leaky_relu':
        out = tf.keras.layers.LeakyReLU(0.2)(x)
        
    else:
        out = x
    
    return out

def norm(x, use_norm, is_train, reuse):
    assert use_norm in ['instance', 'batch', None]
    
    if use_norm == 'instance':
        with tf.variable_scope('norm_instance', reuse=reuse):
            eps = 1e-5
            mean, var = tf.nn.moments(x, [1,2], keep_dims=True)
            out = (x - mean)/ (tf.sqrt(var) + eps)
    
    elif use_norm == 'batch':
        with tf.variable_scope('norm_instance', reuse=reuse):
            batchnorm = tf.keras.layers.BatchNormalization()
            out = batchnorm(x, trainable=is_train)
    
    else:
        out = x
    
    return out

def conv2D(x_in, filters, size, stride, pad='SAME', dtype=tf.float32, bias=False, reuse=False):
    strides_shape = [1, stride, stride, 1]
    filter_shape = [size, size, x_in.get_shape()[3], filters]
    w_init = tf.random_normal_initializer(0., 0.02)
    b_init = tf.constant_initializer(0.0)
    
    W = tf.get_variable('W', filter_shape, dtype, w_init)
    if pad == 'REFLECT':
        p = (size - 1) // 2
        x = tf.pad(x_in, [[0,0], [p,p], [p,p], [0,0]], 'REFLECT')
        out = tf.nn.conv2d(x, W, strides_shape, padding='VALID')
    else:
        assert pad in ['SAME', 'VALID']
        out = tf.nn.conv2d(x_in, W, strides_shape, padding=pad)
    
    if bias:
        b = tf.get_variable('b', [1,1,1,filters], initializer=b_init)
        out = out + b
    
    return out
    
def conv_block(x_in, name, filters, size, stride, is_train, use_norm, use_activation, reuse, pad='SAME', bias=False):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2D(x_in, filters, size, stride, pad, bias=bias, reuse=reuse)
        out = norm(out, use_norm, is_train, reuse)
        out = activation(out, use_activation)
        
        return out

def conv2DT(x_in, filters, size, stride, pad='SAME', dtype=tf.float32, reuse=False):
    assert pad == 'SAME'
    num, height, width, channels = x_in.get_shape().as_list()
    strides_shape = [1, stride, stride, 1]
    filter_shape = [size, size, filters, channels]
    out_shape = [num, height*stride, width*stride, filters]
    w_init = tf.random_normal_initializer(0., 0.02)
    
    W = tf.get_variable('W', filter_shape, dtype, w_init)
    out = tf.nn.conv2d_transpose(x_in, W, out_shape, strides_shape, pad)
    return out

def convT_block(x_in, name, filters, size, stride, is_train, use_norm, use_activation, reuse):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2DT(x_in, filters, size, stride, reuse=reuse)
        out = norm(out, use_norm, is_train, reuse)
        out = activation(out, use_activation)
        
        return out

def residual_block(x_in, name, filters, is_train, use_norm, reuse, pad='REFLECT'):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('res1', reuse=reuse):
            out = conv2D(x_in, filters, 3, 1, pad, reuse=reuse)
            out = norm(out, use_norm, is_train, reuse)
            out = tf.nn.relu(out)
        
        with tf.variable_scope('res2', reuse=reuse):
            out = conv2D(out, filters, 3, 1, pad, reuse=reuse)
            out = norm(out, use_norm, is_train, reuse)
        
        return tf.nn.relu(out+x_in)
