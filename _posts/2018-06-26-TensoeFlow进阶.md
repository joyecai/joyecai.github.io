---
layout:     post
title:      Tensorflow进阶
subtitle:   For Deep Learning
date:       2018-06-26
author:     Jiayue Cai
header-img: img/post-bg-debug.png
catalog: true
tags:
    - Tensorflow

---


>持续更新中... 

### 引入头文件 

	import tensorflow as tf

### 常量&变量 

	#之前的例子中展示了在计算图中引入tensor，以常量和变量的形式存储。
	#TensorFlow还提供了feed机制，该机制可以临时替换图中的tensor。
	
	input1 = tf.placeholder(tf.float32)
	input2 = tf.placeholder(tf.float32)
	output = tf.mul(input1, input2)

	with tf.Session() as sess:
	print sess.run([output], feed_dict={input1:[7.], input2:[2.]})

	# 输出:
	# [array([ 14.], dtype=float32)]
