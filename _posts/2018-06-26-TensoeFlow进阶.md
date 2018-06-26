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

	
### 命令行参数（Flag）

详见[博客1](https://blog.csdn.net/u012436149/article/details/52870069)和[博客2](https://blog.csdn.net/lyc_yongcai/article/details/73456960)
	
###### 新建test.py文件：

	#定义参数
	flags = tf.flags
	flags.DEFINE_string("para_name_1","default_val", "description")
	flags.DEFINE_bool("para_name_2","default_val", "description")
	
	#session主体
	def main(_): 
		...
	
###### 命令行调用

	python test.py --para_name_1=name --para_name_2=name2
	