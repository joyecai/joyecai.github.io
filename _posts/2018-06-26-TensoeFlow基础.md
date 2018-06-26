---
layout:     post
title:      Tensorflow基础
subtitle:   For practice
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

	# 定义常量 a
    a = tf.constant(1)
	
	# 定义变量 b（这个例子中把b作计数器了）
	b = tf.Variable(0, name='counter')
	print(b.name)
	
	# 加法（等式左边在tf中可以理解为步骤名）
	sum = tf.add(a,b)
	
	# 更新赋值
	update = tf.assign(b, sum)

	
### 初始化所有变量并且运行（激活）

	# 如果定义了变量，就一定要初始化它们
	init = tf.global_variables_initializer() 
	
	# 运行（把sess理解为指针，sess.run(xx步骤)代表运行该步骤）
	with tf.Session() as sess:
		sess.run(init)             #运行变量的初始化
		for _ in range(3):
			sess.run(update)       #运行update步骤
			print(sess.run(sum))   #运行sum步骤
	
	
### 矩阵
	
	#定义矩阵
	matrix1 = tf.constant([ [3., 3.] ])   #1x2的矩阵
	matrix2 = tf.constant([ [2.],         #2x1的矩阵
	                        [2.]  ])
	
	#矩阵乘法
	product = tf.matmul(matrix1, matrix2)
	
### 运行（即启动图）

	#方法1
    sess = tf.Session()
	result = sess.run(product)
	print result
	sess.close()
	
	#方法2
	with tf.Session() as sess：
		result = sess.run([product])
		print result

### 指派GPU运行
    
	#通常情况下，你不需要显示指使用CPU或者GPU。TensorFlow能自动检测，如果检测到GPU，TensorFlow会使用第一个GPU来执行操作。
	#如果机器上有多个GPU，除第一个GPU外的其他GPU是不参与计算的，为了使用这些GPU，你必须将op明确指派给他们执行。
	#with…Device语句用来指派特定的CPU或GPU执行操作：
	
	with tf.Session() as sess:
		with tf.device("/gpu:1"):
			matrix1 = tf.constant([[3., 3.]])
			matrix2 = tf.constant([[2.],[2.]])
			product = tf.matmul(matrix1, matrix2)
			...