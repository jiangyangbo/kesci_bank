#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-11 上午11:20
# @Author  : J.W.
# @File    : counter.py

# -*- coding:utf-8 -*-

def create_counter():
    '''
    https://blog.csdn.net/sinat_41701878/article/details/79301449
    :return: 每次调用自增1
    '''

    def increase():  # 定义一个含有自然数算法的生成器,使用next来完成不断调用的递增
        n = 0
        while True:
            n = n + 1
            yield n

    it = increase()  # 一定要将生成器转给一个(生成器)对象,才可以完成

    def counter():  # 再定义一内函数
        return next(it)  # 调用生成器的值,每次调用均自增

    return counter


c = create_counter()

for i in range(100):
    num = c()
    print(num)
