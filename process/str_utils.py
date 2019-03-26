#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-1-2 下午3:09
# @Author  : J.W.
# @File    : str_utils.py
from logger import logger

def category_type():
    types = {}
    names = '跆拳道 武+武道+泰拳 音乐 艺术 琴+棋+画+古筝+钢琴+笛+乐器+打击乐 健身 美术+美院 舞蹈+舞+啦啦操 英语+外 主持+话筒+语音+口才 咨询 教育 字+书院+书法 培训+辅导+教辅 文化 儿+少年+亲子+幼童+早教+托管 管理 体育+足球+溜冰+运动+篮球 妆 乐器 机器人+智能 驾校'

    num = 1
    for item in names.split(' '):
        item = item.strip()
        if len(item) == 0:
            continue
        types[item] = num
        num += 1
    logger.info(types)
    return types


types = category_type()


def get_type_by_name(org_name):
    ''''
    根据机构名字对结构类型简单分类
    例如：跆拳道 舞蹈（舞） 英语 艺术 咨询 主持
    '''

    for name, category in types.items():
        for item in name.split('+'):
            if item in org_name:
                return category
    # print(org_name)
    return '0'
