#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : logger.py
# @Time    : 18-3-14
# @Author  : J.W.
import logging
import os
from logging.handlers import TimedRotatingFileHandler

'''
logger 工具类
'''


def contains_this_handler(handlers, handler):
    '''
    判断 日志中是否已经存在该handler
    :param handlers:
    :param handler:
    :return:
    '''
    for hd in handlers:
        if isinstance(hd, handler):
            return True
    return False


import config

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# LOG_DIR = os.path.join(BASE_DIR, "logs")
#
# if not os.path.exists(LOG_DIR):
#     os.makedirs(LOG_DIR)  # 创建路径
#
# log_file = os.path.join(LOG_DIR, "app.log")
log_file = config.log_file
if config.server:
    # use django setting
    logger = logging.getLogger('app')
else:
    log_file = log_file.replace('app', 'app-local')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-6s: %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        filename=log_file,
                        filemode='a')  # or 'w', default 'a'

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(name)-6s: %(levelname)-6s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logger = logging.getLogger("app")

    # app_log_file = os.path.join(LOG_DIR, "app.log")
    # rubine = logging.FileHandler(filename=app_log_file)
    # rubine.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s %(name)-6s: %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    # rubine.setFormatter(formatter)
    # logger.addHandler(rubine)

    # 添加TimedRotatingFileHandler
    # 根据 when 定义切割方式  midnight M
    filehandler = logging.handlers.TimedRotatingFileHandler(log_file, when='midnight', interval=1,
                                                            backupCount=0)
    # 设置后缀名称，跟strftime的格式一样
    filehandler.suffix = "%Y-%m-%d.log"
    logger.addHandler(filehandler)
