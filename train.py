#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-19 上午10:22
# @Author  : J.W.
# @File    : train.py


from collections import Counter

import pandas as pd
import random
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np
import config
from logger import logger
from process import comm
from report import Report
import xgboost as xgb
import pdb

classfiers = {}

features_name = []

C = 1.0

# classfiers['lr'] = LogisticRegression(random_state=0,C=10, penalty='l2', solver='saga', multi_class='ovr') # 非常不好
#classfiers['xgb-1'] = xgb.XGBClassifier(max_depth=3, n_estimators=400, learning_rate=0.05)
# classfiers['xgb-2'] = xgb.XGBClassifier(max_depth=3, n_estimators=400, learning_rate=0.01)
# classfiers['xgb-3'] = xgb.XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05)
# classfiers['xgb-4'] = xgb.XGBClassifier(max_depth=6, n_estimators=400, learning_rate=0.05)
xgb_5 = xgb.XGBClassifier(
    max_depth=4,        #增加树的深度
    learning_rate=0.05,  #减小学习率
    n_estimators=500,    #增加分类器的个数
    n_jobs=2,
    random_state=22,
    seed=11
)

# xgb_5 = xgb.XGBClassifier(
#     max_depth=4,        #增加树的深度
#     learning_rate=0.05,  #减小学习率
#     n_estimators=50,    #增加分类器的个数
#     n_jobs=2,
#     random_state=22,
#     seed=11
# )

# classfiers['xgb-5']  = xgb_5
svm_1 = svm.SVC(C=1.0, kernel='rbf', gamma='auto', probability=True)
rfc = RandomForestClassifier(n_estimators=1000, min_samples_split=30, min_samples_leaf=5,
                             random_state=42,warm_start=True)  # 0.924
rfc2 = RandomForestClassifier(n_estimators=1000, min_samples_split=30, min_samples_leaf=5, criterion='entropy',
                             random_state=42,warm_start=True)  # 0.926
rfc3 = RandomForestClassifier(n_estimators=500, min_samples_split=30, min_samples_leaf=5, criterion='entropy',
                             random_state=42,warm_start=True)  # 0.928

# rfc_2 = RandomForestClassifier(n_estimators=1100, max_depth=13, min_samples_split=80, min_samples_leaf=10,
#                                oob_score=True, random_state=10, max_features='sqrt')
# rfc_3 = RandomForestClassifier(random_state=10, n_estimators=1600, max_depth=11, max_features=0.5,
#                                criterion='entropy', min_samples_split=140, min_samples_leaf=50)
# rfc_4 = RandomForestClassifier(n_estimators=1300, max_depth=13, max_features=0.5,  random_state=42,warm_start=True,
#                                criterion='entropy', min_samples_split=120, min_samples_leaf=10)
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)  # #0.9186
#
# forest = ExtraTreesClassifier(n_estimators=450, criterion = 'entropy',min_samples_split=10, min_samples_leaf=5,
#                               random_state=41)  # #0.9235


rng = np.random.RandomState(1)
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), #0.918
                         algorithm="SAMME",
                         n_estimators=100)
rng = np.random.RandomState(1)
# bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=6), #0.923
#                          algorithm="SAMME",
#                          n_estimators=100)
# bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=6), #0.923
#                          algorithm="SAMME",
#                          n_estimators=200) #0.926
# classfiers['forest'] = forest
#classfiers['forest'] = bdt   # 0.7679
# classfiers['random-forest'] = rfc
#
#classfiers['svm-svc-1'] = svm_1
# classfiers['rfc-2'] = rfc_2
# classfiers['rfc-3'] = rfc3

eclf = VotingClassifier(estimators=[('forest', forest),('rfc2', rfc2), ('rfc', rfc3), ('bdt', rfc), ('xgb', xgb_5)],
                         voting='soft', weights=[1, 1, 2, 1, 2])  # 0.938

eclf2 = VotingClassifier(estimators=[('forest', forest),('rfc2', rfc2), ('rfc', rfc3), ('bdt', bdt), ('xgb', xgb_5)],
                         voting='soft', weights=[1, 1, 1, 1, 1])  # 0.937
eclf3 = VotingClassifier(estimators=[('forest', forest),('rfc2', rfc2), ('rfc', rfc3), ('bdt', bdt), ('xgb', xgb_5)],
                         voting='soft', weights=[1, 2, 2, 1, 1])  # auc: 0.9367
eclf4 = VotingClassifier(estimators=[('forest', forest),('rfc2', rfc2), ('rfc3', rfc3),('rfc', rfc), ('bdt', bdt), ('xgb', xgb_5)],
                         voting='soft', weights=[1, 2, 2,1, 1, 2])  # 0.9376
classfiers['random-forest'] = eclf
from mlens.ensemble import SuperLearner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# --- Build ---
seed = 2017
from sklearn.metrics import f1_score
def f1(y, p): return f1_score(y, p, average='micro')

def auc(y, p): return metrics.roc_auc_score(y, p)

# Passing a scoring function will create cv scores during fitting
# the scorer should be a simple function accepting to vectors and returning a scalar
ensemble = SuperLearner(scorer=f1, random_state=seed)
# Build the first layer
#ensemble.add([RandomForestClassifier(random_state=seed), SVC(), rfc3])
ensemble.add([rfc ,rfc2, rfc3])
# Attach the final meta estimator
# ensemble.add_meta(LogisticRegression())
ensemble.add_meta(forest)


from mlens.ensemble import SuperLearner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def build_ensemble(incl_meta, propagate_features=None):
    """Return an ensemble."""
    if propagate_features:
        n = len(propagate_features)
        propagate_features_1 = propagate_features
        propagate_features_2 = [i for i in range(n)]
    else:
        propagate_features_1 = propagate_features_2 = None

    #estimators = [RandomForestClassifier(random_state=seed), SVC()]
    estimators = [rfc ,rfc2, rfc3]

    ensemble2 = SuperLearner()
    ensemble2.add(estimators, propagate_features=propagate_features_1)
    ensemble2.add(estimators, propagate_features=propagate_features_2)

    if incl_meta:
        ensemble.add_meta(LogisticRegression())
    return ensemble2
ensemble_base = build_ensemble(False)

# classfiers['svm-svc-2'] = svm.SVC(C=2.0, kernel='rbf', gamma='auto')
# classfiers['random-forest'] = eclf
# classfiers['ensemble'] = ensemble
# classfiers['ensemble'] = ensemble_base

def get_feature_data(data=None, model_columns=None):

    from process import feature
    features = feature.collect_features()

    #pdb.set_trace()
    # 填充　job marital, 效果略差
    # if config.train_flag:
    #     for i in ['job', 'marital']:
    #         if features[features[i] == 'unknown']['y'].count() < 500:
    #             # delete col containing unknown
    #             features = features[features[i] != 'unknown']
    # else:
    #     features.loc[features['job'] == 'unknown', 'job'] = "technician"  # management
    #     features.loc[features['marital'] == 'unknown', 'marital'] = "single"  # married

    # 确定Labels
    if config.train_flag:
        y = features['y'].values.astype('int')
        features.drop('y', axis=1, inplace=True)

    # 清除
    features = feature.clean_data(features)
    # 把　job ,marital 中 unknown的行删除掉

    if config.DEBUG:
        logger.info("\n{}".format(features.head(5)))
        logger.info(features.columns)
        logger.info(features.index)

    # 如果用到这些特征 需要编码
    all_need_coder_columns = config.all_need_coder_columns

    train_accept_feature = config.train_accept_feature

    if config.train_flag:
        accept_columns = []  # target 放第一列
    else:
        accept_columns = []

    accept_columns.extend(list(set(train_accept_feature)))


    #accept_columns
    need_coder_columns = list(set(accept_columns).intersection(all_need_coder_columns))
    org_features = features
    if not config.train_flag:
        ID = features['ID']

    # pdb.set_trace()
    # features = pd.get_dummies(data=features, dummy_na=True, columns=need_coder_columns)   #进行one-hot编码

    # 下面两行代码没用
    # for name in config.extention_features:
    #      features.append(org_features[name])

    features = features.drop(columns=['ID'])

    # pdb.set_trace()
    # if data is not None:
    #     features = features[:len(data)]
    #     logger.info('features shape: {}'.format(features.shape))

    # 去除编码后的无效列
    # try:
    #     for name in features.columns:
    #         if '_nan' in name:
    #             features.drop([name], axis=1, inplace=True)
    #             if config.DEBUG:
    #                 logger.info('drop column: {}'.format(name))
    # except Exception as e:
    #     logger.error(e, exc_info=True)

    # if model_columns is not None:
    #     model_columns.append("deal_type")
    #     diff = list(set(features.columns) - set(model_columns))
    #     if config.DEBUG:
    #         logger.info("diff columns: {}".format(diff))
    #
    #     features.drop(diff, axis=1, inplace=True)
    # logger.info('features shape: {}'.format(features.shape))

    features.sort_index(axis=1, ascending=True, inplace=True)
    if config.DEBUG:
        logger.info("\n {}".format(features.head(5)))
        logger.info("columns: {}".format(features.columns.tolist()))



    #pdb.set_trace()
    x = features.iloc[:].values.astype('int')
    #pdb.set_trace()
    if config.train_flag:
        if config.DEBUG:
            logger.info("y stat: {}".format(Counter(y)))
        return x, y, features.columns.tolist()
    else:
        return ID, x, features.columns.tolist()


def random_result(num=0):
    '''
    随机生成结果，生成baseline
    :param num:
    :return:
    '''
    result = []
    for i in range(num):
        result.append(random.randint(1, 2))
    return result


def cv(X, y, clf):
    import numpy as np
    model = None
    max_p = 0

    reporter = Report()
    for i in range(1, 10):  # 10
        logger.info("Folder {}".format(i))
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21 + i)
        # logger.info("Train x data: ({}, {}), Train y data: {}".format(len(x_train), len(x_train[0]), len(y_train)))
        # logger.info("Test x data: ({}, {}), Test y data: {}".format(len(x_test), len(x_test[0]), len(y_test)))

        clf.fit(x_train, y_train)

        ####  features importance
        # importances = clf.feature_importances_
        # # std = np.std([tree.feature_importances_ for tree in clf.estimators_],
        # #              axis=0)
        # indices = np.argsort(importances)[::-1]
        # sorted_importances = sorted(importances, reverse=True)
        # print("top 10 sum importance %f" %(sum(sorted_importances[0:10])))
        # print("top 20 sum importance %f" % (sum(sorted_importances[0:20])))
        # print("top 50 sum importance %f" % (sum(sorted_importances[0:50])))
        # print("top 100 sum importance %f" % (sum(sorted_importances[0:100])))
        # print("top 150 sum importance %f" % (sum(sorted_importances[0:150])))
        # # Print the feature ranking
        # print("Feature ranking:")
        #
        # for f in range(X.shape[1]):
        #     print("%d. %s feature %d (%f)" % (f + 1, features_name[f], indices[f], importances[indices[f]]))


        predict = clf.predict(x_test)

        # predict = random_result(num=len(y_test))

        # logger.info(y_test.mean())
        # logger.info(predict.mean())
        # logger.info("clf.oob_score_: {}".format(clf.oob_score_))
        #import pdb
        #pdb.set_trace()
        x_predprob = clf.predict_proba(x_train)[:,1]
        # x_predprob = clf.predict_proba(x_train)
        logger.info("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, x_predprob))
        y_predprob = clf.predict_proba(x_test)[:, 1]
        # y_predprob = clf.predict_proba(x_test)
        logger.info("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_predprob))
        auc = metrics.roc_auc_score(y_test, y_predprob)

        reporter.report_one_folder(y_test, predict)

        p = metrics.precision_score(y_test, predict, pos_label=1)
        if auc > max_p:
            max_p = auc
            logger.info("auc: {:.4f}".format(auc))

            model = clf

        # mtx = metrics.classification_report(y_test, clf.predict(x_test))
        # logger.info("\n{}".format(mtx))
    reporter.report_final_result()
    return model


def random_forest_param_search(X, y):
    '''
    调参过程
    https://blog.csdn.net/yingfengfeixiang/article/details/79369059
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    :param x:
    :param y:
    :return:
    '''
    from sklearn.model_selection import GridSearchCV
    # 查看基准的训练情况
    logger.info("base")
    rfo = RandomForestClassifier(oob_score=True, random_state=10)
    rfo.fit(X, y)
    logger.info("oob_score: {}".format(rfo.oob_score_))
    y_predprob = rfo.predict_proba(X)[:, 1]
    logger.info("AUC Score (Train): {}".format(metrics.roc_auc_score(y, y_predprob)))

    # 把调参的结果一步一步加入进来看对比效果
    logger.info("para:")
    rfo = RandomForestClassifier(oob_score=True, random_state=10, n_estimators=1600, max_depth=11, max_features=0.5,
                                 criterion='entropy', min_samples_split=140, min_samples_leaf=50)
    rfo.fit(X, y)
    logger.info("oob_score: {}".format(rfo.oob_score_))
    y_predprob = rfo.predict_proba(X)[:, 1]
    logger.info("AUC Score (Train): {}".format(metrics.roc_auc_score(y, y_predprob)))

    # 一步一步的搜索最佳参数，调后面参数的时候把前面参数带入
    param_test = {}
    # param_test = {'n_estimators': range(100, 3000, 500)}  # best para: {'n_estimators': 2100}
    param_test = {'max_depth': range(3, 20, 2)}  # 11
    # param_test = {'max_features': ['auto', 'sqrt', 'log2', .01, .5, .99]} # .5
    param_test = {'criterion': ['gini', 'entropy']}  # entropy
    # param_test = {'splitter': ['best', 'random']} # error
    param_test = {'min_samples_split': range(80, 150, 20), 'min_samples_leaf': range(10, 60, 10)}  # 140, 50
    param_test = {'n_estimators': range(100, 3000, 500),
                  'max_depth': range(3, 20, 2),
                  'max_features': ['auto', 'sqrt', 'log2', .01, .5, .99],
                  'criterion': ['gini', 'entropy'],
                  'min_samples_split': range(80, 150, 20),
                  'min_samples_leaf': range(10, 60, 10)}

    logger.info("param test: {}".format(param_test))
    gsearch1 = GridSearchCV(
        estimator=RandomForestClassifier(random_state=10),
        param_grid=param_test, scoring='roc_auc', cv=5)
    gsearch1.fit(X, y)
    logger.info("best_estimator_ {} ".format(gsearch1.best_estimator_))
    logger.info("best para: {}".format(gsearch1.best_params_))


# def predict(model, feature):
#     result = model.predict(feature)
#     result_pro = model.predict_proba(feature)
#     # logger.info("result: {}".format(result))
#     return result, result_pro


# 参数搜素
def param_search():
    x, y, columns = get_feature_data()
    random_forest_param_search(x, y)


def train_process():
    x, y, columns = get_feature_data()
    # random_forest_param_search(x, y)
    # cv for ever clf
    features_name = columns
    for index, (name, clf) in enumerate(classfiers.items()):
        print('{}: {}'.format(name, clf))
        model = cv(x, y, clf)
        comm.save_file((clf, columns), config.model_path)

def test_process():
    clf, _ = comm.load_file(config.model_path)
    ID, x, columns = get_feature_data()
    #pdb.set_trace()
    #
    #result = clf.predict(x)
    result = clf.predict_proba(x)[:, 1]
    print('result shape:', result.shape)
    for i in range(40):
        print(ID[i], result[i])
    #pdb.set_trace()
    with open(config.test_output_path, 'w') as f:
        f.write("ID,pred\n")
        for i in range(result.shape[0]):
            # if i == 148:
            #     print("ok")
            #     pdb.set_trace()
            #f.write(str(str(ID[i])) + "," + str(np.around(result[i], decimals=5))+ "\n")
            f.write(str(str(ID[i])) + "," + str(result[i]) + "\n")



if __name__ == "__main__":
    # parameter search
    #param_search()

    # train  and  test process
    if config.train_flag:
        train_process()
    else:
        test_process()
