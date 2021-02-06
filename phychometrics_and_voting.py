# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 21:43:35 2021

@author: do1113
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, accuracy_score
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier

df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test_x.csv')
# df.isnull().values.any() → NaN값 없음

# 칼럼, 고유값 등등 자판기
# df.columns[:]
# df.shape
# df['QaA'].unique()
# df['familysize'].describe()

# 제출 데이터에 적용하기 위해 식으로 만들기.
def preprocessing(df):
    

# 문항별 평균 응답시간 계산
    df.set_index('index', inplace=True)
    df['time_to_answer'] = df[df.columns[1:40:2]].mean(axis=1)

# 가족수에 터무니없는 숫자들은 임의로 최빈값으로 통일, 12명 이상도 응답오류로 간주하고 12로 통일.
    for i in range(len(df['familysize'])):
        if df['familysize'][i]>=100:
            df['familysize'][i]=df['familysize'].mode()[0]
        elif df['familysize'][i]>12:
            df['familysize'][i]=12

# Big 5 Model(tp test) 점수 합산하여 정리. 출처: https://dacon.io/forum/401589
    df['tp_Ext'] = df['tp01']-df['tp06']
    df['tp_A'] = -df['tp02']+df['tp07']
    df['tp_C'] = df['tp03']-df['tp08']
    df['tp_Emo'] = -df['tp04']+df['tp09']
    df['tp_O'] = df['tp05']-df['tp10']
    df.drop(df.columns[49:59], axis=1, inplace = True)
    
# wr도 실존 단어인지에 따라서 점수 더하고.
    df['wr_right'] = df['wr_01']+df['wr_02']+df['wr_03']+df['wr_04']
    +df['wr_05']+df['wr_06']+df['wr_07']+df['wr_08']
    +df['wr_09']+df['wr_10']+df['wr_11']+df['wr_12']+df['wr_13']
    df['wr_false'] = df['wf_01']+df['wf_02']+df['wf_03']
    df.drop(df.columns[51:67], axis=1, inplace = True)

# 각종 str data는 dummy data로 수정
    temp = df
    for column in df.columns[40:43].append(df.columns[44:50]):
        temp = pd.concat([temp, pd.get_dummies(df[column], prefix=column, drop_first=True)], axis = 1)
        temp.drop(column, axis = 1, inplace = True)
    df = temp

# 마키아벨리즘 Score 계산법 적용. 출처: https://dacon.io/competitions/official/235647/talkboard/401557?page=1&dtype=recent&ptype=pub
    Mach = {'T+':['QcA','QoA','QsA',], 'T-':['QfA','QrA',],
        'M+':[], 'M-':['QkA',],
        'V+':['QbA', 'QhA','QjA','QmA',], 'V-':['QeA','QqA',]}

    temp1 = df[df.columns[0:40:2]]
    for i in range(0, 6):
        temp1[list(Mach.keys())[i]] = 0

    for i in range(0, 6):
        for j in Mach[list(Mach.keys())[i]]:
            temp1[list(Mach.keys())[i]] += temp1[j]


    corr = temp1[['QaA','QdA','QgA','QiA','QlA','QnA','QpA','QtA',
                  'T+','T-','M+','M-','V+','V-']].corr(method = 'pearson')
    heatmap = sns.heatmap(corr, cbar = True, annot = True, fmt = '.2f',
                          annot_kws={'size' : 5}, square = False, cmap = 'Blues')

# 상관계수를 보면서 퍼즐 끼워맞추기. 결과 반영 후 T/M/V값 df에 추가
    Mach['T+'] = Mach['T+']+['QlA']
    Mach['T-'] = Mach['T-']+['QgA', 'QiA', 'QnA']
    Mach['M+'] = Mach['M+']+['QtA']
    Mach['V+'] = Mach['V+']+['QpA']
    Mach['V-'] = Mach['V-']+['QaA', 'QdA']

    for i in range(0, 6):
        temp1[list(Mach.keys())[i]] = 0

    for i in range(0, 6):
        for j in Mach[list(Mach.keys())[i]]:
            temp1[list(Mach.keys())[i]] += temp1[j]
            
    df['Mach_T'] = temp1['T+']-temp1['T-']            
    df['Mach_M'] = temp1['M+']-temp1['M-']            
    df['Mach_V'] = temp1['V+']-temp1['V-']

# 기존 설문 데이터 삭제, 열 순서 정렬
    df.drop(df.columns[0:40], axis=1, inplace = True)
    temp2 = df['familysize']
    temp3 = df['time_to_answer']
    df.drop(['familysize', 'time_to_answer'], axis=1, inplace = True)
    df = pd.concat([df, temp2, temp3], axis = 1)

# X, y값 구분
    df_X = df.drop(['voted'], axis=1)
    df_y = df['voted']
      
# 상관관계 확인, 높은 상관관계인 값들은 통합
    temp = df_X.corr(method='pearson')
    for column1 in temp.columns:
        for column2 in temp.columns:
            # if abs(temp[column1][column2]) > 0.6 and temp[column1][column2]!=1:
            #     print(column1, column2)

# Mach 3요소의 상관관계가 0.6 이상, 나머지는 더미변수이므로 살림
    df_X['Mach'] = df['Mach_T']+df['Mach_M']+df['Mach_V']
    df_X.drop(['Mach_T', 'Mach_M', 'Mach_V'], axis=1, inplace=True)
    return df_X, df_y

# 예측 전 전처리
# df_X, df_y = preprocessing(df1)
df2.insert(61, 'voted', 0, True)
df_y_fin=[]
df_X_fin, df_y_fin = preprocessing(df2)

df_X_fin_trans = StandardScaler().fit_transform(df_X_fin)



# 1번 결과물
df_y_fin = pd.DataFrame(model.predict(df_X_fin_trans), columns=['voted'])
df_y_fin.to_csv('submission.csv')

# 2번 결과물
df_y_fin2 = pd.DataFrame(rnd_clf.predict(df_X_fin_trans), columns=['voted'])
df_y_fin2.to_csv('submission2.csv')


# 아래와 같은 방법으로 best estimator 확인. random forest tree로 학습
# df_X_trans = StandardScaler().fit_transform(df_X)
# X_train, X_test, y_train, y_test = train_test_split(df_X_trans, df_y, test_size = .2, random_state = 23)

# params = {'n_estimators':[1000],'max_features':[22, 23, 24, 25],
#           'max_depth':[5,6,7,8,9,10], 'random_state':[23]}

# grid = GridSearchCV(RandomForestClassifier(), param_grid = params, cv=5, n_jobs=-1)
# grid.fit(X_train, y_train)

# print('베스트 하이퍼 파라미터: {0}'.format(grid.best_params_))
# print('베스트 하이퍼 파라미터 일 때 R^2 점수: {0:.2f}'.format(grid.best_score_))
# model = grid.best_estimator_
# r2_score = model.score(X_test, y_test)
# print('테스트세트에서의 R^2 점수: {0:.2f}'.format(r2_score))

# pred_y_train = model.predict(X_train)
# pred_y_test = model.predict(X_test)
# print('Accuracy score in testset: {0:2f}'.format(accuracy_score(pred_y_train, y_train)))
# print('Accuracy score in testset: {0:2f}'.format(accuracy_score(pred_y_test, y_test)))

 
rnd_clf = RandomForestClassifier(n_estimators = 1000, max_depth=7, max_features = 22, random_state=23)
rnd_clf.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(rnd_clf.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rnd_clf.score(X_test, y_test)))