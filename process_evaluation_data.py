# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# +
for i in ['06','07','08','09','10','11','12']:
    data_temp=pd.read_csv('2021_'+i+'.csv',encoding='cp949')
    data_temp.rename(columns={'SRCHWRD_NM':'tour','SCCNT_VALUE':i+'SCCNT','SCCNT':i+'SCCNT'},inplace=True)
    data_temp=data_temp.iloc[:,[4,5]]
    globals()['data'+i] = data_temp

data01=pd.read_csv('2022_'+'01'+'.csv',encoding='utf8')
data01.rename(columns={'SRCHWRD_NM':'tour','SCCNT_VALUE':'01SCCNT','SCCNT':'01SCCNT'},inplace=True)
data01=data01.iloc[:,[4,5]]

for j in ['02','03','04','05']:
    data_temp=pd.read_csv('2022_'+j+'.csv',encoding='cp949')
    data_temp.rename(columns={'SRCHWRD_NM':'tour','SCCNT_VALUE':j+'SCCNT','SCCNT':j+'SCCNT'},inplace=True)
    data_temp=data_temp.iloc[:,[4,5]]
    globals()['data'+j] = data_temp
# -

merge_data=data06
for i in ['07','08','09','10','11','12','01','02','03','04','05']:
    merge_data=pd.merge(merge_data,globals()['data'+i],how='outer',on='tour')

merge_data=merge_data.drop_duplicates(['tour'])
merge_data=merge_data.reset_index(drop=True)
merge_data=merge_data.fillna(0)
merge_data

# +
remove_list=[]
for i in range(7326):
    count=0
    for j in range(1,13):
        if merge_data.iloc[i][j]>0:
            count=count+1
    if count<3:
        remove_list.append(i)

merge_data = merge_data.drop(remove_list, axis=0)
merge_data=merge_data.reset_index(drop=True)

# +
merge_data['spring']=0
merge_data['summer']=0
merge_data['autumn']=0
merge_data['winter']=0

for i in range(6658):
    
    for j in range(1,13):
        if j==1:
            avg_value=merge_data.iloc[i,j]
        else: 
            avg_value=avg_value+merge_data.iloc[i,j]
    avg_value=avg_value/12

    season_list=[]
    for k in range(1,13):
        if merge_data.iloc[i,k]>avg_value:
            season_list.append(k)
    
    for l in season_list:
        if l>0 and l<4:
            merge_data.iloc[i,14]=1
        if l>3 and l<7:
            merge_data.iloc[i,15]=1
        if l>6 and l<10:
            merge_data.iloc[i,16]=1
        if l>9 and l<13:
            merge_data.iloc[i,13]=1

merge_data.head(5)
# -

evaluation_data=merge_data.iloc[:,[0,13,14,15,16]]
evaluation_data.head(5)

evaluation_data.to_csv('C:/Users/strai/season_evaluation_2.csv',sep=',',na_rep='NaN',index=False)

evaluation_data=pd.read_csv('season_evaluation_2.csv')

evaluation_data

x=['06','07','08','09','10','11','12','01','02','03','04','05']
y1=[]
y2=[]
for i in range(12):
    y1.append(merge_data.iloc[0,i+1])
for j in range(12):
    y2.append(sum(y1)/len(y1))
plt.plot(x,y1,'o')
plt.plot(x,y2)
plt.show()


