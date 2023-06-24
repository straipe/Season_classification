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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from gensim.models.word2vec import Word2Vec
import json
import os
import sys

from eunjeon import Mecab

mecab=Mecab()

import requests; from urllib.parse import urlparse


def kakao_api_blog(keyword,page):
    url="https://dapi.kakao.com/v2/search/blog?&query="+keyword+"&size=50"+"&page="+str(page)
    result=requests.get(urlparse(url).geturl(),
                       headers={"Authorization":"KakaoAK secretkey입력"})
    json_obj=result.json()
    return json_obj


list1=[]; keyword="봄 여행"; page=1
while page <= 50:
    json_obj=kakao_api_blog(keyword,page)
    for document in json_obj['documents']:
        val=[document['title'].replace("<b>","").replace("</b>",""),
            document['contents'].replace("<b>","").replace("</b>",""),
            document['blogname'], document['datetime'],document['url']]
        list1.append(val)
    if json_obj['meta']['is_end'] is True: break
    page +=1
df1=pd.DataFrame(list1,columns=['title','contents','name','datetime','url'])
df1


def kakao_api_cafe(keyword,page):
    url="https://dapi.kakao.com/v2/search/cafe?&query="+keyword+"&size=50"+"&page="+str(page)
    result=requests.get(urlparse(url).geturl(),
                       headers={"Authorization":"KakaoAK secretkey입력"})
    json_obj=result.json()
    return json_obj


list2=[]; keyword="봄 여행"; page=1
while page <= 50:
    json_obj=kakao_api_cafe(keyword,page)
    for document in json_obj['documents']:
        val=[document['title'].replace("<b>","").replace("</b>",""),
            document['contents'].replace("<b>","").replace("</b>",""),
            document['cafename'], document['datetime'],document['url']]
        list2.append(val)
    if json_obj['meta']['is_end'] is True: break
    page +=1
df2=pd.DataFrame(list2,columns=['title','contents','name','datetime','url'])
df2.shape


def kakao_api_web(keyword,page):
    url="https://dapi.kakao.com/v2/search/web?&query="+keyword+"&size=50"+"&page="+str(page)
    result=requests.get(urlparse(url).geturl(),
                       headers={"Authorization":"KakaoAK secretkey입력"})
    json_obj=result.json()
    return json_obj


list3=[]; keyword="봄 여행"; page=1
while page <= 50:
    json_obj=kakao_api_web(keyword,page)
    for document in json_obj['documents']:
        val=[document['title'].replace("<b>","").replace("</b>",""),
            document['contents'].replace("<b>","").replace("</b>",""),
            document['datetime'],document['url']]
        list3.append(val)
    if json_obj['meta']['is_end'] is True: break
    page +=1
df3=pd.DataFrame(list3,columns=['title','contents','datetime','url'])
df3.shape

df=pd.concat([df1['title'],df2['title'],df3['title']],axis=0)
df.shape

# +
#df.to_csv('C:/Users/strai/kakao_spring.txt',index=False,header=False)
#df.to_csv('C:/Users/strai/kakao_summer.txt',index=False,header=False)
#df.to_csv('C:/Users/strai/kakao_autumn.txt',index=False,header=False)
#df.to_csv('C:/Users/strai/kakao_winter.txt',index=False,header=False)
# -

four_season={"spring":[],"summer":[],"autumn":[],"winter":[]}
for season_word in ["spring","summer","autumn","winter"]:
    f=open("C:/Users/strai/kakao_"+season_word+".txt",'r',encoding='UTF-8')
    lines=f.readlines()
    f.close()
    for j in range(len(lines)):
        four_season[season_word].append(mecab.nouns(lines[j]))
    four_season[season_word]=[[y for y in x if not len(y)==1] for x in four_season[season_word]]
    four_season[season_word]=[[y for y in x if not y.isdigit()] for x in four_season[season_word]]

# +
model=Word2Vec(four_season["spring"],sg=1,window=3,min_count=1)
model.init_sims(replace=True)

model2=Word2Vec(four_season["summer"],sg=1,window=3,min_count=1)
model2.init_sims(replace=True)

model3=Word2Vec(four_season["autumn"],sg=1,window=3,min_count=1)
model3.init_sims(replace=True)

model4=Word2Vec(four_season["winter"],sg=1,window=3,min_count=1)
model4.init_sims(replace=True)
# -

df_tmp=pd.DataFrame(model2.wv.most_similar("여행",topn=2000),columns=['단어','유사도'])
df_tmp[:10]

n=1000
df_tmp[n:n+10]

try:
    b=model5.wv.similarity('여행','장미')
except:
    print("키에러")
print(b)

model3.wv.similarity('여행','단풍')


def classification(name,standard1,standard2):
    word_list=mecab.nouns(name)
    classify_vector=np.zeros(4)
    season_dict={"spring":0,"summer":1,"autumn":2,"winter":3}
    season_result=[]
    zero_count=0
    
    for word in word_list:
        if len(word)<2 or word=='축제':
            continue
        try:
            spring_similarity=model.wv.similarity("여행",word)
            if spring_similarity>0.998110:
                classify_vector[0]+=2
            elif spring_similarity>(0.97335):
                classify_vector[0]+=1
            else:
                classify_vector[0]+=0
        except:
            classify_vector[0]+=0
        try:
            summer_similarity=model2.wv.similarity('여행',word)
            if summer_similarity>0.997890:
                classify_vector[1]+=2
            elif summer_similarity>(0.979896):
                classify_vector[1]+=1
            else:
                classify_vector[1]+=0
        except:
            classify_vector[1]+=0
        try:
            autumn_similarity=model3.wv.similarity('여행',word)
            if autumn_similarity>0.997411:
                classify_vector[2]+=2
            elif autumn_similarity>(0.982249):
                classify_vector[2]+=1
            else:
                classify_vector[2]+=0
        except:
            classify_vector[2]+=0
        try:
            winter_similarity=model4.wv.similarity('여행',word)
            if winter_similarity>0.997154:
                classify_vector[3]+=2
            elif winter_similarity>(0.985182):
                classify_vector[3]+=1
            else:
                classify_vector[3]+=0
        except:
            classify_vector[3]+=0
    
    max_value=classify_vector.max()
    
    for i in range(4):
        if classify_vector[i]==max_value:
            season_result.append(i)
    
    return season_result


classification('축제',0.9979,0.9719)

#evaluation_data=pd.read_csv('season_evaluation.csv')
evaluation_data2=pd.read_csv('season_evaluation_2.csv')

model3.wv.similarity('여행','별빛') # 0 0 0 0

df2[['title']].to_csv('C:/Users/strai/kakao_web_spring_tour2.txt',index=False,header=False)

mecab.nouns('월악산')


def loss2(evaluation_data2,standard1,standard2):
    loss=0
    accuracy_count=0
    for i,name in enumerate(list(evaluation_data2.iloc[:,0])):
        season_list=[]
        predict_list=classification(name,standard1,standard2)
        
        #if len(predict_list)==4:
         #   continue
        
        for j in [1,2,3,4]:
            if evaluation_data2.iloc[i,j]==1:
                season_list.append(j-1)
        
        for k in season_list:
            accuracy_count=accuracy_count+1
            if k not in predict_list:
                loss = loss + 1

    return [loss,accuracy_count]


# +
predict_result=loss2(evaluation_data2[4000:],0.998,0.98)

predict_accuracy=1-predict_result[0]/predict_result[1]

print(predict_accuracy)
print(predict_result[0]) #3308 3308    continue O , X  (9979,9719)-> 3088 3155
print(predict_result[1]) #11920 17003 #1226 
# -

accuracy_list=[]; max_index={'standard1':0,'standard2':0}
max_accuracy=0
for i in np.arange(0.9975,0.9985,0.0001):
    for j in np.arange(0.97,0.99,0.02):
        now_result=loss2(evaluation_data2[:4000],i,j)
        accuracy=1-now_result[0]/now_result[1]
        accuracy_list.append(accuracy)
        if max(accuracy_list)>max_accuracy:
            max_accuracy=max(accuracy_list)
            max_index['standard1']=i;max_index['standard2']=j
            print(max_index)
print(max_index)

result2=loss2(evaluation_data2,0.998,0.971)
print(result2[0]) #2203
print(result2[1]) #4660 => continue X일때 대폭 감소

import urllib.request
client_id = "id입력"
client_secret = "secretkey입력"
encText = urllib.parse.quote("겨울 여행")
temp_list=[]
for i in [1,101,201,301,401,501,601,701,801,901]:
    url = "https://openapi.naver.com/v1/search/webkr?query=" + encText+"&display=100"+"&start="+str(i) # JSON 결과
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        result=json.loads(response_body.decode('utf-8'))
        for document in result['items']:
            val=[document['title'].replace("<b>","").replace("</b>","")]
            temp_list.append(val)
    else:
        print("Error Code:" + rescode)
df4=pd.DataFrame(temp_list,columns=['title'])

df4.to_csv('C:/Users/strai/naver_web_winter.txt',index=False,header=False)

df4


def fail_count(evaluation_data2):
    fail=0
    real_fail=0
    for name in list(evaluation_data2.iloc[:,0]):
        count=0
        for i in mecab.nouns(name):
            count=count+1
            try:
                model.wv.similarity('여행',i)
            except:
                try:
                    model2.wv.similarity('여행',i)
                except:
                    try:
                        model3.wv.similarity('여행',i)
                    except:
                        try:
                            model4.wv.similarity('여행',i)
                        except:
                            fail=fail+1
        if fail==count:
            real_fail=real_fail+1
    return real_fail


fail_count(evaluation_data2)

evaluation_data2[:4000]

mecab.nouns('롯데월드썸머페스티벌')


