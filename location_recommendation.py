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
import numpy as np
from haversine import haversine
from eunjeon import Mecab
from gensim.models.word2vec import Word2Vec
from random import shuffle
data=pd.read_csv("Tour.csv")

# +
data=data.drop(data.columns[4],axis=1)

Current_season=input('현재 월을 입력하세요: ')
data2=pd.read_csv("2021_"+Current_season+".csv",encoding='cp949') # csv파일이 UTF-8형식이 아니라서 encoding 인자 정의

now_season=3 #겨울
if int(Current_season)>2 and int(Current_season)<6:
    now_season=0 #봄
elif int(Current_season)>5 and int(Current_season)<9:
    now_season=1 #여름
elif int(Current_season)>8 and int(Current_season)<12:
    now_season=2 #가을

data2=data2.drop(data2.columns[[2,6]],axis=1)
data2=data2.sort_values("SCCNT",ascending=False) # 검색량 기준 내림차순 정렬


# +
def string_extraction(s,t): # s 문자열에서 t 문자열을 찾아 삭제 
    n=s.find(t)
    m=len(t)
    i=s[n:m]
    s=s.replace(i,'')
    return s

def text_preprocessor(data,top_key): #축제에 '제','축제'.. 제거 
    name=str(data['관광지명'])
    
    if name.find('축제') != -1:
        name=name.rstrip('축제')
    
    if name.find('제') != -1: 
        name=name.rstrip('제')
        
    if name.find('페스티벌') != -1:
        name=name.rstrip('페스티벌')
    
    # 롯데월드, 에버랜드, .. 접두어 제거
    for k in top_key:
        if name.find(k) != -1:
            name=string_extraction(name,k)
                
    return name

def minmaxscaler(s):
    return (s-s.min())/(s.max()-s.min())


# +
mecab=Mecab()

four_season={"spring":[],"summer":[],"autumn":[],"winter":[]}

for season_word in ["spring","summer","autumn","winter"]:
    f=open("C:/Users/strai/kakao_"+season_word+".txt",'r',encoding='UTF-8')
    lines=f.readlines()
    f.close()
    for j in range(len(lines)):
        four_season[season_word].append(mecab.nouns(lines[j]))
    four_season[season_word]=[[y for y in x if not len(y)==1] for x in four_season[season_word]]
    four_season[season_word]=[[y for y in x if not y.isdigit()] for x in four_season[season_word]]
    
model=Word2Vec(four_season["spring"],sg=1,window=5,min_count=1)
model.init_sims(replace=True)

model2=Word2Vec(four_season["summer"],sg=1,window=5,min_count=1)
model2.init_sims(replace=True)

model3=Word2Vec(four_season["autumn"],sg=1,window=5,min_count=1)
model3.init_sims(replace=True)

model4=Word2Vec(four_season["winter"],sg=1,window=5,min_count=1)
model4.init_sims(replace=True)


# -

def classification(name):
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
            elif spring_similarity>0.97335:
                classify_vector[0]+=1
            else:
                classify_vector[0]+=0
        except:
            classify_vector[0]+=0
        try:
            summer_similarity=model2.wv.similarity('여행',word)
            if summer_similarity>0.997890:
                classify_vector[1]+=2
            elif summer_similarity>0.979896:
                classify_vector[1]+=1
            else:
                classify_vector[1]+=0
        except:
            classify_vector[1]+=0
        try:
            autumn_similarity=model3.wv.similarity('여행',word)
            if autumn_similarity>0.997411:
                classify_vector[2]+=2
            elif autumn_similarity>0.982249:
                classify_vector[2]+=1
            else:
                classify_vector[2]+=0
        except:
            classify_vector[2]+=0
        try:
            winter_similarity=model4.wv.similarity('여행',word)
            if winter_similarity>0.997154:
                classify_vector[3]+=2
            elif winter_similarity>0.985182:
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


#recommendation(data,data2,(37.5838657,127.0587771))
def season_classification(result_list):
    for m in result_list[:]:
        data_record=data[data['관광지명']==m].squeeze()
        season=classification(text_preprocessor(data_record,['롯데월드','에버랜드','경복궁','서울랜드']))
        if now_season not in season:
            result_list.remove(m)
    return result_list


def recommendation(data,data2,Current_location):
    #data2=data2.drop(data2.columns[[2,6]],axis=1)
    #data2=data2.sort_values("SCCNT",ascending=False)
    recommend_list=list()
    result=list()
    tmp_dictionary={'관광지명':[],'거리':[],'검색량':[]}
    popularity_list=list(data2.loc[:,'SRCHWRD_NM'])
    #Current_location=(37.5838657,127.0587771) #서울시립대학교 위도 경도
    
    for n in range(len(data.index)):
        data_location=data.iloc[n,[-1,-2]]
        if haversine(Current_location,data_location,unit='km')< 10: #현 위치에서 10km 이내 추천 관광지만
            recommend_list.append(data.iloc[n,0])
            
    for j in popularity_list:
        for i in recommend_list:
            if j in i:
                result.append(i)

    result_list=season_classification(list(dict.fromkeys(result)))
    
    # 거리 검색량 순위 알고리즘
    for m in result_list:
        tmp_dictionary['관광지명'].append(m)
        
        tmp_record=data[data['관광지명']==m]
        tmp_location=tmp_record.iloc[0,[-1,-2]]
        tmp_dictionary['거리'].append(haversine(Current_location,tmp_location,unit='km'))
        for j in popularity_list:
            if j in m:
                tmp2_record=data2[data2['SRCHWRD_NM']==j]
                tmp_dictionary['검색량'].append(tmp2_record.iloc[0,-1])
                break
    
    rank_dist_sccnt=pd.DataFrame(tmp_dictionary)
    rank_dist_sccnt['가중치']=0.2*minmaxscaler(rank_dist_sccnt['검색량'])-0.8*minmaxscaler(rank_dist_sccnt['거리'])
    rank_dist_sccnt=rank_dist_sccnt.sort_values("가중치",ascending=False)
    return rank_dist_sccnt.iloc[:30,0]


click=recommendation(data,data2,(37.530233,126.964864)) #서울시립대학교 위도 경도:37.5838657,127.0587771 미아 
click


def recommendation2(data,data2,address):
    address_list=address.split() #address는 시도, 시군구까지만
    local_list=data[(data['시도']==address_list[0]) & (data['시군구']==address_list[1])]
    local_list=list(local_list.iloc[:,0])
    result=list()
    popularity_list=list(data2.loc[:,'SRCHWRD_NM'])
    
    for j in popularity_list:
        for i in local_list:
            if j in i:
                result.append(i)
    
    result_list=list(dict.fromkeys(result))
    return result_list


print(recommendation2(data,data2,'서울특별시 성북구'))
print(season_classification(recommendation2(data,data2,'서울특별시 성북구')))


def recommendation3(click,data,category):
    recommend_list=[]
    click_record=data[data['관광지명']==click].squeeze()
    
    for n in range(len(data.index)):
        data_location=data.iloc[n,[-1,-2]]
        if haversine((click_record[-1],click_record[-2]),data_location,unit='km') < 5: #현 위치에서 5km 이내 추천 관광지만
            if data.iloc[n,3] == category: 
                recommend_list.append(data.iloc[n,0])
    shuffle(recommend_list)
    return recommend_list[:10]


print(season_classification(recommendation3(click.iloc[2],data,'지역축제')))
#category: 지역축제, 먹거리/패션거리, 토속/특산물/기념품매장, 휴양림/수목원, 관광농원/허브마을, 식물원`
#          테마공원/대형놀이공원, 지역호수/저수지, 지역사찰, 폭포/계곡, 서원/향교/서당, 유명사찰, 일반관광지
#          아쿠아리움/대형수족관

# +
#나중에 도움될 코드들
#model.wv.similarity('별빛','여행')

#df_tmp=pd.DataFrame(model3.wv.most_similar("여행",topn=1000),columns=['단어','유사도'])
#df_tmp

#data_record=data[data['관광지명']=='보현산별빛축제'].squeeze()
#season=classification(text_preprocessor(data_record,['롯데월드','에버랜드','경복궁','서울랜드']))
#season
#a=data[data['관광지명']=='에버랜드썸머워터펀']
#a['위도']
click.iloc[2]
# -


