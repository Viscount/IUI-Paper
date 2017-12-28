#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:00:13 2017

@author: Doris
"""

import jieba
import re
import codecs
import os

# regular replace commanly appeared patterns
REPLACE_DICT = {
    "233+": "233",
    "33*": "233",
    "666+": "666",
    "哈哈+": "哈哈",
    "嘿嘿+": "嘿嘿",
    "啊啊+": "啊啊",
    "蓝+": "蓝",
    "紫+": "紫",
    "绿+": "绿",
    "红+": "红",
    "灰+": "灰",
    "黄+": "黄"
}

def check_replace(word):
    for item in REPLACE_DICT.keys():
        pattern = re.compile(item)
        if re.match(pattern, word) is not None:
            new_word = REPLACE_DICT[item]
            return new_word
    return(word)


# word segmentation
def word_segment(content):
    words = []
    results = jieba.cut(content)
    for word in results:
        word = check_replace(word)
        if word not in stopwords:
            words.append(word.upper())
    return(words)



if __name__ == "__main__":
    #load user defined dict
    jieba.load_userdict('userdict.txt')
 
    #load stop words
    f = codecs.open('stopwords.txt','r','utf8')
    stopwords = set()
    for line in f:
        word = line.split('\n')[0]
        stopwords.add(word)
    
    #test cases
    print(word_segment("我要蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝啦"))
    print(word_segment("士郎这是要搞基？"))
    print(word_segment("斯巴鲁！"))
    print(word_segment("其实我大多数cp都可逆不可拆，但是有几对是不可逆不可拆的"))
    print(word_segment("要是炸了就神作了"))
    print(word_segment('爱蜜莉雅的声音？？'))
    print(word_segment(',彩虹小哥哥？？？！！！对不起我跳戏了。。。'))
    print(word_segment(',如果有真爱，那就是**色！'))
    print(word_segment(',高能预警！高能预警！BGM即将上线，这不是演习！'))
    print(word_segment('白内障，看不清，高锰酸钾滴眼睛'))
    print(word_segment('吾等从未见过如此厚颜无耻之人'))
    
    #####  segment all TSC in the original data #####
    i=0
    danmaku_src = codecs.open('danmaku.csv','r','utf8')
    word_seg = codecs.open('word_seg.txt','w','utf8')
    for line in danmaku_src:
        #last column is TSC
        seg = word_segment(line.split(',')[-1].strip('\n')) 
        for ele in seg: 
            word_seg.write(ele+'\n')


    danmaku_src.close()
    word_seg.close()

    #######  count words #######
    wc_dict=dict() #dict for word count
    word_seg = codecs.open('word_seg.txt','r','utf8')

    i=0
    line = word_seg.readline()

    while line:
        i = i+1
        word = line.split('\n')[0]
        if word in wc_dict.keys():
            wc_dict[word] = wc_dict[word]+1
        else:
            wc_dict[word] = 1
        line = word_seg.readline()

    wc_df = pd.Series(wc_dict).reset_index()
    wc_df.rename(columns={'index':'word', 0:'count'},inplace=True)
    wc_df['word_len'] = w_df.wordapply(len)
    wc_df.to_csv('wc_result_final.txt')
  
    
    
    
    

    
    
    
    
    
    
    