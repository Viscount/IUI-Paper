import jieba
import re
import codecs
import os


# 正则替换一下233333这种
# 替换列表，key为正则表达式，value为替换后的词
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

class WordSegment(object):
    def __init__(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        # Load user dict
        jieba.load_userdict(ROOT_DIR + '/userdict.txt')
        # Load stop words
        with codecs.open(ROOT_DIR + '/stopwords.txt','r','utf8') as f:
            self.stopwords = set([line.strip() for line in f])

    # Check whether the word is in replace list
    def check_replace(self, word):
        for item in REPLACE_DICT.keys():
            pattern = re.compile(item)
            if re.match(pattern, word) is not None:
                new_word = REPLACE_DICT[item]
                return new_word
        return(word)

    # Preprocess and segment the content
    def word_segment(self, content):
        words = []
        results = jieba.cut(content)
        for word in results:
            word = self.check_replace(word)
            if word not in self.stopwords:
                words.append(word.upper())
        return(words)

if __name__ == "__main__":
    ws = WordSegment()
    print(ws.word_segment("我要蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝蓝啦"))
    print(ws.word_segment("士郎这是要搞基？"))
    print(ws.word_segment("斯巴鲁！"))
    print(ws.word_segment("其实我大多数cp都可逆不可拆，但是有几对是不可逆不可拆的"))
    print(ws.word_segment("要是炸了就神作了"))
    print(ws.word_segment('爱蜜莉雅的声音？？'))
    print(ws.word_segment(',彩虹小哥哥？？？！！！对不起我跳戏了。。。'))
    print(ws.word_segment(',如果有真爱，那就是**色！'))
    print(ws.word_segment(',高能预警！高能预警！BGM即将上线，这不是演习！'))
    print(ws.word_segment('白内障，看不清，高锰酸钾滴眼睛'))
    print(ws.word_segment('吾等从未见过如此厚颜无耻之人'))
