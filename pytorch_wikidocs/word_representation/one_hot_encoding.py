from konlpy.tag import Okt
okt = Okt()

# 우선 문장에 대해 토큰화를 진행해준다.
token = okt.morphs("나는 자연어 처리를 배운다")
print(token)

# 그리고 각 토큰에 대해 고유한 index를 부여한다.
# 문장이 길어지면, 빈도수 순대로 단어를 정렬하여 부여한다.
word2index = {}
for voca in token:
    if voca not in word2index.keys():
        word2index[voca] = len(word2index)


def one_hot_encoding(word, word2index):
    one_hot_vector = [0]*(len(word2index))
    index = word2index[word]
    one_hot_vector[index] = 1
    return one_hot_vector


one_hot_encoding("자연어",word2index)