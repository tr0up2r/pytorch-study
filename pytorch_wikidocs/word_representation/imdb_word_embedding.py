from torchtext.legacy import data, datasets
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
from torchtext.vocab import Vectors

# 두 개의 Field 객체를 정의.
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)

# IMDB dataset 다운로드 후, train dataset과 test dataset으로 나눠본다.
trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

print('훈련 데이터의 크기 : {}' .format(len(trainset)))
print(vars(trainset[0]))


# 앞서 만들어두었던 모델을 load해서 사용.
word2vec_model = KeyedVectors.load_word2vec_format('eng_w2v')

# 영어 단어 'this'의 임베딩 벡터값 출력
print(word2vec_model['this'])