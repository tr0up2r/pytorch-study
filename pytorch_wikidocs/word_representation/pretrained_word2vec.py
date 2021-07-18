import gensim

# 구글의 사전 훈련된 Word2Vec 모델을 로드한다.
# 모델을 다운로드 해야 사용해볼 수 있다.
# 알맞은 모델 경로를 넣어주어야 동작.
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin 모델 경로', binary=True)

# 모델의 크기 확인.
print(model.vectors.shape)

# 두 단어의 유사도 계산하기.
print (model.similarity('this', 'is'))
print (model.similarity('post', 'book'))

# 단어 'book'의 벡터 출력
print(model['book'])