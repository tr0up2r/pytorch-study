import nltk
nltk.download('punkt')

import urllib.request
import zipfile
from lxml import etree
import re
from nltk.tokenize import word_tokenize, sent_tokenize

# 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/master/ted_en-20160408.xml", filename="ted_en-20160408.xml")


targetXML=open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)

# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
parse_text = '\n'.join(target_text.xpath('//content/text()'))

# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# 해당 코드는 괄호로 구성된 내용을 제거.
content_text = re.sub(r'\([^)]*\)', '', parse_text)

# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
sent_text = sent_tokenize(content_text)

# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
normalized_text = []
for string in sent_text:
    tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
    normalized_text.append(tokens)

# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
result = []
result = [word_tokenize(sentence) for sentence in normalized_text]

print('총 샘플의 개수 : {}'.format(len(result)))

# 샘플 3개만 출력해보기.
for line in result[:3]:
    print(line)


from gensim.models import Word2Vec, KeyedVectors

# size = 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원.
# window = 컨텍스트 윈도우 크기.
# min_count = 단어 최소 빈도 수 제한. (빈도가 적은 단어들은 학습하지 않는다.)
# workers = 학습을 위한 프로세스 수.
# sg = 0은 CBOW, 1은 Skip-gram.

model = Word2Vec(sentences=result, size=100, window=5, min_count=5, workers=4, sg=0)

# Word2Vec을 통해 단어의 유사도를 계산할 수 있게 되었음.
model_result = model.wv.most_similar("man")
print(model_result)


# Word2Vec 모델 저장하고 로드하기.
# 모델 저장.
model.wv.save_word2vec_format('./eng_w2v')
# 모델 로드.
loaded_model = KeyedVectors.load_word2vec_format("eng_w2v")

# Load한 모델에 대해 다시 man과 유사한 단어 출력 수행.
model_result = loaded_model.most_similar("man")
print(model_result)