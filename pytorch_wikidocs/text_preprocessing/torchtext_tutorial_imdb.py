import urllib.request
import pandas as pd
from torchtext import data
from torchtext.data import TabularDataset

# 인터넷에서 IMDB 리뷰 데이터를 다운로드 받음.
# urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")

# IMDB 리뷰 데이터를 데이터 프레임에 저장 후, 상위 5개 행만 출력.
df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')
print(df.head())

# 전체 샘플의 개수 50,000개.
print(f'전체 샘플의 개수 : {format(len(df))}')

# training dataset과 test dataset를 절반씩 나눠서 분리.
train_df = df[:25000]
test_df = df[25000:]

# 각각을 csv 파일로 저장.
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)




# 필드 정의

# sequential : 시퀀스 데이터 여부. (True가 기본값)
# use_vocab : 단어 집합을 만들 것인지 여부. (True가 기본값)
# tokenize : 어떤 토큰화 함수를 사용할 것인지 지정. (string.split이 기본값)
# lower : 영어 데이터를 전부 소문자화한다. (False가 기본값)
# batch_first : 미니 배치 차원을 맨 앞으로 하여 데이터를 불러올 것인지 여부. (False가 기본값)
# is_target : 레이블 데이터 여부. (False가 기본값)
# fix_length : 최대 허용 길이. 이 길이에 맞춰서 패딩 작업(Padding)이 진행된다.

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True,
                  fix_length=20)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)

