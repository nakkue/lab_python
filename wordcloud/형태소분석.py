import matplotlib.pyplot as plt  # 시각화 라이브러리
from wordcloud import WordCloud  # 워드클라우드 생성
from collections import Counter  # 단어 빈도수 계산
from kiwipiepy import Kiwi  # 형태소 분석기

# 1. 텍스트 파일 읽기
with open("test8.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 2. 내담자 대사만 추출
client_lines = []
for line in lines:
    line = line.strip()
    if line.startswith("내담자 :"):  # "내담자 :"로 시작하는 줄만 필터링
        client_lines.append(line[len("내담자 :"):].strip())  # "내담자 :" 부분을 제거하고 공백 제거
text_client = " ".join(client_lines)   # 내담자 대사들을 하나의 문자열로 합치기

# 4. 형태소 분석기 준비
kiwi = Kiwi()

# 5. 형태소 분석 + 명사/형용사 필터링
morphs = kiwi.analyze(text_client)[0][0]
filtered_tokens = [form for form, tag, _, _ in morphs if tag in ('NNG', 'VA')]  # 명사(NNG)와 형용사(VA)만 필터링

# 6. 워드클라우드 생성
wc_filtered = WordCloud(font_path=r"C:/Windows/Fonts/malgun.ttf", background_color="white", width=800, height=400)
cloud_filtered = wc_filtered.generate_from_frequencies(Counter(filtered_tokens)) # 필터링된 단어 빈도수로 워드클라우드 생성

# 7. 시각화
plt.figure(figsize=(10, 5))
plt.imshow(cloud_filtered, interpolation='bilinear')
plt.axis('off')
plt.show()

