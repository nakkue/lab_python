from kiwipiepy import Kiwi
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 파일 읽기
with open("test8.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 내담자 대사 추출
client_lines = [line[len("내담자 :"):].strip() for line in lines if line.startswith("내담자 :")]
text_client = " ".join(client_lines)

# Kiwi 분석기
kiwi = Kiwi()
morphs = kiwi.analyze(text_client)[0][0]  # 첫 번째 분석 결과 가져오기

# 명사(NNG) + 2음절 이상 형용사만 필터링
filtered_tokens = []
for form, tag, start, end in morphs:
    if tag == 'NNG':  # 일반 명사
        filtered_tokens.append(form)
    elif tag == 'VA' and len(form) >= 2:  # 너무 짧은 형용사(예: 있, 없)는 제거
        filtered_tokens.append(form + '다')  # 어간이니까 '다' 붙여서 자연스럽게

# 단어 개수 세기
counter = Counter(filtered_tokens)

# 워드클라우드 생성
wc = WordCloud(font_path="C:/Windows/Fonts/malgun.ttf", background_color="white", width=800, height=400)
cloud = wc.generate_from_frequencies(counter)

# 시각화
plt.figure(figsize=(10, 5))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()
