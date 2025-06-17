import json
from kiwipiepy import Kiwi
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. 감정 사전 로딩
with open(r"SentiWord_info.json", encoding='utf-8') as f:
    senti_dict = json.load(f)

emotion_words = {item['word']: item['polarity'] for item in senti_dict}

# 2. 텍스트 파일에서 상담 대화 불러오기
with open("test.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 2-1. 내담자 대화만 추출
client_lines = []
for line in lines:
    line = line.strip()
    if line.startswith("내담자 :"):
        client_lines.append(line[len("내담자 :"):].strip())

text_client = " ".join(client_lines)

# 3. 형태소 분석 (kiwipiepy 사용)
kiwi = Kiwi()
result = kiwi.tokenize(text_client)
tokens = [token.form for token in result]

# 4. 감정 단어 필터링
filtered_emotions = [word for word in tokens if word in emotion_words]

unique_emotions = set(filtered_emotions)
emotion_freq = {word:1 for word in unique_emotions}

# 5. 워드클라우드 생성
wc = WordCloud(
    font_path = r"C:/Windows/Fonts/malgun.ttf",  # 윈도우 기본 한글 폰트
    background_color="white",
    width=800,
    height=400
)
cloud = wc.generate_from_frequencies(emotion_freq)

# 6. 시각화
plt.figure(figsize=(10,5))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

