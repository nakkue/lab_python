import json
from kiwipiepy import Kiwi
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from soynlp.word import WordExtractor

# 1. 감정 사전 로딩
with open(r"SentiWord_info.json", encoding='utf-8') as f:
    senti_dict = json.load(f)

emotion_words = {item['word']: item['polarity'] for item in senti_dict}

# 2. 텍스트 파일에서 상담 대화 불러오기
with open("test.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 2-1. 내담자 대사만 추출
client_lines = []
for line in lines:
    line = line.strip()
    if line.startswith("내담자 :"):
        client_lines.append(line[len("내담자 :"):].strip())

text_client = " ".join(client_lines)

# 3. 불용어 후보 추출을 위한 단어 학습용 문장 리스트 생성
sentences = [line.strip() for line in client_lines if line.strip()]

# 4. soynlp 기반 자주 나오는 단어 분석
word_extractor = WordExtractor(min_frequency=2, min_cohesion_forward=0.05)
word_extractor.train(sentences)
word_scores = word_extractor.extract()  # 변수 이름 통일

# 4-1. 상위 등장 단어 중 의미 없는 단어(짧거나 응집도 낮은 단어) 제거
candidate_stopwords = set()
for word, score in word_scores.items():
    if len(word) <= 1 or score.cohesion_forward < 0.3:
        candidate_stopwords.add(word)

# 5. 형태소 분석 (kiwipiepy)
kiwi = Kiwi()
tokens = [token.form for token in kiwi.tokenize(text_client)]

# 6. 불용어 필터링 + 감정 단어만 남기기
filtered_emotions = [
    word for word in tokens
    if word not in candidate_stopwords and word in emotion_words
]

# 7. 감정 단어 개수 세기
unique_emotions = set(filtered_emotions)
emotion_freq = {word: filtered_emotions.count(word) for word in unique_emotions}

# 8. 워드클라우드 생성
wc = WordCloud(
    font_path=r"C:/Windows/Fonts/malgun.ttf",  # 윈도우 기본 한글 폰트
    background_color="white",
    width=800,
    height=400
)
cloud = wc.generate_from_frequencies(emotion_freq)

# 9. 시각화
plt.figure(figsize=(10, 5))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()
