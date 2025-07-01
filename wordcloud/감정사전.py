import json  # JSON 파일 로딩
from wordcloud import WordCloud  # 워드클라우드 생성
import matplotlib.pyplot as plt  # 시각화 라이브러리

# 1. 감정 사전 로딩
with open(r"SentiWord_info.json", encoding='utf-8') as f:   
    senti_dict = json.load(f)  # JSON 파일에서 감정 사전 로딩
emotion_words = {item['word']: item['polarity'] for item in senti_dict}  # 감정 단어와 극성 정보 추출

# 2. 상담 텍스트 불러오기
with open("test1.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()  # 파일에서 모든 줄 읽기

# 3. 내담자 대사만 추출
client_lines = []
for line in lines:
    line = line.strip()
    if line.startswith("내담자 :"):
        client_lines.append(line[len("내담자 :"):].strip())  # "내담자 :" 부분을 제거하고 공백 제거

text_client = " ".join(client_lines)

# 4. 감정 단어 필터링 (형태소 분석 없이, 문자열에서 단어 매칭)
filtered_emotions = [   
    word for word in text_client.split()   # 공백으로 단어 분리
    if word in emotion_words  # 감정 단어 사전에 있는 단어만 필터링
]   

# 5. 감정 단어 빈도 계산
unique_emotions = set(filtered_emotions)  # 고유 감정 단어 추출
emotion_freq = {word: filtered_emotions.count(word) for word in unique_emotions}  # 각 단어 빈도수 계산

# 6. 워드클라우드 생성
wc = WordCloud(  
    font_path=r"C:/Windows/Fonts/malgun.ttf",  
    background_color="white",
    width=800,
    height=400
)
cloud = wc.generate_from_frequencies(emotion_freq)  # 감정 단어로 워드클라우드 생성

# 7. 시각화
plt.figure(figsize=(10, 5))
plt.imshow(cloud, interpolation='bilinear')  
plt.axis('off')
plt.show()
