import json   # JSON 파일 로딩
from kiwipiepy import Kiwi  # 형태소 분석기
from wordcloud import WordCloud  # 워드클라우드 생성
import matplotlib.pyplot as plt  # 시각화 라이브러리
from soynlp.word import WordExtractor  # soynlp 기반 단어 추출

# 1. 감정 사전 로딩
with open(r"SentiWord_info.json", encoding='utf-8') as f:       
    senti_dict = json.load(f)  # JSON 파일에서 감정 사전 로딩   
emotion_words = {item['word']: item['polarity'] for item in senti_dict}  # 감정 단어와 극성 정보 추출
  
# 2. 텍스트 파일에서 상담 대화 불러오기
with open("test8.txt", "r", encoding="utf-8") as f:    
    lines = f.readlines()  # 파일에서 모든 줄 읽기

# 2-1. 내담자 대사만 추출
client_lines = []    
for line in lines:     
    line = line.strip()  # 각 줄의 앞뒤 공백 제거
    if line.startswith("내담자 :"):    
        client_lines.append(line[len("내담자 :"):].strip())    

text_client = " ".join(client_lines)  # 내담자 대사들을 하나의 문자열로 합치기

# 3. 불용어 후보 추출을 위한 단어 학습용 문장 리스트 생성
sentences = [line.strip() for line in client_lines if line.strip()]  # 내담자 대사에서 빈 줄 제거 후 리스트 생성

# 4. soynlp 기반 자주 나오는 단어 분석
word_extractor = WordExtractor(min_frequency=2, min_cohesion_forward=0.05)  # 단어 추출기 초기화
word_extractor.train(sentences)  # 학습용 문장으로 단어 추출기 학습
words_score = word_extractor.extract()  # 단어와 점수 추출

candidate_stopwords = {  #불용어로 간주할 단어들을 담을 집합 생성
    word for word, score in words_score.items() 
    if len(word) <= 1  #단어 길이가 1 이하이면
    or (score.leftside_frequency + score.rightside_frequency) >= 10  
        #해당 단어가 문장에서 너무 자주 등장하면 (앞,뒤에서 함께 자주 등장한 횟수가 10 이상이면)
        # 예: 조사, 접속사 등 문법적으로 자주 나오는 단어일 수 있으므로 불용어로 간주
}
    

# 5. 형태소 분석 + 명사/형용사 필터링    
kiwi = Kiwi() 
morphs = kiwi.analyze(text_client)[0][0]  # 형태소 분석 결과에서 첫 번째 분석 결과 추출
filtered_tokens = [form for form, tag, _, _ in morphs if tag in ('NNG', 'VA')]  # 명사(NNG)와 형용사(VA)만 필터링

# 6. 불용어 필터링 + 감정 단어만 남기기
filtered_emotions = [          
    word for word in filtered_tokens           
    if word not in candidate_stopwords and word in emotion_words              
]     

# 7. 감정 단어 개수 세기
unique_emotions = set(filtered_emotions)  # 고유 감정 단어 추출
emotion_freq = {word: filtered_emotions.count(word) for word in unique_emotions}  # 각 단어 빈도수 계산

# 8. 워드클라우드 생성
wc = WordCloud(
    font_path=r"C:/Windows/Fonts/malgun.ttf",  # 한글 폰트 경로
    background_color="white",    
    width=800,   
    height=400   
)
cloud = wc.generate_from_frequencies(emotion_freq)  # 감정 단어로 워드클라우드 생성

# 9. 시각화
plt.figure(figsize=(10, 5))      
plt.imshow(cloud, interpolation='bilinear')    
plt.axis('off')    
plt.show()  
