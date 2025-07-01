from wordcloud import WordCloud  # 워드클라우드 생성
import matplotlib.pyplot as plt  # 시각화 라이브러리
from soynlp.word import WordExtractor  # soynlp 기반 단어 추출

# 텍스트 파일에서 상담 대화 불러오기
with open("test1.txt", "r", encoding="utf-8") as f:      
    lines = f.readlines()   # 파일에서 모든 줄 읽기

# 내담자 대사만 추출
client_lines = []  
for line in lines:   
    line = line.strip()  
    if line.startswith("내담자 :"):  # "내담자 :"로 시작하는 줄만 필터링
        client_lines.append(line[len("내담자 :"):].strip())  # "내담자 :" 부분을 제거하고 공백 제거

text_client = " ".join(client_lines)  # 내담자 대사들을 하나의 문자열로 합치기
sentences = [line.strip() for line in client_lines if line.strip()]  # 내담자 대사에서 빈 줄 제거 후 리스트 생성

# 단어 추출기 학습
word_extractor = WordExtractor(min_frequency=2, min_cohesion_forward=0.05)  # 단어 추출기 초기화
word_extractor.train(sentences)   
words_score = word_extractor.extract()  # 단어와 점수 추출

candidate_stopwords = {  #불용어로 간주할 단어들을 담을 집합 생성
    word for word, score in words_score.items() 
    if len(word) <= 1  #단어 길이가 1 이하이면
    or (score.leftside_frequency + score.rightside_frequency) >= 10  
        #해당 단어가 문장에서 너무 자주 등장하면 (앞,뒤에서 함께 자주 등장한 횟수가 10 이상이면)
        # 예: 조사, 접속사 등 문법적으로 자주 나오는 단어일 수 있으므로 불용어로 간주
}


# 워드클라우드 생성
wc = WordCloud(font_path=r"C:/Windows/Fonts/malgun.ttf", background_color="white", width=800, height=400)  
cloud = wc.generate_from_frequencies({      
    word: score.leftside_frequency + score.rightside_frequency   
    for word, score in words_score.items()     
    if word not in candidate_stopwords  # 불용어 후보에 포함되지 않는 단어만 사용    
})

plt.figure(figsize=(10, 5))    
plt.imshow(cloud, interpolation='bilinear')    
plt.axis('off')      
plt.show()    
