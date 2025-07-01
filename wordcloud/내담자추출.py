import matplotlib.pyplot as plt  # 시각화 라이브러리
from wordcloud import WordCloud  # 워드클라우드 생성
from collections import Counter  # 단어 빈도수 계산

with open("test1.txt", "r", encoding="utf-8") as f:      
    lines = f.readlines()  # 파일에서 모든 줄 읽기

client_lines = []  # 내담자 대사만 추출
for line in lines:      
    line = line.strip()  # 각 줄의 앞뒤 공백 제거
    if line.startswith("내담자 :"):    
        client_lines.append(line[len("내담자 :"):].strip())  # 내담자 대사만 리스트에 추가

text_client = " ".join(client_lines)    # 내담자 대사들을 하나의 문자열로 합치기

wc = WordCloud(font_path=r"C:/Windows/Fonts/malgun.ttf", background_color="white", width=800, height=400)   
cloud = wc.generate_from_frequencies(Counter(text_client.split()))  # 단어 빈도수로 워드클라우드 생성
# 시각화
plt.figure(figsize=(10, 5))
plt.imshow(cloud, interpolation='bilinear')  
plt.axis('off')  
plt.show()  
