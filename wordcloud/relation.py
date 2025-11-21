import json, re, stanza, networkx as nx, matplotlib.pyplot as plt, platform
from collections import deque
from matplotlib import font_manager, rc
from networkx.drawing.nx_pydot import graphviz_layout

# --- Stanza 초기화 ---
stanza.download('ko')
nlp = stanza.Pipeline('ko')

# --- 한글 폰트 설정 ---
if platform.system() == 'Windows':
    font_path = "C:/Windows/Fonts/malgun.ttf"
elif platform.system() == 'Darwin':
    font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
else:
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font_prop = font_manager.FontProperties(fname=font_path)
rc('font', family=font_prop.get_name())

# --- 사전 불러오기 ---
with open("SentiWord_info.json", encoding="utf-8") as f:
    senti_dict = json.load(f)

with open("stopwords_ko.json", encoding="utf-8") as f:
    stopwords = set(json.load(f))

# --- 감정어 사전 ---
emotion_words = {
    item["word"]: float(item["polarity"])
    for item in senti_dict
    if abs(float(item.get("polarity", 0))) >= 0.3
}

filter_out = {"좋을", "살기", "받다", "할인", "욕을", "쓴", "해"}

# --- 텍스트 불러오기 ---
with open("test1.txt", encoding="utf-8") as f:
    text = f.read()

sentences = re.split(r'(?<=[.?!])\s+', text.strip())
person_names = ["가족", "엄마", "아빠", "부모", "동생", "친구", "상담사", "작업자", "동료", "BJ", "방송인"]
pronouns = ["그", "그녀", "너", "재", "걔"]
recent_entity = deque(maxlen=3)
pronoun_map = {}

# --- 필터 함수 ---
def is_valid_emotion(word, pos):
    return word in emotion_words and pos in ("ADJ", "VERB", "NOUN") and word not in stopwords and word not in filter_out

# --- 그래프 ---
G = nx.DiGraph()
G.add_node("내담자", type="client")
existing_nodes = set(["내담자"])

def add_node_once(graph, node, **kwargs):
    if node not in existing_nodes and node:
        graph.add_node(node, **kwargs)
        existing_nodes.add(node)

# --- 문장 처리 ---
for sent in sentences:
    sent = sent.strip()
    if not sent:
        continue
    doc = nlp(sent)
    for sentence in doc.sentences:
        tokens = sentence.words
        emotion_candidates = [w for w in tokens if is_valid_emotion(w.text, w.upos)]
        for emo_word in emotion_candidates:
            emotion = emo_word.text
            polarity = emotion_words.get(emotion, 0)
            subj = None

            for w in tokens:
                if w.head == emo_word.id and w.deprel == "nsubj" and (w.text in person_names or w.text.startswith("@NAME")):
                    subj = w.text

            found_persons = []
            if subj:
                found_persons.append(subj)
                recent_entity.append(subj)
            else:
                for name in person_names:
                    if name in sent:
                        found_persons.append(name)
                        recent_entity.append(name)
            for pro in pronouns:
                if pro in sent and pro in pronoun_map:
                    found_persons.append(pronoun_map[pro])
            if recent_entity:
                last = recent_entity[-1]
                for pro in ["그", "그녀", "재"]:
                    pronoun_map[pro] = last
            if not found_persons:
                continue

            for person in found_persons:
                add_node_once(G, person, type="person")
                add_node_once(G, emotion, type="emotion", polarity=polarity)
                G.add_edge("내담자", person)
                G.add_edge(person, emotion)

# --- Graphviz 기반 layout 적용 ---
pos = graphviz_layout(G, prog="dot")  # 계층 구조 (위→아래)

# --- 시각화 ---
plt.figure(figsize=(16, 12))
shapes = {"client": "o", "person": "D", "emotion": "s"}
colors = {
    "client": "#fff5cc",
    "person": "#add8e6",
    "emotion_pos": "#b2f0b2",
    "emotion_neg": "#ffc0cb"
}
sizes = {"client": 2500, "person": 2000, "emotion": 1600}

for t in shapes:
    if t == "emotion":
        pos_list = [n for n, d in G.nodes(data=True) if d["type"] == "emotion" and d["polarity"] > 0]
        neg_list = [n for n, d in G.nodes(data=True) if d["type"] == "emotion" and d["polarity"] < 0]
        nx.draw_networkx_nodes(G, pos, nodelist=pos_list, node_shape=shapes[t], node_color=colors["emotion_pos"], node_size=sizes[t])
        nx.draw_networkx_nodes(G, pos, nodelist=neg_list, node_shape=shapes[t], node_color=colors["emotion_neg"], node_size=sizes[t])
    else:
        ns = [n for n, d in G.nodes(data=True) if d["type"] == t]
        nx.draw_networkx_nodes(G, pos, nodelist=ns, node_shape=shapes[t], node_color=colors[t], node_size=sizes[t], alpha=0.95)

nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=10, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10, font_family=font_prop.get_name())
plt.axis("off")
plt.tight_layout()
plt.show()
