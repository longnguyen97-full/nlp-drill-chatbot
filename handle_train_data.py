import json
import csv
import random
import unicodedata
from tqdm import tqdm # hiển thị progress bar

def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace('\n', ' ').strip()
    return text

def load_legal_articles(corpus_path):
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    aid2content = {}
    for doc in corpus:
        for article in doc["content"]:
            aid = article["aid"]
            content = article["content_Article"]
            aid2content[aid] = content
    return aid2content

def load_train_data(train_path):
    with open(train_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_triplets(train_data, aid2content):
    triplets = []
    all_aids = list(aid2content.keys())

    for item in tqdm(train_data, desc="Generating triplets"):
        q = clean_text(item["question"])
        pos_aids = item["relevant_laws"]

        for pos_aid in pos_aids:
            pos_text = clean_text(aid2content.get(pos_aid, ""))
            if not pos_text:
                continue

            # chọn negative aid không trùng positive
            while True:
                neg_aid = random.choice(all_aids)
                if neg_aid not in pos_aids:
                    break
            neg_text = clean_text(aid2content[neg_aid])
            triplets.append((q, pos_text, neg_text))
    return triplets

def generate_pairs(train_data, aid2content, negative_ratio=2):
    pairs = []
    all_aids = list(aid2content.keys())

    for item in tqdm(train_data, desc="Generating pairs"):
        q = clean_text(item["question"])
        pos_aids = item["relevant_laws"]

        # Positive pairs
        for aid in pos_aids:
            pos_text = clean_text(aid2content.get(aid, ""))
            if pos_text:
                pairs.append((q, pos_text, 1))
        
        # Negative pairs
        neg_needed = len(pos_aids) * negative_ratio
        sampled_neg = set()
        while len(sampled_neg) < neg_needed:
            neg_aid = random.choice(all_aids)
            if neg_aid not in pos_aids and neg_aid not in sampled_neg:
                neg_text = clean_text(aid2content[neg_aid])
                pairs.append((q, neg_text, 0))
                sampled_neg.add(neg_aid)
    return pairs

def save_csv_triplets(triplets, out_path):
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["question", "positive", "negative"])
        for row in triplets:
            writer.writerow(row)

def save_csv_pairs(pairs, out_path):
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["question", "article", "label"])
        for row in pairs:
            writer.writerow(row)

if __name__ == "__main__":
    # === Đường dẫn dữ liệu gốc ===
    data_path = "data"
    output_path = f"{data_path}/output"
    train_path = f"{data_path}/train.json"
    corpus_path = f"{data_path}/legal_corpus.json"

    # === Đọc dữ liệu ===
    print("🔄 Đang đọc dữ liệu...")
    aid2content = load_legal_articles(corpus_path)
    train_data = load_train_data(train_path)

    # === Tạo triplet và pair ===
    triplets = generate_triplets(train_data, aid2content)
    pairs = generate_pairs(train_data, aid2content)

    # === Ghi ra file ===
    print("💾 Đang lưu file CSV...")
    save_csv_triplets(triplets, f"{output_path}/train_triplets.csv")
    save_csv_pairs(pairs, f"{output_path}/train_pairs.csv")
    print("✅ Hoàn tất. Đã tạo xong 2 file:")
    print(f"- {output_path}/train_triplets.csv")
    print(f"- {output_path}/train_pairs.csv")
