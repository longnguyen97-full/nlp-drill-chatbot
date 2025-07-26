import json

# --- Bước 1: Xây dựng Map tra cứu ---
# Mục tiêu: Tạo một dictionary cho phép truy cập nhanh nội dung điều luật từ aid
# aid_to_content_map = {1: "Nội dung điều 1...", 2: "Nội dung điều 2...", ...}

common_path = 'data'

def build_aid_to_content_map(file_path):
    """Builds a map from aid to content from the given JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return {article['aid']: article['content_Article'] 
                for law_doc in json.load(f) 
                for article in law_doc['content']}

print("Đang xây dựng map tra cứu aid -> content...")
aid_to_content_map = build_aid_to_content_map(f'{common_path}/legal_corpus.json')
print(f"Hoàn thành! Có {len(aid_to_content_map)} điều luật trong kho tri thức.")

# --- Bước 2: Đọc và hiểu file train.json ---
def load_json_data(file_path):
    """Loads JSON data from the given file path."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

print("\nĐọc file train.json...")
train_data = load_json_data(f'{common_path}/train.json')

# --- Bước 3: Thử liên kết dữ liệu ---
# Lấy một mẫu từ train_data để xem nó hoạt động thế nào
sample_question = train_data[0]
qid = sample_question['qid']
question_text = sample_question['question']
relevant_aids = sample_question['relevant_laws']

print(f"\n--- VÍ DỤ LIÊN KẾT DỮ LIỆU ---")
print(f"Câu hỏi (QID: {qid}): {question_text}")
print(f"Các điều luật liên quan (AID): {relevant_aids}")

for aid in relevant_aids:
    content = aid_to_content_map.get(aid)
    if content:
        print(f"\nNội dung của AID {aid}:")
        # In 150 ký tự đầu tiên cho gọn
        print(f"'{content[:150]}...'")
    else:
        print(f"Cảnh báo: Không tìm thấy AID {aid} trong legal_corpus.")