# Giới thiệu dự án: Truy vấn điều luật Việt Nam

Dự án này nhằm mục đích xây dựng một hệ thống truy vấn điều luật Việt Nam. Dự án được chia thành hai giai đoạn chính:
- **Phase 1**: Xây dựng mô hình xử lý ngôn ngữ tự nhiên để hiểu và phân tích các điều luật.
- **Phase 2**: Tạo chatbot để hỗ trợ người dùng truy vấn và tìm kiếm thông tin về các điều luật.

## Công nghệ áp dụng
Dự án sử dụng các công nghệ tiên tiến trong lĩnh vực xử lý ngôn ngữ tự nhiên và học máy, bao gồm:
- Mô hình ngôn ngữ BERT
- Thư viện TensorFlow và PyTorch
- Công cụ xử lý ngôn ngữ tự nhiên spaCy

## Hướng dẫn sử dụng chatbot
Để sử dụng chatbot, người dùng có thể truy cập vào nền tảng trực tuyến của chúng tôi và nhập câu hỏi hoặc từ khóa liên quan đến điều luật mà họ quan tâm. Chatbot sẽ phân tích và cung cấp thông tin chi tiết về điều luật đó.

## Hướng dẫn chạy source

Để chạy source code của dự án, bạn cần thực hiện các bước sau:

1. **Tạo môi trường ảo (virtual environment):**
     ```bash
     python3 -m venv venv
     source venv/Scripts/activate
     ```

2. **Cài đặt các thư viện cần thiết:**
   - Sau khi kích hoạt môi trường ảo, cài đặt các thư viện cần thiết bằng lệnh:
     ```bash
     pip install -r requirements.txt
     ```

3. **Chạy ứng dụng:**
   - Sau khi cài đặt xong, bạn có thể thực thi ứng dụng bằng lệnh:
     ```bash
     python3 explore_data.py
     ```

Lưu ý: Đảm bảo rằng bạn đã cài đặt Python và pip trên hệ thống của mình trước khi thực hiện các bước trên.
