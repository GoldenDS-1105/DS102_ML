# Phân Loại Ảnh X-Quang Phổi Bằng Support Vector Machine (SVM)

Dự án này là bài tập thực hành môn học (Lab 03), tập trung vào việc tự triển khai thuật toán học máy Support Vector Machine (Soft-margin SVM) từ đầu bằng Numpy và so sánh với mô hình công nghiệp từ thư viện Scikit-Learn.

## Mục Tiêu Dự Án
- **Assignment 1:** Tự cài đặt mô hình Soft-margin SVM bằng thư viện `numpy`. Tối ưu hóa hàm mất mát Hinge Loss bằng phương pháp Stochastic Gradient Descent (SGD).
- **Assignment 2:** Sử dụng mô hình SVM chuẩn từ thư viện `sklearn` (`sklearn.svm.SVC`).
- **Đánh giá & So sánh:** Huấn luyện trên bộ dữ liệu ảnh y tế và đánh giá bằng các chỉ số Precision, Recall, F1-Score.

## Tập Dữ Liệu (Dataset)
Sử dụng bộ dữ liệu [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) từ Kaggle.
- **Tiền xử lý:** Ảnh được chuyển sang ảnh xám (Grayscale), resize về kích thước `128x128` pixels và làm phẳng (flatten) thành vector 1D (16384 features).
- **Chuẩn hóa:** Dữ liệu được chuẩn hóa Z-score Standardization `(X - mean) / std`.

## Kết Quả Đánh Giá (Test Set)

| Mô hình | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Custom SVM (Numpy)** | 62.90% | **100%** | 77.22% |
| **Sklearn SVM (SVC)** | **71.66%** | 99.23% | **83.22%** |

**Nhận xét:**
- Do đặc thù mất cân bằng dữ liệu của tập Chest X-Ray, cả hai mô hình đều đạt chỉ số **Recall rất cao** (gần 100%). Đây là một đặc tính tốt trong chẩn đoán y tế (hạn chế tối đa việc bỏ sót bệnh nhân).
- Mô hình Sklearn sử dụng các bộ solver tối ưu hơn nên đem lại độ chính xác (Precision) và F1-Score tổng thể cao hơn so với phương pháp SGD cơ bản tự cài đặt. Tuy nhiên, mô hình tự code vẫn chứng minh được sự hội tụ và tính đúng đắn của thuật toán.

## 🛠 Cách Chạy Code
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install numpy opencv-python tqdm scikit-learn
