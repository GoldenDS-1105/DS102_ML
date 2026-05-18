# DS102 - Machine Learning Labs

Repository này lưu trữ các bài thực hành môn **DS102 - Machine Learning**.

## 1. Thông tin chung

- Môn học: DS102 - Machine Learning
- Nội dung: Các bài lab thực hành về tiền xử lý dữ liệu, mô hình học máy và đánh giá mô hình.
- Ngôn ngữ sử dụng: Python
- Công cụ chính: Jupyter Notebook, NumPy, Pandas, Scikit-learn

---

## 2. Cấu trúc repository

```text
DS102_ML/
├── Lab01/
├── Lab02/
├── Lab03/
├── Lab04/
├── README.md
└── .gitignore
```

---

## 3. Danh sách bài lab

| Lab | Nội dung chính | Trạng thái |
|---|---|---|
| Lab01 | Bài thực hành nền tảng ban đầu | Đã hoàn thành |
| Lab02 | Bài thực hành tiếp theo về xử lý dữ liệu / mô hình cơ bản | Đã hoàn thành |
| Lab03 | Support Vector Machine và bài toán phân loại ảnh X-ray | Đã hoàn thành |
| Lab04 | Decision Tree và Random Forest trên Wine Quality Dataset | Đã hoàn thành |

---

## 4. Mô tả từng lab

### Lab01

Thư mục `Lab01` chứa các file thực hành đầu tiên của môn học.

### Lab02

Thư mục `Lab02` chứa các bài thực hành tiếp theo, phục vụ quá trình làm quen với pipeline học máy.

### Lab03

Thư mục `Lab03` chứa bài thực hành liên quan đến:

- SVM
- SMO
- Lagrange KKT Multiplier
- Dataset ảnh X-ray ngực

Cấu trúc chính:

```text
Lab03/
├── SMO.ipynb
├── SVM.py
├── main.py
├── preliminaries.ipynb
├── lagrange-KTT-multiplier.ipynb
└── chest_xray/
```

### Lab04

Thư mục `Lab04` chứa bài thực hành liên quan đến:

- Decision Tree tự cài đặt bằng NumPy
- Random Forest tự cài đặt bằng NumPy
- Decision Tree bằng Scikit-learn
- Random Forest bằng Scikit-learn
- Đánh giá bằng F1-score

Cấu trúc chính:

```text
Lab04/
├── assignments (1).ipynb
├── decision_tree_numpy.py
├── decision_tree_sklearn.py
├── main_dt.py
├── main_rdf.py
├── random_forest_numpy.py
└── random_forest_sklearn.py
```

---

## 5. Cài đặt môi trường

Cài đặt các thư viện cần thiết:

```bash
pip install numpy pandas scikit-learn matplotlib ucimlrepo
```

Nếu sử dụng `python3`:

```bash
python3 -m pip install numpy pandas scikit-learn matplotlib ucimlrepo
```

---

## 6. Cách chạy bài lab

Clone repository:

```bash
git clone https://github.com/GoldenDS-1105/DS102_ML.git
```

Di chuyển vào thư mục repo:

```bash
cd DS102_ML
```

Chạy từng lab bằng cách vào thư mục tương ứng.

Ví dụ chạy Lab04:

```bash
cd Lab04
python3 main_dt.py
python3 main_rdf.py
```

---

## 7. Ghi chú

- Các thư mục lab được tách riêng để dễ quản lý.
- File `.DS_Store` và thư mục `__pycache__` không nên commit lên GitHub.
- Với các dataset lớn, nên cân nhắc dùng `.gitignore` hoặc Git LFS nếu cần.
