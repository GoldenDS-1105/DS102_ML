# Lab04 - Decision Tree & Random Forest

## 1. Mục tiêu bài lab

Lab04 thực hành xây dựng và đánh giá hai mô hình học máy cơ bản:

- Decision Tree
- Random Forest

Bài lab gồm hai hướng triển khai:

- Tự cài đặt mô hình bằng NumPy.
- Sử dụng thư viện Scikit-learn để so sánh kết quả.

Dataset được sử dụng là **Wine Quality** từ UCI Machine Learning Repository.

---

## 2. Cấu trúc thư mục

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

Ý nghĩa các file:

| File | Mô tả |
|---|---|
| `assignments (1).ipynb` | Notebook đề bài hoặc phần thực hành ban đầu |
| `decision_tree_numpy.py` | Cài đặt Decision Tree bằng NumPy |
| `decision_tree_sklearn.py` | Cài đặt wrapper Decision Tree bằng Scikit-learn |
| `main_dt.py` | File chạy và so sánh Decision Tree NumPy với Scikit-learn |
| `random_forest_numpy.py` | Cài đặt Random Forest bằng NumPy |
| `random_forest_sklearn.py` | Cài đặt wrapper Random Forest bằng Scikit-learn |
| `main_rdf.py` | File chạy và so sánh Random Forest NumPy với Scikit-learn |

---

## 3. Cài đặt thư viện

Chạy lệnh sau trong terminal:

```bash
pip install numpy pandas scikit-learn ucimlrepo
```

Nếu máy dùng `python3`, có thể dùng:

```bash
python3 -m pip install numpy pandas scikit-learn ucimlrepo
```

---

## 4. Cách chạy chương trình

Di chuyển vào thư mục `Lab04`:

```bash
cd Lab04
```

Chạy Decision Tree:

```bash
python3 main_dt.py
```

Chạy Random Forest:

```bash
python3 main_rdf.py
```

Nếu máy có lệnh `python`, có thể dùng:

```bash
python main_dt.py
python main_rdf.py
```

---

## 5. Nội dung thực hiện

### 5.1 Decision Tree bằng NumPy

File `decision_tree_numpy.py` tự cài đặt mô hình Decision Tree với các bước chính:

- Tính độ đo phân chia node.
- Tìm feature và threshold tốt nhất.
- Xây dựng cây theo đệ quy.
- Dự đoán nhãn cho tập test.

### 5.2 Decision Tree bằng Scikit-learn

File `decision_tree_sklearn.py` sử dụng `DecisionTreeClassifier` của Scikit-learn để train và dự đoán.

### 5.3 Random Forest bằng NumPy

File `random_forest_numpy.py` tự cài đặt Random Forest bằng cách:

- Tạo nhiều cây Decision Tree.
- Lấy mẫu bootstrap cho từng cây.
- Dự đoán bằng majority voting.

### 5.4 Random Forest bằng Scikit-learn

File `random_forest_sklearn.py` sử dụng `RandomForestClassifier` của Scikit-learn để so sánh với bản tự cài đặt.

---

## 6. Đánh giá mô hình

Các mô hình được đánh giá bằng chỉ số:

```text
F1-score
```

Bài lab so sánh kết quả giữa:

- Decision Tree NumPy
- Decision Tree Scikit-learn
- Random Forest NumPy
- Random Forest Scikit-learn

---

## 7. Ghi chú

- Dataset Wine Quality được tải bằng thư viện `ucimlrepo`.
- Nếu chạy lần đầu, máy cần có kết nối Internet để tải dataset.
- Nếu terminal báo không có lệnh `python`, hãy dùng `python3`.
