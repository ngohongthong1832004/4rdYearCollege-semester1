# TỔNG HỢP CÁC DẠNG BÀI TẬP HỆ KHUYẾN NGHỊ
### Phân loại theo chủ đề từ Lab 01 đến Lab 05

---

## 📊 I. TÍNH TOÁN KHOẢNG CÁCH & ĐỘ TƯƠNG ĐỒNG

### **1.1. Tính khoảng cách giữa các vector**
- **Lab 01 - Problem 1a**: Tính Euclidean Distance và Manhattan Distance giữa các users
- **Công thức sử dụng:**
  - Euclidean: `scipy.spatial.distance.euclidean()`
  - Manhattan: `scipy.spatial.distance.cityblock()`
- **Mục đích**: So sánh độ gần giữa các users dựa trên ratings

### **1.2. Tính độ tương đồng (Similarity)**
- **Lab 01 - Problem 1b**: Tạo ma trận Pearson Similarity
- **Lab 02 - Problem 2**: Jaccard Similarity giữa các phim
- **Lab 03**: Cosine Similarity trong lọc cộng tác
- **Công thức sử dụng:**
  - Pearson: `scipy.stats.pearsonr()`
  - Jaccard: `sklearn.metrics.jaccard_score()`
  - Cosine: `sklearn.metrics.pairwise.cosine_similarity()`
- **Ứng dụng**: Tìm users/items tương tự nhau

---

## 🎯 II. HỆ KHUYẾN NGHỊ CƠ BẢN (POPULARITY-BASED)

### **2.1. Weighted Rating (WR)**
- **Lab 01 - Problem 2**: Khuyến nghị theo xu hướng cho người dùng mới
- **Công thức IMDB:**
  ```
  WR = (v/(v+m) × R) + (m/(v+m) × C)
  ```
  - v: số lượt vote
  - m: ngưỡng tối thiểu (quantile)
  - R: rating trung bình của phim
  - C: rating trung bình toàn bộ
- **Xử lý**: Cold-start problem
- **Bài tập tương tự**: Tất cả bài về popularity-based ranking

---

## 🤝 III. LỌC CỘNG TÁC (COLLABORATIVE FILTERING)

### **3.1. User-based Collaborative Filtering**
- **Lab 02 - Problem 1a**: Dự đoán rating bằng User-based CF
- **Lab 03 - Bài 1**: Lập trình class CF với User-based
- **Lab 03 - Bài 2**: KNN với user similarity
- **Nguyên lý**: Tìm k users gần nhất với user mục tiêu, lấy trung bình rating của họ
- **Similarity**: Pearson correlation
- **Công thức dự đoán:**
  ```
  P(a,i) = r̄(a) + [Σ(r(u,i) - r̄(u)) × w(a,u)] / Σ|w(a,u)|
  ```

### **3.2. Item-based Collaborative Filtering**
- **Lab 02 - Problem 1b**: Dự đoán rating bằng Item-based CF
- **Lab 03 - Bài 1**: Lập trình class CF với Item-based
- **Nguyên lý**: Tìm k items tương tự với item mục tiêu mà user đã rated
- **Similarity**: Adjusted Cosine Similarity
- **Công thức:**
  ```
  w(i,j) = Σ[(r(u,i) - r̄(u))(r(u,j) - r̄(u))] / [√Σ(r(u,i) - r̄(u))² × √Σ(r(u,j) - r̄(u))²]
  ```

### **3.3. K-Nearest Neighbors (KNN)**
- **Lab 03 - Bài 2**: Xây dựng KNN model với `sklearn.neighbors.NearestNeighbors`
- **Lab 03**: Step-by-step KNN predictions
- **Lab 05 - Bài 1**: Content-based KNN với scikit-learn
- **Metric**: Cosine similarity
- **Số neighbors**: Thường k=20-30
- **Ứng dụng**: Item-item và User-user CF

### **3.4. So sánh User-based vs Item-based**
- **Lab 03**: Comparing item-based and user-based models
- **RMSE Comparison**: Item-based thường tốt hơn User-based
- **Độ phức tạp**: Item-based nhanh hơn khi #users >> #items

---

## 📝 IV. HỆ KHUYẾN NGHỊ DỰA TRÊN NỘI DUNG (CONTENT-BASED)

### **4.1. TF-IDF Features**
- **Lab 02 - Problem 2**: Tạo genre cross-table
- **Lab 04 - Bài 3**: TF-IDF cho movie genres
- **Công cụ**: `sklearn.feature_extraction.text.TfidfTransformer`
- **Đầu vào**: Ma trận genres (binary hoặc text)
- **Đầu ra**: Feature vector cho mỗi item

### **4.2. Ridge Regression cho User Profile**
- **Lab 04 - Bài 3**: Ridge Regression với TF-IDF features
- **Công thức**: 
  ```
  minimize ||Xw - y||² + λ||w||²
  ```
- **Mục đích**: Học user preference từ rated items
- **Công cụ**: `sklearn.linear_model.Ridge`

### **4.3. One-Hot Encoding**
- **Lab 05 - Bài 1**: OneHotEncoder cho genres
- **Lab 02 - Problem 2**: pd.crosstab() cho genre matrix
- **Ứng dụng**: Chuyển categorical data thành numerical

### **4.4. Count Vectorizer**
- **Lab 05 - Bài 3**: CountVectorizer cho combined features
- **Features**: keywords + cast + genres + director
- **Mục đích**: Tạo count matrix từ text data

---

## 🔢 V. MATRIX FACTORIZATION (MF)

### **5.1. Gradient Descent cho MF**
- **Lab 04 - Bài 2**: Class MF với gradient descent
- **Mục tiêu**: Phân tích R ≈ X × W
  - X: item latent factors (n_items × K)
  - W: user latent factors (K × n_users)
  - K: số chiều ẩn (thường 2-20)
- **Loss function**: 
  ```
  L = 0.5 × Σ(r - x·w)² + λ/2 × (||X||² + ||W||²)
  ```
- **Regularization**: Tránh overfitting với λ (thường 0.01-0.1)

### **5.2. SVD (Singular Value Decomposition)**
- **Lab 04 - Bài 1**: Decomposition và reconstruction
- **Công thức**: A = U × Σ × V^T
- **Công cụ**: `scipy.linalg.svd()`
- **Ứng dụng**: Dimensionality reduction, topic modeling

### **5.3. Truncated SVD**
- **Lab 05 - Bài 2**: TruncatedSVD từ sklearn
- **Số components**: n_components=12
- **Correlation matrix**: `np.corrcoef()` trên transformed matrix
- **Ứng dụng**: Collaborative filtering với ma trận sparse

---

## 🔍 VI. TÌM KIẾM ITEMS TƯƠNG TỰ

### **6.1. Finding Movie Pairs**
- **Lab 01 - Problem 3**: Tìm tất cả cặp phim được xem bởi cùng user
- **Công cụ**: `itertools.combinations()` hoặc `permutations()`
- **GroupBy**: Nhóm theo userId để tìm pairs
- **Output**: Counting occurrences của từng pair

### **6.2. Making Recommendations**
- **Lab 01 - Problem 3**: Khuyến nghị dựa trên movie co-occurrence
- **Lab 02 - Problem 2**: Recommendations based on Jaccard similarity
- **Lab 03**: Recommend function trong class CF
- **Nguyên tắc**: Predicted rating > threshold → recommend

---

## 📈 VII. ĐÁNH GIÁ MÔ HÌNH (EVALUATION)

### **7.1. Root Mean Squared Error (RMSE)**
- **Lab 03**: RMSE cho User-based và Item-based CF
- **Lab 04**: RMSE với Matrix Factorization
- **Công thức**: 
  ```
  RMSE = √[Σ(predicted - actual)² / n]
  ```
- **Kết quả điển hình**:
  - Content-based: ~1.2-1.3
  - CF Neighborhood: ~0.99
  - Matrix Factorization: ~0.87-1.02

### **7.2. Train-Test Split**
- **Lab 04**: `sklearn.model_selection.train_test_split()`
- **Tỷ lệ**: 67% train, 33% test (hoặc 80-20)
- **MovieLens**: Có sẵn ub.base và ub.test

---

## 🧹 VIII. XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)

### **8.1. Xử lý Missing Values**
- **Lab 03**: Fill NaN với 0 sau khi center data
- **Lab 04**: Dropna() trước khi train
- **Lab 05**: fillna('') cho text features

### **8.2. Normalization (Mean Centering)**
- **Lab 03**: Centered ratings = ratings - user_mean
- **Mục đích**: Loại bỏ user bias
- **Ứng dụng**: User-based và Item-based CF

### **8.3. Pivot Table**
- **Lab 03**: `df.pivot(index='userId', columns='title', values='rating')`
- **Mục đích**: Tạo user-item matrix
- **Fill**: fillna(0) hoặc fillna(user_mean)

### **8.4. Sparse Matrix**
- **Lab 03**: `scipy.sparse.csr_matrix()` cho CF
- **Lý do**: Tiết kiệm memory khi ma trận có nhiều giá trị 0
- **Ứng dụng**: MovieLens 1M, 100K

### **8.5. Removing Noise**
- **Lab 03 - Bài 2**: Lọc movies có vote_count >= threshold
- **Lab 03 - Bài 2**: Lọc users có votes >= threshold
- **Nguyên tắc**: Percentile (70th, 90th)

---

## 📊 IX. TRỰC QUAN HÓA (VISUALIZATION)

### **9.1. Heatmap**
- **Lab 01**: Pearson similarity heatmap với seaborn
- **Code**: `sns.heatmap(matrix, annot=True, cmap='coolwarm')`

### **9.2. Bar Chart**
- **Lab 01**: Plotting top recommended movies
- **Code**: `df.plot.bar(x='movie', y='count')`

### **9.3. Scatter Plot**
- **Lab 03**: Vote count distribution
- **Mục đích**: Xác định threshold cho filtering

---

## 🗂️ X. CẤU TRÚC DỮ LIỆU MOVIELENS

### **10.1. MovieLens 100K**
- ratings: userId, movieId, rating, timestamp
- movies: movieId, title, genres
- users: userId, age, sex, occupation, zip_code

### **10.2. MovieLens 1M**
- ~1 triệu ratings
- 6000 users, 4000 movies
- Format tương tự 100K

### **10.3. Metadata CSV**
- title, genres, vote_count, vote_average
- keywords, cast, director
- Dùng cho content-based filtering

---

## 🛠️ XI. THƯ VIỆN & CÔNG CỤ

### **11.1. Core Libraries**
- `pandas`: Data manipulation
- `numpy`: Matrix operations
- `scipy`: Distance, sparse matrix
- `sklearn`: ML algorithms

### **11.2. Recommendation Algorithms**
- `sklearn.neighbors.NearestNeighbors`: KNN
- `sklearn.decomposition.TruncatedSVD`: Matrix factorization
- `sklearn.linear_model.Ridge`: Regression
- `sklearn.metrics.pairwise`: Similarity measures

### **11.3. Feature Engineering**
- `sklearn.feature_extraction.text.TfidfTransformer`
- `sklearn.feature_extraction.text.CountVectorizer`
- `sklearn.preprocessing.OneHotEncoder`

---

## 📝 XII. DẠNG BÀI TẬP THƯỜNG GẶP

### **Dạng 1: Tính toán cơ bản**
✅ Tính khoảng cách/similarity giữa users/items  
✅ Normalize ratings (mean centering)  
✅ Tạo utility matrix (pivot table)  

### **Dạng 2: Dự đoán rating**
✅ User-based CF: Dự đoán r(u,i)  
✅ Item-based CF: Dự đoán r(u,i)  
✅ Matrix Factorization: Dự đoán bằng X·W  

### **Dạng 3: Xây dựng hệ thống**
✅ Lập trình class CF từ đầu  
✅ Sử dụng sklearn để build recommender  
✅ Tích hợp nhiều features (hybrid)  

### **Dạng 4: Đánh giá & tối ưu**
✅ Tính RMSE trên test set  
✅ So sánh các phương pháp  
✅ Hyperparameter tuning (k, λ, learning_rate)  

### **Dạng 5: Xử lý dữ liệu**
✅ Handle missing values  
✅ Remove noise (low vote count)  
✅ Feature engineering (TF-IDF, one-hot)  

### **Dạng 6: Cold-start problem**
✅ Popularity-based cho user mới  
✅ Content-based cho item mới  
✅ Hybrid approaches  

---

## 🎓 XIII. TỔNG KẾT THEO LAB

| Lab | Chủ đề chính | Kỹ thuật |
|-----|--------------|----------|
| **Lab 01** | Distance & Similarity, Popularity-based, Movie pairs | Euclidean, Manhattan, Pearson, Weighted Rating, Combinations |
| **Lab 02** | Collaborative Filtering (manual), Content-based basics | User-based CF, Item-based CF, Jaccard, Crosstab |
| **Lab 03** | CF with OOP, KNN, Sparse matrix | Class CF, NearestNeighbors, Mean centering, RMSE |
| **Lab 04** | SVD, Matrix Factorization, Content-based Ridge | Gradient descent, Regularization, TF-IDF, Ridge regression |
| **Lab 05** | Sklearn ecosystem | OneHotEncoder, TruncatedSVD, CountVectorizer, KNN |

---

## 💡 XIV. TIPS & BEST PRACTICES

### **14.1. Lựa chọn phương pháp**
- **User mới**: Popularity-based
- **Item mới**: Content-based
- **Có đủ ratings**: Collaborative Filtering
- **Large dataset**: Matrix Factorization
- **Cold-start**: Hybrid (Content + CF)

### **14.2. Hyperparameters**
- **k (neighbors)**: 20-30 cho CF
- **K (latent factors)**: 2-20 cho MF
- **λ (regularization)**: 0.01-0.1
- **learning_rate**: 0.5-2.0
- **threshold (vote_count)**: 70th-90th percentile

### **14.3. Performance**
- **User-based CF**: Tốt khi #users nhỏ
- **Item-based CF**: Tốt khi #items nhỏ hơn #users
- **MF**: Tốt cho large sparse matrix
- **Content-based**: Không phụ thuộc ratings

### **14.4. Common Pitfalls**
❌ Quên normalize data → bias  
❌ Không xử lý missing values → error  
❌ Overfitting → regularization quá thấp  
❌ Cold-start không handle → poor UX  

---

## 📚 XV. TÀI LIỆU THAM KHẢO

### **Datasets**
- MovieLens 100K: https://grouplens.org/datasets/movielens/100k/
- MovieLens 1M: https://grouplens.org/datasets/movielens/1m/

### **Concepts**
- Weighted Rating (IMDB formula)
- Pearson Correlation vs Cosine Similarity
- Adjusted Cosine Similarity
- SVD for Recommendation Systems
- Matrix Factorization with SGD

### **Libraries Documentation**
- Scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/
- Scipy: https://scipy.org/

---

## 🔥 XVI. BÀI TẬP MỞ RỘNG

### **Advanced Topics (không trong lab nhưng liên quan)**
1. **Deep Learning Recommenders**: Neural Collaborative Filtering
2. **Hybrid Systems**: Kết hợp CF + Content-based
3. **Contextual Recommendations**: Thêm time, location
4. **Implicit Feedback**: Click, view thay vì rating
5. **Diversity & Serendipity**: Không chỉ accuracy
6. **Evaluation Metrics**: Precision@K, Recall@K, MAP, NDCG

---

**📌 Lưu ý**: File này tổng hợp TẤT CẢ các dạng bài từ Lab 01 đến Lab 05. Sinh viên nên:
- ✅ Làm lần lượt từng lab để hiểu concepts
- ✅ So sánh các phương pháp trên cùng dataset
- ✅ Thử thay đổi hyperparameters
- ✅ Đọc code mẫu và tự implement lại
- ✅ Vẽ sơ đồ để hiểu workflow

**Good luck! 🚀**
