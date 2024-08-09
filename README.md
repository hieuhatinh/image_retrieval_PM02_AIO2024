# Truy vấn hình ảnh dùng các toán cơ bản

## 1. Giới thiệu bài toán 
- **Truy vấn hình ảnh (Images Retrieval)** là một bài toán thuộc lĩnh vực Truy vấn thông tin ***(Information Retrieval)***. Trong đó, nhiệm vụ của ta là xây dựng một chương trình trả về các hình ảnh(Images) có liên quan đến hình ảnh truy vấn đầu vào(Query) và các hình ảnh được lấy từ một bộ dữ liệu hình ảnh cho trước, hiện nay có một số ứng dụng truy vấn ảnh như: Google Search Image, chức năng tìm kiếm sản phẩm bằng hình ảnh trên Shopee, Lazada, Tiki, ...
- Project này xây dựng một chương trình khi đưa vào 1 hình ảnh, chương trình sẽ đưa ra các hình ảnh tương tự
    
    - **Input**: Hình ảnh truy vấn Query Image và bộ dữ liệu Images Library
    - **Output**: Danh sách hình ảnh có sự tương tự đến hình ảnh truy vấn

## 2. Tải bộ dữ liệu
- Chạy lệnh sau để tải bộ dữ liệu Images Library:
```
    !gdown 1msLVo0g0LFmL9-qZ73vq9YEVZwbzOePF
    !unzip /content/data.zip
```

## 3. Xây dựng chương trình:
### 3.1. Chương trình truy vấn ảnh cơ bản
#### 3.1.1. Truy vấn hình ảnh sử dụng độ đo L1
- Xây dựng hàm absolute_difference() để tính độ đo L1:

    $L1(\vec{a}, \vec{b}) = \sum_{i=1}^{N} \left| a - b \right|$

- Tạo hàm get_l1_score(), hàm này sẽ trả về ảnh query và ls_path_score chứa danh sách hình ảnh và giá trị độ tương đồng với từng ảnh

#### 3.1.2. Truy vấn hình ảnh sử dụng độ đo L2
- Xây dựng hàm mean_square_difference() để tính độ đo L1:

    $L2(\vec{a}, \vec{b}) = \sqrt{\sum_{i=1}^{N} (a_i - b_i)^2}$

- Tạo hàm get_l2_score(), tương tự với get_l1_score()

#### 3.1.3 Truy vấn hình ảnh với độ đo Cosine Similarity
- Xây dựng hàm cosine_similarity() để tính độ đo L1:

    $cosine\_similarity(\vec{a}, \vec{b}) = \frac{a * b}{\left\|a \right\| \left\|b \right\|} = \frac{\sum_{i=1}^{N} a_i b_i}{\sqrt{\sum_{i=1}^{N} a_i^2} \sqrt{\sum_{i=1}^{N} b_i^2}}$

- Tạo hàm get_cosine_similarity_score(), tương tự với get_l1_score()

#### 3.1.4 Truy vấn hình ảnh với độ đo Correlation Coefficient
- Xây dựng hàm correlation_coefficient() để tính độ đo L1:

    $r = \frac{E[(X - \mu_X)(Y - \mu_Y)]}{\sigma_X \sigma_Y} = \frac{\sum (x_i - \mu_X)(y_i - \mu_Y)}{\sqrt{\sum(x_i - \mu_X)^2 \sum(y_i - \mu_Y)^2}}$

- Tạo hàm get_correlation_coefficient_score(), tương tự với get_l1_score()

### 3.2. Chương trình truy vấn ảnh nâng cao:
#### 3.2.1 Truy vấn ảnh với pretrained deep learning model
- Thư viện chromadb hỗ trợ việc quản lý và truy xuất dữ liệu hình ảnh hiệu quả (chúng ta cũng sử dụng thêm với mục đích tạo vector database), và chromadb có thể dùng open-clip-torch để cung cấp khả năng sử dụng mô hình CLIP đã được đào tạo sẵn, đây là một công cụ mạnh mẽ để phân tích nội dung hình ảnh thông qua học sâu
```
    !pip install chromadb
    !pip install open-clip-torch
```

- Thêm 1 hàm trích xuất vector đặc trưng cho mỗi hình ảnh. Mô hình CLIP sẽ được sử dụng để biến đổi hình ảnh thành các vector đặc trưng đại diện cho nội dung và ngữ cảnh của hình ảnh đó. Việc so sánh các hình ảnh sẽ thông qua so sánh độ tương đồng giữa các vector mà không sử dụng trực tiếp ảnh gốc
```
    embedding_function = OpenCLIPEmbeddingFunction()
        
    def get_single_image_embedding(image):
        embedding = embedding_function._encode_image(image=image)
        return np.array(embedding)
```