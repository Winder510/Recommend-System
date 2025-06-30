import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from underthesea import word_tokenize

# Tải stopwords
def get_stopwords_list(stop_file_path):
    try:
        with open(stop_file_path, 'r', encoding="utf-8") as f:
            stopwords = f.readlines()
            stop_set = set(m.strip() for m in stopwords)
            return list(frozenset(stop_set))
    except FileNotFoundError:
        st.error("File stopwords không tồn tại.")
        return []

# Tiền xử lý văn bản
def preprocess_text(data, stopwords):
    data = data.lower()
    data = re.sub(r'\W+', ' ', data)
    data = word_tokenize(data, format="text")
    data = ' '.join([word for word in data.split() if word not in stopwords])
    return data

# Xử lý mô tả không đầy đủ
def preprocess_description(products_df, stopwords):
    products_df = products_df.copy()
    if 'description' not in products_df.columns:
        st.warning("Cột 'description' không tồn tại, tạo từ 'name_product' và 'category'.")
        products_df['description'] = products_df['name_product'] + ' ' + products_df['category']
    
    def enhance_description(row):
        desc = row['description']
        if pd.isna(desc) or desc.strip().lower() in ['không có mô tả', 'xem thêm'] or len(desc.split()) < 5:
            desc = f"{row['name_product']} {row['category']}"
        return preprocess_text(desc, stopwords)
    
    products_df['processed_description'] = products_df.apply(enhance_description, axis=1)
    products_df = products_df[products_df['processed_description'].str.strip() != '']
    if products_df.index.duplicated().any():
        st.warning("Duplicate id_product found in products_df. Keeping first occurrence.")
        products_df = products_df[~products_df.index.duplicated(keep='first')]
    products_df.set_index('id_product', inplace=True)
    return products_df

# Tạo ma trận TF-IDF
def create_tfidf_matrix(processed_texts, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    return vectorizer, tfidf_matrix

# Tải và tiền xử lý dữ liệu với cache
@st.cache_data
def load_and_preprocess_data(products_path, train_path, test_path, stopwords_path):
    try:
        products_df = pd.read_csv(products_path)
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError:
        st.error("Không tìm thấy file dữ liệu.")
        return None, None, None, None, None
    
    stopwords = get_stopwords_list(stopwords_path)
    products_df = preprocess_description(products_df, stopwords)
    if products_df.empty:
        st.error("Không có sản phẩm nào sau khi tiền xử lý.")
        return None, None, None, None, None
    
    vectorizer, description_matrix = create_tfidf_matrix(products_df['processed_description'])
    if description_matrix is None:
        st.error("Không thể tạo ma trận TF-IDF.")
        return None, None, None, None, None
    
    return products_df, train_df, test_df, stopwords, (vectorizer, description_matrix)

# Hàm gợi ý sản phẩm
def get_content_based_recommendations(username, products_df, ratings_df, num_recommendations=5, stopwords=None, rating_threshold=3, description_matrix=None):
    user_ratings = ratings_df[(ratings_df['username'] == username) & (ratings_df['rating'] >= rating_threshold)]
    if user_ratings.empty:
        st.warning(f"Không tìm thấy đánh giá cho người dùng {username} với rating >= {rating_threshold}.")
        return [], []
    
    user_product_ids = set(user_ratings['id_product'])
    user_product_indices = [products_df.index.get_loc(pid) for pid in user_product_ids if pid in products_df.index]
    if not user_product_indices:
        st.warning(f"Không tìm thấy sản phẩm nào của người dùng {username} trong products_df.")
        return [], []
    
    # Lấy thông tin sản phẩm đã đánh giá
    rated_products = products_df[products_df.index.isin(user_product_ids)][['name_product', 'description', 'category', 'img']].reset_index().to_dict('records')
    
    user_profile = np.asarray(description_matrix[user_product_indices].mean(axis=0))
    cosine_sim = cosine_similarity(user_profile, description_matrix)[0]
    
    rated_product_ids = user_product_ids
    product_indices = []
    category_count = {}
    max_per_category = 3
    
    for idx, score in sorted(enumerate(cosine_sim), key=lambda x: x[1], reverse=True):
        product_id = products_df.index[idx]
        if product_id not in rated_product_ids:
            try:
                category = products_df.loc[product_id, 'category']
                if not isinstance(category, str):
                    st.warning(f"Invalid category for product_id {product_id}: {category}")
                    continue
                if category not in category_count:
                    category_count[category] = 0
                if category_count[category] < max_per_category:
                    product_indices.append(idx)
                    category_count[category] += 1
                if len(product_indices) >= num_recommendations:
                    break
            except KeyError:
                st.warning(f"Product ID {product_id} not found in products_df.")
                continue
    
    if not product_indices:
        st.warning(f"Không có sản phẩm gợi ý hợp lệ cho người dùng {username}.")
        return [], rated_products
    
    recommendations = products_df.iloc[product_indices][['name_product', 'description', 'category', 'img']].reset_index().to_dict('records')
    return recommendations, rated_products

# Tính relevance dựa trên độ tương đồng
def calculate_relevance_similarity(recommendations, products_df, ratings_df, username, rating_threshold, description_matrix, test_df):
    user_ratings = ratings_df[(ratings_df['username'] == username) & (ratings_df['rating'] >= rating_threshold)]
    if user_ratings.empty:
        return {product['id_product']: 0.0 for product in recommendations}
    
    user_product_ids = set(user_ratings['id_product'])
    user_product_indices = [products_df.index.get_loc(pid) for pid in user_product_ids if pid in products_df.index]
    if not user_product_indices:
        return {product['id_product']: 0.0 for product in inspirations}
    
    test_user_ratings = test_df[(test_df['username'] == username) & (test_df['rating'] >= rating_threshold)]
    test_product_ids = set(test_user_ratings['id_product'])
    all_user_product_ids = user_product_ids.union(test_product_ids)
    all_user_product_indices = [products_df.index.get_loc(pid) for pid in all_user_product_ids if pid in products_df.index]
    
    user_profile = np.asarray(description_matrix[all_user_product_indices].mean(axis=0))
    
    recommended_indices = []
    for product in recommendations:
        product_id = product['id_product']
        if product_id in products_df.index:
            recommended_indices.append(products_df.index.get_loc(product_id))
        else:
            st.warning(f"Product ID {product_id} not found in products_df.")
    
    if not recommended_indices:
        return {product['id_product']: 0.0 for product in recommendations}
    
    similarities = cosine_similarity(description_matrix[recommended_indices], user_profile).flatten()
    
    relevance_scores = {}
    sim_threshold_low = 0.2
    sim_threshold_high = 0.8
    
    for product, similarity in zip(recommendations, similarities):
        product_id = product['id_product']
        relevance_scores[product_id] = 1.0 if sim_threshold_low <= similarity <= sim_threshold_high else 0.0
    
    return relevance_scores

# Tính Serendipity
def calculate_serendipity(recommendations, products_df, ratings_df, username, rating_threshold, description_matrix, test_df):
    if not recommendations:
        return 0.0
    
    user_ratings = ratings_df[(ratings_df['username'] == username) & (ratings_df['rating'] >= rating_threshold)]
    if user_ratings.empty:
        return 0.0
    
    user_product_ids = set(user_ratings['id_product'])
    user_product_indices = [products_df.index.get_loc(pid) for pid in user_product_ids if pid in products_df.index]
    if not user_product_indices:
        return 0.0
    
    test_user_ratings = test_df[(test_df['username'] == username) & (test_df['rating'] >= rating_threshold)]
    test_product_ids = set(test_user_ratings['id_product'])
    all_user_product_ids = user_product_ids.union(test_product_ids)
    all_user_product_indices = [products_df.index.get_loc(pid) for pid in all_user_product_ids if pid in products_df.index]
    
    user_profile = np.asarray(description_matrix[all_user_product_indices].mean(axis=0))
    recommended_indices = []
    for product in recommendations:
        product_id = product['id_product']
        if product_id in products_df.index:
            recommended_indices.append(products_df.index.get_loc(product_id))
        else:
            st.warning(f"Product ID {product_id} not found in products_df.")
    
    if not recommended_indices:
        return 0.0
    
    similarities = cosine_similarity(description_matrix[recommended_indices], user_profile).flatten()
    relevance_scores = calculate_relevance_similarity(recommendations, products_df, ratings_df, username, rating_threshold, description_matrix, test_df)
    
    serendipity_sum = 0
    for idx, similarity in zip(recommended_indices, similarities):
        unexpectedness = 1 - similarity
        product_id = products_df.index[idx]
        relevance = relevance_scores.get(product_id, 0.0)
        serendipity_sum += unexpectedness * relevance
    
    serendipity = serendipity_sum / len(recommended_indices) if recommended_indices else 0.0
    return serendipity

# Ứng dụng Streamlit
def main():
    st.title("Hệ thống khuyến nghị sản phẩm cá nhân hóa")
    st.subheader("Content-Based Filtering")
    
    products_df, train_df, test_df, stopwords, tfidf_data = load_and_preprocess_data(
        '../data/product/products_cleaned.csv',
        '../data/train_data.csv',
        '../data/test_data.csv',
        '../vietnamese-stopwords.txt'
    )
    
    if products_df is None:
        return
    
    vectorizer, description_matrix = tfidf_data
    
    st.write(f"Số sản phẩm trong products_df: {len(products_df)}")
    st.write(f"Số đánh giá trong train_df: {len(train_df)}")
    st.write(f"Số đánh giá trong test_df: {len(test_df)}")
    st.write(f"Các danh mục trong products_df: {products_df['category'].unique().tolist()}")
    
    common_ids = set(train_df['id_product']).intersection(set(products_df.index))
    if not common_ids:
        st.error("Không có id_product chung giữa train_df và products_df.")
        return
    
    st.write(f"Số id_product chung: {len(common_ids)}/{len(products_df)}")
    st.write(f"Số người dùng trong train_df: {train_df['username'].nunique()}")
    
    st.write("### Gợi ý sản phẩm")
    username = st.selectbox("Chọn người dùng:", train_df['username'].unique().tolist())
    num_recommendations = st.slider("Số lượng sản phẩm gợi ý:", 1, 20, 5)
    rating_threshold = st.slider("Ngưỡng rating:", 1, 5, 3)
    
    if st.button("Gợi ý"):
        recommendations, rated_products = get_content_based_recommendations(
            username, products_df, train_df, num_recommendations, stopwords, rating_threshold, description_matrix
        )
        
        if rated_products:
            st.write(f"### Sản phẩm đã được {username} đánh giá (Rating >= {rating_threshold}):")
            for product in rated_products:
                st.write(f"**{product['name_product']}**")
                st.write(f"**Mô tả**: {product['description']}")
                st.write(f"**Danh mục**: {product['category']}")
                st.image(product['img'], width=200)
                st.write("---")
        
        if recommendations:
            st.write(f"### Sản phẩm được gợi ý (Top {num_recommendations}):")
            for product in recommendations:
                st.write(f"**{product['name_product']}**")
                st.write(f"**Mô tả**: {product['description']}")
                st.write(f"**Danh mục**: {product['category']}")
                st.image(product['img'], width=200)
                st.write("---")
            
            serendipity_score = calculate_serendipity(
                recommendations, products_df, train_df, username, rating_threshold, description_matrix, test_df
            )
            
            st.write("### Kết quả đánh giá")
            st.write(f"- **Số sản phẩm được khuyến nghị**: {len(recommendations)}")
            st.write(f"- **Serendipity**: {serendipity_score:.4f}")
            
            if serendipity_score >= 0.7:
                st.success("Serendipity cao: Danh sách khuyến nghị rất bất ngờ và phù hợp!")
            elif serendipity_score >= 0.4:
                st.info("Serendipity trung bình: Danh sách khuyến nghị có sự bất ngờ vừa phải và phù hợp.")
            else:
                st.warning("Serendipity thấp: Danh sách khuyến nghị thiếu bất ngờ hoặc không đủ phù hợp.")
        else:
            st.warning(f"Không tìm thấy gợi ý cho người dùng {username}.")

if __name__ == "__main__":
    main()