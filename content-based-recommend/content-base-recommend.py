import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import re
import numpy as np
from underthesea import word_tokenize

# Transformer tùy chỉnh cho từng bước xử lý văn bản
class LowercaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [x.lower() for x in X] if isinstance(X, list) else X.lower()

class RemoveStopwordsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        stopwords = st.session_state.get('stopwords', [])
        if isinstance(X, list):
            return [' '.join(word for word in x.split() if word not in stopwords) for x in X]
        else:
            return ' '.join(word for word in X.split() if word not in stopwords)

class WordTokenizeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [word_tokenize(x, format="text") for x in X] if isinstance(X, list) else word_tokenize(X, format="text")

class RemoveSpecialCharsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        import re
        if isinstance(X, list):
            return [re.sub(r'\W+', ' ', x) for x in X]
        else:
            return re.sub(r'\W+', ' ', X)

# Tải stopwords
@st.cache_data
def get_stopwords_list(stop_file_path):
    try:
        with open(stop_file_path, 'r', encoding="utf-8") as f:
            stopwords = frozenset(m.strip() for m in f.readlines())
            return list(stopwords)
    except FileNotFoundError:
        st.error("File stopwords không tồn tại.")
        return []

# Pipeline tiền xử lý văn bản
text_preprocessing_pipeline = Pipeline([
    ('lowercase', LowercaseTransformer()),
    ('remove_special_chars', RemoveSpecialCharsTransformer()),
    ('word_tokenize', WordTokenizeTransformer()),
    ('remove_stopwords', RemoveStopwordsTransformer())
])

# Xử lý mô tả không đầy đủ
@st.cache_data
def preprocess_description(products_df, stopwords):
    products_df = products_df.copy()
    if 'description' not in products_df.columns:
        st.warning("Cột 'description' không tồn tại, tạo từ 'name_product' và 'category'.")
        products_df['description'] = products_df['name_product'] + ' ' + products_df['category']
    
    def enhance_description(row):
        desc = row['description']
        if pd.isna(desc) or desc.strip().lower() in ['không có mô tả', 'xem thêm'] or len(desc.split()) < 5:
            desc = f"{row['name_product']} {row['category']}"
        return desc
    
    products_df['description'] = products_df.apply(enhance_description, axis=1)
    st.session_state['stopwords'] = stopwords
    
    processed_descriptions = text_preprocessing_pipeline.transform(products_df['description'].tolist())
    products_df['processed_description'] = processed_descriptions
    
    products_df = products_df[products_df['processed_description'].str.strip() != '']
    
    if products_df.index.duplicated().any():
        st.warning("Duplicate id_product found in products_df. Keeping first occurrence.")
        products_df = products_df[~products_df.index.duplicated(keep='first')]
    products_df.set_index('id_product', inplace=True)
    return products_df

# Tạo ma trận TF-IDF
@st.cache_data
def create_tfidf_matrix(processed_texts, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    return vectorizer, tfidf_matrix

# Tải và tiền xử lý dữ liệu
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
def get_content_based_recommendations(username, products_df, ratings_df, num_recommendations=5, rating_threshold=3, description_matrix=None):
    user_ratings = ratings_df[(ratings_df['username'] == username) & (ratings_df['rating'] >= rating_threshold)]
    if user_ratings.empty:
        st.warning(f"Không tìm thấy đánh giá cho người dùng {username} với rating >= {rating_threshold}.")
        return [], []
    
    user_product_ids = set(user_ratings['id_product'])
    user_product_indices = [products_df.index.get_loc(pid) for pid in user_product_ids if pid in products_df.index]
    if not user_product_indices:
        st.warning(f"Không tìm thấy sản phẩm nào của người dùng {username} trong products_df.")
        return [], []
    
    rated_products = products_df[products_df.index.isin(user_product_ids)][['name_product', 'description', 'category', 'img']].reset_index().to_dict('records')
    
    user_profile = np.asarray(description_matrix[user_product_indices].mean(axis=0))
    cosine_sim = cosine_similarity(user_profile, description_matrix)[0]
    
    sim_df = pd.DataFrame({
        'id_product': products_df.index,
        'cosine_sim': cosine_sim
    })
    sim_df = sim_df[~sim_df['id_product'].isin(user_product_ids)]
    sim_df = sim_df.merge(products_df[['category']], left_on='id_product', right_index=True)
    sim_df = sim_df.sort_values('cosine_sim', ascending=False)
    sim_df = sim_df.groupby('category').head(3).head(num_recommendations)
    
    recommendations = products_df.loc[sim_df['id_product']][['name_product', 'description', 'category', 'img']].reset_index().to_dict('records')
    return recommendations, rated_products

# Tính relevance dựa trên độ tương đồng
def calculate_relevance_similarity(recommendations, products_df, ratings_df, username, rating_threshold, description_matrix, test_df):
    user_ratings = ratings_df[(ratings_df['username'] == username) & (ratings_df['rating'] >= rating_threshold)]
    if user_ratings.empty:
        return {product['id_product']: 0.0 for product in recommendations}
    
    user_product_ids = set(user_ratings['id_product'])
    user_product_indices = [products_df.index.get_loc(pid) for pid in user_product_ids if pid in products_df.index]
    if not user_product_indices:
        return {product['id_product']: 0.0 for product in recommendations}
    
    test_user_ratings = test_df[(test_df['username'] == username) & (test_df['rating'] >= rating_threshold)]
    test_product_ids = set(test_user_ratings['id_product'])
    all_user_product_ids = user_product_ids.union(test_product_ids)
    all_user_product_indices = [products_df.index.get_loc(pid) for pid in all_user_product_ids if pid in products_df.index]
    
    user_profile = np.asarray(description_matrix[all_user_product_indices].mean(axis=0))
    
    recommended_indices = [products_df.index.get_loc(product['id_product']) 
                         for product in recommendations if product['id_product'] in products_df.index]
    
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
    recommended_indices = [products_df.index.get_loc(product['id_product']) 
                         for product in recommendations if product['id_product'] in products_df.index]
    
    if not recommended_indices:
        return 0.0
    
    similarities = cosine_similarity(description_matrix[recommended_indices], user_profile).flatten()
    relevance_scores = calculate_relevance_similarity(recommendations, products_df, ratings_df, username, rating_threshold, description_matrix, test_df)
    
    serendipity_sum = sum((1 - similarity) * relevance_scores.get(products_df.index[idx], 0.0) 
                         for idx, similarity in zip(recommended_indices, similarities))
    
    return serendipity_sum / len(recommended_indices) if recommended_indices else 0.0

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
    
    st.write("### Gợi ý sản phẩm")
    username = st.selectbox("Chọn người dùng:", train_df['username'].unique().tolist())
    num_recommendations = st.slider("Số lượng sản phẩm gợi ý:", 1, 20, 5)
    rating_threshold = st.slider("Ngưỡng rating:", 1, 5, 3)
    
    if st.button("Gợi ý"):
        recommendations, rated_products = get_content_based_recommendations(
            username, products_df, train_df, num_recommendations, rating_threshold, description_matrix
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
            serendipity_score = calculate_serendipity(
                recommendations, products_df, train_df, username, rating_threshold, description_matrix, test_df
            )
            
            st.write(f"### Kết quả đánh giá")
            st.write(f"- **Serendipity**: {serendipity_score:.4f}")
            if serendipity_score >= 0.7:
                st.success("Serendipity cao: Danh sách khuyến nghị rất bất ngờ và phù hợp!")
            elif serendipity_score >= 0.4:
                st.info("Serendipity trung bình: Danh sách khuyến nghị có sự bất ngờ vừa phải và phù hợp.")
            else:
                st.warning("Serendipity thấp: Danh sách khuyến nghị thiếu bất ngờ hoặc không đủ phù hợp.")
            
            st.write(f"### Sản phẩm được gợi ý (Top {num_recommendations}):")
            for product in recommendations:
                st.write(f"**{product['name_product']}**")
                st.write(f"**Mô tả**: {product['description']}")
                st.write(f"**Danh mục**: {product['category']}")
                st.image(product['img'], width=200)
                st.write("---")
        else:
            st.warning(f"Không tìm thấy gợi ý cho người dùng {username}.")

if __name__ == "__main__":
    main()