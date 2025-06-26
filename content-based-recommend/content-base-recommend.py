import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Tải dữ liệu stopwords tiếng Việt
def get_stopwords_list(stop_file_path):
    """Load stopwords from a file."""
    try:
        with open(stop_file_path, 'r', encoding="utf-8") as f:
            stopwords = f.readlines()
            stop_set = set(m.strip() for m in stopwords)
            return list(frozenset(stop_set))
    except FileNotFoundError:
        st.error("File stopwords không tồn tại. Vui lòng cung cấp file vietnamese.txt.")
        return []

# Tiền xử lý văn bản
def preprocess_text(data, stopwords):
    """Preprocess text data: lowercase, remove special characters, remove stopwords."""
    data = data.lower()
    data = re.sub(r'\W+', ' ', data)
    data = ' '.join([word for word in data.split() if word not in stopwords])
    return data

# Hàm gợi ý sản phẩm tương tự dựa trên một sản phẩm
def get_content_based_recommendations_by_product(product_id, products_df, num_recommendations=5, stopwords=[]):
    """Get content-based recommendations based on a specific product."""
    products_df['processed_description'] = products_df['description'].apply(
        lambda x: preprocess_text(x, stopwords)
    )
    
    vectorizer = TfidfVectorizer(max_features=4500)
    description_matrix = vectorizer.fit_transform(products_df['processed_description'])
    
    # Lấy index của sản phẩm
    product_index = products_df[products_df['id_product'] == product_id].index
    if len(product_index) == 0:
        return []
    
    product_index = product_index[0]
    
    # Tính độ tương đồng cosine
    cosine_sim = cosine_similarity(description_matrix[product_index], description_matrix)[0]
    
    # Sắp xếp sản phẩm
    sim_sp = list(enumerate(cosine_sim))
    sim_sp = sorted(sim_sp, key=lambda x: x[1], reverse=True)
    sim_sp = sim_sp[1:num_recommendations+1]  # Bỏ sản phẩm gốc
    product_indices = [i[0] for i in sim_sp]
    
    return products_df.iloc[product_indices][['id_product', 'name_product', 'description', 'category', 'img']].to_dict('records')

# Ứng dụng Streamlit
def main():
    st.title("Hệ thống khuyến nghị sản phẩm")
    st.subheader("Content-Based Filtering (Dựa trên sản phẩm)")
    
    # Tải dữ liệu
    try:
        products_df = pd.read_csv('../final_products.csv')
    except FileNotFoundError:
        st.error("Không tìm thấy file final_products.csv.")
        return
    
    stopwords = get_stopwords_list('../vietnamese-stopwords.txt')
    
    # Giao diện người dùng
    product_id = st.selectbox("Chọn sản phẩm:", products_df['name_product'].tolist())
    num_recommendations = st.slider("Số lượng sản phẩm gợi ý:", 1, 10, 5)
    
    if st.button("Gợi ý sản phẩm"):
        product_id_selected = products_df[products_df['name_product'] == product_id]['id_product'].iloc[0]
        recommendations = get_content_based_recommendations_by_product(
            product_id_selected, products_df, num_recommendations, stopwords
        )
        
        if recommendations:
            st.write("### Sản phẩm được gợi ý:")
            for product in recommendations:
                st.write(f"**{product['name_product']}**")
                st.write(f"**Mô tả**: {product['description']}")
                st.write(f"**Danh mục**: {product['category']}")
                st.image(product['img'], width=200)
                st.write("---")
        else:
            st.warning("Không tìm thấy gợi ý. Vui lòng kiểm tra sản phẩm.")

if __name__ == "__main__":
    main()