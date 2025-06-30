import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# Paths for saving models
USER_MODEL_PATH = "user_similarity_model.pkl"
ITEM_MODEL_PATH = "item_similarity_model.pkl"
USER_MATRIX_PATH = "user_matrix.pkl"
ITEM_MATRIX_PATH = "item_matrix.pkl"
USER_MAP_PATH = "user_map.pkl"
ITEM_MAP_PATH = "item_map.pkl"

# Rating constraints
MIN_RATING = 1.0
MAX_RATING = 5.0

def clip_rating(rating):
    """Clip rating to valid range"""
    return np.clip(rating, MIN_RATING, MAX_RATING)

# Function to load data
@st.cache_data
def load_data():
    data_paths = {
        'user_based': {
            'train': './data/collabrative/user_based_training_set.csv',
            'test': './data/collabrative/user_based_testing_set.csv'
        },
        'item_based': {
            'train': './data/collabrative/item_based_training_set.csv',
            'test': './data/collabrative/item_based_testing_set.csv'
        }
    }
    
    try:
        user_train = pd.read_csv(data_paths['user_based']['train'])
        user_test = pd.read_csv(data_paths['user_based']['test'])
        item_train = pd.read_csv(data_paths['item_based']['train'])
        item_test = pd.read_csv(data_paths['item_based']['test'])
        return user_train, user_test, item_train, item_test
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None, None, None, None

# Function to create user-item matrix
def create_user_item_matrix(df):
    user_ids = df['username'].unique()
    item_ids = df['id_product'].unique()
    user_map = {user: idx for idx, user in enumerate(user_ids)}
    item_map = {item: idx for idx, item in enumerate(item_ids)}
    
    rows = df['username'].map(user_map)
    cols = df['id_product'].map(item_map)
    ratings = df['rating'].values
    matrix = csr_matrix((ratings, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    
    return matrix, user_map, item_map

# Function to train and save models
def train_and_save_models(user_train, item_train):
    try:
        # User-based model
        user_matrix, user_map, item_map = create_user_item_matrix(user_train)
        user_sim = cosine_similarity(user_matrix)
        joblib.dump(user_sim, USER_MODEL_PATH)
        joblib.dump(user_matrix, USER_MATRIX_PATH)
        joblib.dump(user_map, USER_MAP_PATH)
        joblib.dump(item_map, ITEM_MAP_PATH)
        
        # Item-based model
        item_matrix, _, _ = create_user_item_matrix(item_train)
        item_sim = cosine_similarity(item_matrix.T)
        joblib.dump(item_sim, ITEM_MODEL_PATH)
        joblib.dump(item_matrix, ITEM_MATRIX_PATH)
        
        return True
    except Exception as e:
        st.error(f"Error training models: {e}")
        return False

# Function to load models
@st.cache_resource
def load_models():
    if (os.path.exists(USER_MODEL_PATH) and os.path.exists(ITEM_MODEL_PATH) and 
        os.path.exists(USER_MATRIX_PATH) and os.path.exists(ITEM_MATRIX_PATH) and
        os.path.exists(USER_MAP_PATH) and os.path.exists(ITEM_MAP_PATH)):
        try:
            user_sim = joblib.load(USER_MODEL_PATH)
            item_sim = joblib.load(ITEM_MODEL_PATH)
            user_matrix = joblib.load(USER_MATRIX_PATH)
            item_matrix = joblib.load(ITEM_MATRIX_PATH)
            user_map = joblib.load(USER_MAP_PATH)
            item_map = joblib.load(ITEM_MAP_PATH)
            return user_sim, item_sim, user_matrix, item_matrix, user_map, item_map
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None, None, None, None, None, None
    return None, None, None, None, None, None

# User-based CF prediction with improved algorithm
def user_based_cf(user_sim, user_matrix, user_map, item_map, username, k=5):
    if username not in user_map:
        return []
    
    user_idx = user_map[username]
    # Get user's mean rating for bias correction
    user_ratings = user_matrix[user_idx].toarray().flatten()
    user_mean = np.mean(user_ratings[user_ratings > 0]) if np.sum(user_ratings > 0) > 0 else 0
    
    # Find top k similar users (excluding self)
    similar_users = np.argsort(user_sim[user_idx])[::-1][1:k+1]
    
    predictions = {}
    for item_idx in range(user_matrix.shape[1]):
        # Skip items already rated by the user
        if user_matrix[user_idx, item_idx] > 0:
            continue
            
        # Find similar users who have rated this item
        rated_users = user_matrix[:, item_idx].nonzero()[0]
        valid_users = np.intersect1d(similar_users, rated_users)
        
        if len(valid_users) > 0:
            # Get similarities and ratings
            sim_scores = user_sim[user_idx][valid_users]
            ratings = user_matrix[valid_users, item_idx].toarray().flatten()
            
            # Filter out zero similarities
            valid_mask = sim_scores > 0
            if np.sum(valid_mask) > 0:
                sim_scores = sim_scores[valid_mask]
                ratings = ratings[valid_mask]
                
                # Calculate weighted average with bias correction
                numerator = np.sum(sim_scores * ratings)
                denominator = np.sum(sim_scores)
                
                if denominator > 0:
                    pred_rating = numerator / denominator
                    # Clip to valid rating range
                    pred_rating = clip_rating(pred_rating)
                    predictions[item_idx] = pred_rating
    
    return sorted(predictions.items(), key=lambda x: x[1], reverse=True)

# Item-based CF prediction with improved algorithm
def item_based_cf(item_sim, item_matrix, user_map, item_map, username, k=5):
    if username not in user_map:
        return []
    
    user_idx = user_map[username]
    rated_items = item_matrix[user_idx].nonzero()[1]
    
    if len(rated_items) == 0:
        return []
    
    predictions = {}
    for item_idx in range(item_matrix.shape[1]):
        # Skip items already rated by the user
        if item_idx in rated_items:
            continue
        
        # Get similarities with rated items
        sim_scores = item_sim[item_idx, rated_items]
        ratings = item_matrix[user_idx, rated_items].toarray().flatten()
        
        # Filter positive similarities and valid ratings
        valid_mask = (sim_scores > 0) & (ratings > 0)
        
        if np.sum(valid_mask) > 0:
            valid_sims = sim_scores[valid_mask]
            valid_ratings = ratings[valid_mask]
            
            # Take top k most similar items
            if len(valid_sims) > k:
                top_indices = np.argsort(valid_sims)[-k:]
                valid_sims = valid_sims[top_indices]
                valid_ratings = valid_ratings[top_indices]
            
            # Calculate weighted average
            numerator = np.sum(valid_sims * valid_ratings)
            denominator = np.sum(valid_sims)
            
            if denominator > 0:
                pred_rating = numerator / denominator
                # Clip to valid rating range
                pred_rating = clip_rating(pred_rating)
                predictions[item_idx] = pred_rating
    
    return sorted(predictions.items(), key=lambda x: x[1], reverse=True)

# Get single prediction for evaluation
def get_single_prediction(user_sim, item_sim, user_matrix, item_matrix, user_map, item_map, username, product_id, rec_type, k=5):
    if username not in user_map or product_id not in item_map:
        return None
    
    user_idx = user_map[username]
    item_idx = item_map[product_id]
    
    if rec_type == "User-Based":
        # Check if user has already rated this item
        if user_matrix[user_idx, item_idx] > 0:
            return None
        
        similar_users = np.argsort(user_sim[user_idx])[::-1][1:k+1]
        rated_users = user_matrix[:, item_idx].nonzero()[0]
        
        # Find similar users who have rated this item
        valid_users = np.intersect1d(similar_users, rated_users)
        if len(valid_users) == 0:
            return None
            
        sim_scores = user_sim[user_idx][valid_users]
        ratings = user_matrix[valid_users, item_idx].toarray().flatten()
        
        # Filter positive similarities
        valid_mask = sim_scores > 0
        if np.sum(valid_mask) > 0:
            sim_scores = sim_scores[valid_mask]
            ratings = ratings[valid_mask]
            
            if np.sum(sim_scores) > 0:
                pred_rating = np.sum(ratings * sim_scores) / np.sum(sim_scores)
                return clip_rating(pred_rating)
    
    else:  # Item-Based
        # Check if user has already rated this item
        if item_matrix[user_idx, item_idx] > 0:
            return None
            
        rated_items = item_matrix[user_idx].nonzero()[1]
        if len(rated_items) == 0:
            return None
            
        sim_scores = item_sim[item_idx, rated_items]
        ratings = item_matrix[user_idx, rated_items].toarray().flatten()
        
        valid_mask = (sim_scores > 0) & (ratings > 0)
        if np.sum(valid_mask) > 0:
            valid_sims = sim_scores[valid_mask]
            valid_ratings = ratings[valid_mask]
            
            pred_rating = np.sum(valid_ratings * valid_sims) / np.sum(valid_sims)
            return clip_rating(pred_rating)
    
    return None

# Evaluate metrics
def evaluate_metrics(actual, predicted):
    if len(actual) == 0 or len(predicted) == 0:
        return {'MAE': 0, 'MSE': 0, 'RMSE': 0, 'NMAE': 0}
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    nmae = mae / (actual.max() - actual.min()) if actual.max() != actual.min() else mae
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'NMAE': nmae}

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Product Recommendation System",
        page_icon="🛍️",
        layout="wide"
    )
    
    st.title("🛍️ Product Recommendation System")
    st.markdown("---")
    
    # Show rating range info
    st.info(f"📊 Rating Range: {MIN_RATING} - {MAX_RATING}")
    
    # Load data
    with st.spinner("Loading data..."):
        data_result = load_data()
        if data_result[0] is None:
            st.error("Failed to load data. Please check file paths.")
            return
        
        user_train, user_test, item_train, item_test = data_result
    
    # Display data info and rating statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("User Train", len(user_train))
    with col2:
        st.metric("User Test", len(user_test))
    with col3:
        st.metric("Item Train", len(item_train))
    with col4:
        st.metric("Item Test", len(item_test))
    
    # Show rating distribution
    st.markdown("### 📈 Rating Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Training Data:**")
        rating_dist = user_train['rating'].value_counts().sort_index()
        st.bar_chart(rating_dist)
    with col2:
        st.write("**Statistics:**")
        st.write(f"Min Rating: {user_train['rating'].min()}")
        st.write(f"Max Rating: {user_train['rating'].max()}")
        st.write(f"Mean Rating: {user_train['rating'].mean():.2f}")
        st.write(f"Std Rating: {user_train['rating'].std():.2f}")
    
    # Load or train models
    st.markdown("### 🤖 Model Status")
    user_sim, item_sim, user_matrix, item_matrix, user_map, item_map = load_models()
    
    if user_sim is None:
        st.warning("Models not found. Training new models...")
        with st.spinner("Training models for the first time..."):
            if train_and_save_models(user_train, item_train):
                st.success("Models trained and saved successfully!")
                user_sim, item_sim, user_matrix, item_matrix, user_map, item_map = load_models()
            else:
                st.error("Failed to train models.")
                return
    else:
        st.success("✅ Models loaded successfully!")
    
    # Product name mapping
    product_names = dict(zip(user_train['id_product'], user_train['name_product']))
    
    st.markdown("---")
    st.header("🎯 Get Recommendations")
    
    # User interface
    col1, col2 = st.columns(2)
    
    with col1:
        usernames = sorted(user_train['username'].unique())
        selected_user = st.selectbox("📤 Select a User", usernames)
        n_recs = st.slider("📊 Number of Recommendations", 1, 20, 5)
    
    with col2:
        rec_type = st.radio("🔧 Recommendation Type", ["User-Based", "Item-Based"])
        k_neighbors = st.slider("👥 Number of Neighbors (K)", 3, 20, 5)
    
    if st.button("🎯 Generate Recommendations", type="primary", use_container_width=True):
        with st.spinner("Generating recommendations..."):
            if rec_type == "User-Based":
                recommendations = user_based_cf(user_sim, user_matrix, user_map, item_map, selected_user, k=k_neighbors)
            else:
                recommendations = item_based_cf(item_sim, item_matrix, user_map, item_map, selected_user, k=k_neighbors)
        
        st.subheader(f"🏆 Top {n_recs} Recommended Products for **{selected_user}**")
        
        if recommendations:
            # Create recommendation cards
            for i, (item_idx, rating) in enumerate(recommendations[:n_recs]):
                # Get product info
                product_id = list(item_map.keys())[list(item_map.values()).index(item_idx)]
                product_name = product_names.get(product_id, "Unknown Product")
                
                # Create card
                with st.container():
                    col1, col2, col3 = st.columns([1, 4, 2])
                    
                    with col1:
                        st.markdown(f"### #{i+1}")
                    
                    with col2:
                        st.markdown(f"**{product_name}**")
                        st.markdown(f"*Product ID: {product_id}*")
                    
                    with col3:
                        # Color code the rating
                        if rating >= 4.0:
                            color = "🟢"
                        elif rating >= 3.0:
                            color = "🟡"
                        else:
                            color = "🔴"
                        st.metric("Predicted Rating", f"{color} {rating:.2f}")
                    
                    st.markdown("---")
        else:
            st.warning("⚠️ No recommendations available for this user.")
        
        # Evaluation section
        st.markdown("---")
      
    # Model management
    st.markdown("---")
    st.subheader("⚙️ Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Retrain Models", help="This will retrain and overwrite existing models"):
            with st.spinner("Retraining models..."):
                if train_and_save_models(user_train, item_train):
                    st.success("✅ Models retrained and saved successfully!")
                    st.rerun()  # Refresh the app to load new models
                else:
                    st.error("❌ Failed to retrain models.")
    
    with col2:
        # Show model file info
        if os.path.exists(USER_MODEL_PATH):
            file_size = os.path.getsize(USER_MODEL_PATH) / (1024 * 1024)  # MB
            st.info(f"📁 Model size: {file_size:.2f} MB")
        else:
            st.info("📁 No models found")

if __name__ == "__main__":
    main()