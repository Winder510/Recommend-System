import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ast
import warnings
warnings.filterwarnings('ignore')

# Enhanced data loading with adaptive thresholds
@st.cache_data
def load_and_process_data():
    """Load and process data with adaptive filtering"""
    try:
        # Load data
        products_df = pd.read_csv('data.csv')
        ratings_df = pd.read_csv('ratings.csv')
        
        # Basic info
        st.sidebar.write(f"**Original Data:**")
        st.sidebar.write(f"- Users: {ratings_df['user_id'].nunique()}")
        st.sidebar.write(f"- Products: {ratings_df['product_id'].nunique()}")
        st.sidebar.write(f"- Ratings: {len(ratings_df)}")
        
        # Merge data
        df = ratings_df.merge(products_df, on='product_id', how='left')
        
        return df, ratings_df, products_df
        
    except FileNotFoundError as e:
        st.error(f"Error: {str(e)}. Please ensure data.csv and ratings.csv are in the correct directory.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

@st.cache_data
def create_filtered_matrices(ratings_df, min_user_ratings=30, min_item_ratings=10):
    """Create filtered user-item matrices with adaptive thresholds"""
    
    # Start with original data
    original_users = ratings_df['user_id'].nunique()
    original_items = ratings_df['product_id'].nunique()
    original_ratings = len(ratings_df)
    
    # Apply filtering iteratively
    filtered_df = ratings_df.copy()
    
    for iteration in range(5):  # Maximum 5 iterations
        # Count ratings per user and item
        user_counts = filtered_df['user_id'].value_counts()
        item_counts = filtered_df['product_id'].value_counts()
        
        # Filter users and items
        valid_users = user_counts[user_counts >= min_user_ratings].index
        valid_items = item_counts[item_counts >= min_item_ratings].index
        
        # Apply filters
        new_filtered_df = filtered_df[
            (filtered_df['user_id'].isin(valid_users)) & 
            (filtered_df['product_id'].isin(valid_items))
        ]
        
        # Check if we have enough data
        if len(new_filtered_df) == 0:
            st.warning(f"Filtering too aggressive - no data remaining at thresholds: users≥{min_user_ratings}, items≥{min_item_ratings}")
            break
            
        # Check convergence
        if len(new_filtered_df) == len(filtered_df):
            break
            
        filtered_df = new_filtered_df
    
    # Show filtering results
    final_users = filtered_df['user_id'].nunique()
    final_items = filtered_df['product_id'].nunique()
    final_ratings = len(filtered_df)
    
    st.sidebar.write(f"**After filtering (≥{min_user_ratings} user ratings, ≥{min_item_ratings} item ratings):**")
    st.sidebar.write(f"- Users: {final_users} ({final_users/original_users*100:.1f}%)")
    st.sidebar.write(f"- Products: {final_items} ({final_items/original_items*100:.1f}%)")
    st.sidebar.write(f"- Ratings: {final_ratings} ({final_ratings/original_ratings*100:.1f}%)")
    
    if final_ratings == 0:
        st.error("No data remaining after filtering. Please reduce the minimum rating thresholds.")
        return None, None, None
    
    # Create matrices
    user_item_matrix = filtered_df.pivot_table(
        index='user_id', 
        columns='product_id', 
        values='rating', 
        fill_value=0
    )
    
    item_user_matrix = user_item_matrix.T
    
    # Calculate sparsity
    sparsity = 1 - (len(filtered_df) / (final_users * final_items))
    st.sidebar.write(f"- Sparsity: {sparsity*100:.1f}%")
    
    return user_item_matrix, item_user_matrix, filtered_df

def safe_cosine_similarity(matrix):
    """Safely calculate cosine similarity with error handling"""
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        st.error("Empty matrix - cannot calculate similarity")
        return None
    
    # Remove zero vectors
    non_zero_mask = (matrix != 0).any(axis=1)
    if not non_zero_mask.any():
        st.error("All vectors are zero - cannot calculate cosine similarity")
        return None
    
    filtered_matrix = matrix[non_zero_mask]
    
    try:
        similarity = cosine_similarity(filtered_matrix)
        
        # Create full similarity matrix with zeros for removed rows
        full_similarity = np.zeros((matrix.shape[0], matrix.shape[0]))
        valid_indices = np.where(non_zero_mask)[0]
        
        for i, idx_i in enumerate(valid_indices):
            for j, idx_j in enumerate(valid_indices):
                full_similarity[idx_i, idx_j] = similarity[i, j]
        
        return pd.DataFrame(full_similarity, index=matrix.index, columns=matrix.index)
        
    except Exception as e:
        st.error(f"Error calculating cosine similarity: {str(e)}")
        return None

def safe_pearson_similarity(matrix):
    """Safely calculate Pearson correlation with error handling"""
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        st.error("Empty matrix - cannot calculate similarity")
        return None
    
    n_users = matrix.shape[0]
    similarity_matrix = np.zeros((n_users, n_users))
    
    for i in range(n_users):
        for j in range(n_users):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                user_i = matrix.iloc[i].values
                user_j = matrix.iloc[j].values
                
                # Find common rated items
                common_items = (user_i != 0) & (user_j != 0)
                
                if np.sum(common_items) > 1:
                    try:
                        corr, _ = pearsonr(user_i[common_items], user_j[common_items])
                        similarity_matrix[i, j] = corr if not np.isnan(corr) else 0
                    except:
                        similarity_matrix[i, j] = 0
                else:
                    similarity_matrix[i, j] = 0
    
    return pd.DataFrame(similarity_matrix, index=matrix.index, columns=matrix.index)

def user_based_recommendations(user_id, user_item_matrix, df, similarity_method='cosine', n_recommendations=5):
    """Enhanced user-based collaborative filtering"""
    
    if user_item_matrix is None or user_item_matrix.empty:
        st.error("No valid user-item matrix available")
        return pd.DataFrame()
    
    if user_id not in user_item_matrix.index:
        st.error(f"User {user_id} not found in filtered dataset")
        return pd.DataFrame()
    
    # Check user's ratings
    user_ratings = user_item_matrix.loc[user_id]
    if user_ratings.sum() == 0:
        st.warning(f"User {user_id} has no ratings in filtered dataset")
        return pd.DataFrame()
    
    # Calculate similarity
    if similarity_method == 'cosine':
        similarity_df = safe_cosine_similarity(user_item_matrix)
    else:
        similarity_df = safe_pearson_similarity(user_item_matrix)
    
    if similarity_df is None:
        return pd.DataFrame()
    
    # Get similar users
    similar_users = similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users.drop(user_id, errors='ignore')
    similar_users = similar_users[similar_users > 0].dropna()
    
    if len(similar_users) == 0:
        st.warning("No similar users found")
        return pd.DataFrame()
    
    # Calculate recommendations
    recommendation_scores = {}
    user_rated_items = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
    
    for product_id in user_item_matrix.columns:
        if product_id in user_rated_items:
            continue
        
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_user_id, similarity in similar_users.head(50).items():
            rating = user_item_matrix.loc[similar_user_id, product_id]
            if rating > 0:
                weighted_sum += similarity * rating
                similarity_sum += abs(similarity)
        
        if similarity_sum > 0:
            recommendation_scores[product_id] = weighted_sum / similarity_sum
    
    if not recommendation_scores:
        st.warning("No recommendations could be generated")
        return pd.DataFrame()
    
    # Create recommendations dataframe
    recommendation_df = pd.DataFrame(
        list(recommendation_scores.items()), 
        columns=['product_id', 'score']
    )
    
    top_recommendations = recommendation_df.sort_values('score', ascending=False).head(n_recommendations)
    
    # Merge with product details
    recommendations = top_recommendations.merge(
        df[['product_id', 'name', 'description', 'images']].drop_duplicates(subset=['product_id']),
        on='product_id',
        how='left'
    )
    
    return recommendations

def item_based_recommendations(user_id, user_item_matrix, item_user_matrix, df, similarity_method='cosine', n_recommendations=5):
    """Enhanced item-based collaborative filtering"""
    
    if user_item_matrix is None or item_user_matrix is None:
        st.error("No valid matrices available")
        return pd.DataFrame()
    
    if user_id not in user_item_matrix.index:
        st.error(f"User {user_id} not found in filtered dataset")
        return pd.DataFrame()
    
    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0]
    
    if len(rated_items) == 0:
        st.warning(f"User {user_id} has no ratings")
        return pd.DataFrame()
    
    # Calculate item similarity
    if similarity_method == 'cosine':
        similarity_df = safe_cosine_similarity(item_user_matrix)
    else:
        similarity_df = safe_pearson_similarity(item_user_matrix)
    
    if similarity_df is None:
        return pd.DataFrame()
    
    # Calculate recommendations
    recommendation_scores = {}
    
    for item_id in item_user_matrix.index:
        if item_id not in rated_items.index:
            weighted_sum = 0
            similarity_sum = 0
            
            for rated_item_id, rating in rated_items.items():
                if rated_item_id in similarity_df.index:
                    similarity = similarity_df.loc[item_id, rated_item_id]
                    if not np.isnan(similarity) and similarity > 0.1:
                        weighted_sum += similarity * rating
                        similarity_sum += abs(similarity)
            
            if similarity_sum > 0:
                recommendation_scores[item_id] = weighted_sum / similarity_sum
    
    if not recommendation_scores:
        st.warning("No item-based recommendations could be generated")
        return pd.DataFrame()
    
    # Create recommendations dataframe
    recommendation_df = pd.DataFrame(
        list(recommendation_scores.items()),
        columns=['product_id', 'score']
    )
    
    top_recommendations = recommendation_df.sort_values('score', ascending=False).head(n_recommendations)
    
    # Merge with product details
    recommendations = top_recommendations.merge(
        df[['product_id', 'name', 'description', 'images']].drop_duplicates(subset=['product_id']),
        on='product_id',
        how='left'
    )
    
    return recommendations

def display_recommendations(recommendations, model_name, user_id):
    """Display recommendations with enhanced formatting"""
    if recommendations.empty:
        st.warning(f"No recommendations available for User {user_id} using {model_name}")
        return
    
    st.subheader(f"🎯 Recommendations for User {user_id}")
    st.caption(f"Model: {model_name}")
    
    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
        with st.container():
            col1, col2 = st.columns([1, 4])
            
            with col1:
                if pd.notna(row['images']):
                    try:
                        image_list = ast.literal_eval(str(row['images']))
                        if isinstance(image_list, list) and len(image_list) > 0:
                            st.image(image_list[0], width=120)
                        else:
                            st.info("📷 No image")
                    except:
                        st.info("📷 Invalid image")
                else:
                    st.info("📷 No image")
            
            with col2:
                st.markdown(f"**#{idx} {row['name']}**")
                st.markdown(f"*Score: {row['score']:.4f}*")
                if pd.notna(row['description']):
                    st.write(f"📝 {row['description'][:200]}...")
                st.markdown("---")

def evaluate_model(train_df, test_df, model_type='user_based', similarity_method='cosine'):
    """Evaluate model with enhanced error handling"""
    
    if train_df.empty or test_df.empty:
        return None, None, None, None
    
    try:
        # Create matrices
        train_user_item, train_item_user, _ = create_filtered_matrices(
            train_df, min_user_ratings=2, min_item_ratings=2
        )
        
        if train_user_item is None:
            return None, None, None, None
        
        predictions = []
        actuals = []
        
        progress_bar = st.progress(0)
        total = min(len(test_df), 100)  # Limit for faster evaluation
        
        for idx, (_, row) in enumerate(test_df.head(total).iterrows()):
            user_id = row['user_id']
            item_id = row['product_id']
            actual_rating = row['rating']
            
            if user_id in train_user_item.index and item_id in train_user_item.columns:
                # Simple prediction using user/item averages
                if model_type == 'user_based':
                    user_avg = train_user_item.loc[user_id].replace(0, np.nan).mean()
                    predicted = user_avg if not np.isnan(user_avg) else 3.0
                else:  # item_based
                    item_avg = train_user_item[item_id].replace(0, np.nan).mean()
                    predicted = item_avg if not np.isnan(item_avg) else 3.0
                
                predictions.append(predicted)
                actuals.append(actual_rating)
            
            progress_bar.progress((idx + 1) / total)
        
        progress_bar.empty()
        
        if len(predictions) == 0:
            return None, None, None, None
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        
        rating_range = max(actuals) - min(actuals)
        nmae = mae / rating_range if rating_range > 0 else 0
        
        return mse, rmse, mae, nmae
        
    except Exception as e:
        st.error(f"Evaluation error: {str(e)}")
        return None, None, None, None

def main():
    st.set_page_config(
        page_title="Enhanced Collaborative Filtering", 
        page_icon="🛍️", 
        layout="wide"
    )
    
    st.title("🛍️ Enhanced Collaborative Filtering System")
    st.markdown("### Adaptive Memory-Based Recommendations")
    
    # Load data
    df, ratings_df, products_df = load_and_process_data()
    
    if df is None:
        return
    
    # Sidebar controls
    st.sidebar.header("🎛️ Configuration")
    
    # Adaptive filtering thresholds
    st.sidebar.subheader("Filtering Thresholds")
    min_user_ratings = st.sidebar.slider("Min User Ratings", 1, 50, 5)
    min_item_ratings = st.sidebar.slider("Min Item Ratings", 1, 50, 5)
    
    # Create filtered matrices
    user_item_matrix, item_user_matrix, filtered_df = create_filtered_matrices(
        ratings_df, min_user_ratings, min_item_ratings
    )
    
    if user_item_matrix is None:
        st.error("Cannot proceed - no data after filtering")
        return
    
    # Model selection
    st.sidebar.subheader("Model Settings")
    model_options = {
        "User-Based Cosine": ("user_based", "cosine"),
        "User-Based Pearson": ("user_based", "pearson"),
        "Item-Based Cosine": ("item_based", "cosine"),
        "Item-Based Pearson": ("item_based", "pearson")
    }
    
    selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))
    model_type, similarity_method = model_options[selected_model]
    
    # User selection
    available_users = sorted(user_item_matrix.index)
    selected_user = st.sidebar.selectbox("Select User ID", available_users)
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider("Number of Recommendations", 1, 20, 5)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Generate Recommendations", type="primary"):
            with st.spinner(f"Generating recommendations using {selected_model}..."):
                try:
                    if model_type == "user_based":
                        recommendations = user_based_recommendations(
                            selected_user, user_item_matrix, df, 
                            similarity_method, n_recommendations
                        )
                    else:  # item_based
                        recommendations = item_based_recommendations(
                            selected_user, user_item_matrix, item_user_matrix, df,
                            similarity_method, n_recommendations
                        )
                    
                    display_recommendations(recommendations, selected_model, selected_user)
                    
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
    
    with col2:
        st.markdown("### 📊 Data Quality")
        
        if user_item_matrix is not None:
            # Show user's ratings
            user_ratings = user_item_matrix.loc[selected_user]
            rated_items = user_ratings[user_ratings > 0]
            
            st.metric("User's Ratings", len(rated_items))
            st.metric("Average Rating", f"{rated_items.mean():.2f}" if len(rated_items) > 0 else "N/A")
            
            # Data distribution
            st.markdown("### 📈 Rating Distribution")
            if not filtered_df.empty:
                rating_dist = filtered_df['rating'].value_counts().sort_index()
                st.bar_chart(rating_dist)
    
    # Model comparison section
    st.markdown("---")
    st.header("🔄 Model Comparison")
    
    if st.button("Compare All Models"):
        comparison_results = {}
        
        for model_name, (m_type, sim_method) in model_options.items():
            try:
                if m_type == "user_based":
                    recs = user_based_recommendations(
                        selected_user, user_item_matrix, df, sim_method, 3
                    )
                else:
                    recs = item_based_recommendations(
                        selected_user, user_item_matrix, item_user_matrix, df, sim_method, 3
                    )
                
                comparison_results[model_name] = recs
                
            except Exception as e:
                st.error(f"Error with {model_name}: {str(e)}")
        
        # Display comparison
        if comparison_results:
            cols = st.columns(2)
            
            for idx, (model_name, recs) in enumerate(comparison_results.items()):
                with cols[idx % 2]:
                    st.subheader(model_name)
                    if not recs.empty:
                        for _, row in recs.iterrows():
                            st.write(f"• {row['name']} ({row['score']:.3f})")
                    else:
                        st.write("No recommendations")
                    st.markdown("---")
    
    # Evaluation section
    st.markdown("---")
    st.header("📊 Model Evaluation")
    
    if st.button("Evaluate Models"):
        with st.spinner("Running evaluation..."):
            # Split data for evaluation
            train_df, test_df = train_test_split(filtered_df, test_size=0.2, random_state=42)
            
            results = []
            
            for model_name, (m_type, sim_method) in model_options.items():
                start_time = time.time()
                mse, rmse, mae, nmae = evaluate_model(train_df, test_df, m_type, sim_method)
                eval_time = time.time() - start_time
                
                if mse is not None:
                    results.append({
                        'Model': model_name,
                        'MSE': mse,
                        'RMSE': rmse,
                        'MAE': mae,
                        'NMAE': nmae,
                        'Time (s)': eval_time
                    })
            
            if results:
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                
                # Best model
                best_model = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
                st.success(f"🏆 Best model by RMSE: {best_model}")
            else:
                st.warning("No evaluation results available")

if __name__ == "__main__":
    main()