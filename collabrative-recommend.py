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

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        # Load product details
        products_df = pd.read_csv('data.csv')
        # Load ratings
        ratings_df = pd.read_csv('ratings.csv')
        
        # Merge ratings with product details
        df = ratings_df.merge(products_df, on='product_id', how='left')
        
        # Create user-item matrix
        user_item_matrix = ratings_df.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
        
        # Create item-user matrix (transpose for item-based filtering)
        item_user_matrix = user_item_matrix.T
        
        return df, user_item_matrix, item_user_matrix
    except FileNotFoundError:   
        st.error("Error: data.csv or ratings.csv file not found. Please ensure both files are in the same directory.")
        return None, None, None

# Calculate Pearson correlation coefficient
def pearson_similarity(matrix):
    """Calculate Pearson correlation similarity matrix"""
    n_users = matrix.shape[0]
    similarity_matrix = np.zeros((n_users, n_users))
    
    for i in range(n_users):
        for j in range(n_users):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                user_i = matrix.iloc[i].values
                user_j = matrix.iloc[j].values
                
                # Find common rated items (non-zero ratings)
                common_items = (user_i != 0) & (user_j != 0)
                
                if np.sum(common_items) > 1:  # Need at least 2 common items
                    try:
                        corr, _ = pearsonr(user_i[common_items], user_j[common_items])
                        similarity_matrix[i, j] = corr if not np.isnan(corr) else 0
                    except:
                        similarity_matrix[i, j] = 0
                else:
                    similarity_matrix[i, j] = 0
    
    return similarity_matrix

# User-User Collaborative Filtering
def user_user_recommendations(user_id, user_item_matrix, df, similarity_method='cosine', n_recommendations=5):
    """Generate recommendations using User-User Collaborative Filtering"""
    
    # Debug: Check if user exists
    if user_id not in user_item_matrix.index:
        st.error(f"User {user_id} not found in the dataset!")
        return pd.DataFrame()
    
    # Check if user has any ratings
    user_ratings = user_item_matrix.loc[user_id]
    if user_ratings.sum() == 0:
        st.warning(f"User {user_id} has no ratings. Cannot generate recommendations.")
        return pd.DataFrame()
    
    if similarity_method == 'cosine':
        # Calculate cosine similarity between users
        user_similarity = cosine_similarity(user_item_matrix)
        user_similarity_df = pd.DataFrame(user_similarity, 
                                        index=user_item_matrix.index, 
                                        columns=user_item_matrix.index)
    else:  # pearson
        # Calculate Pearson correlation similarity
        user_similarity = pearson_similarity(user_item_matrix)
        user_similarity_df = pd.DataFrame(user_similarity,
                                        index=user_item_matrix.index,
                                        columns=user_item_matrix.index)
    
    # Get similarity scores for the target user
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    
    # Remove the user itself from similar users
    similar_users = similar_users.drop(user_id, errors='ignore')
    
    # Filter out users with similarity <= 0 or NaN
    similar_users = similar_users[similar_users > 0]
    similar_users = similar_users.dropna()
    
    if len(similar_users) == 0:
        st.warning("No similar users found for recommendations.")
        return pd.DataFrame()
    
    # Get top similar users (limit to avoid noise)
    top_similar_users = similar_users.head(min(50, len(similar_users)))
    
    # Debug info
    st.info(f"Found {len(top_similar_users)} similar users with similarity > 0")
    
    # Calculate recommendations for each item
    recommendation_scores = {}
    user_rated_items = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
    
    for product_id in user_item_matrix.columns:
        if product_id in user_rated_items:
            continue  # Skip items already rated by the user
        
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_user_id, similarity in top_similar_users.items():
            rating = user_item_matrix.loc[similar_user_id, product_id]
            if rating > 0:  # Only consider items that the similar user has rated
                weighted_sum += similarity * rating
                similarity_sum += abs(similarity)
        
        if similarity_sum > 0:
            recommendation_scores[product_id] = weighted_sum / similarity_sum
        else:
            recommendation_scores[product_id] = 0
    
    # Convert to DataFrame
    recommendation_df = pd.DataFrame(list(recommendation_scores.items()), 
                                   columns=['product_id', 'score'])
    
    # Filter out zero scores and sort
    recommendation_df = recommendation_df[recommendation_df['score'] > 0]
    top_recommendations = recommendation_df.sort_values('score', ascending=False).head(n_recommendations)
    
    if len(top_recommendations) == 0:
        st.warning("No recommendations could be generated with positive scores.")
        return pd.DataFrame()
    
    # Merge with product details
    recommendations = top_recommendations.merge(
        df[['product_id', 'name', 'description', 'images']].drop_duplicates(subset=['product_id']),
        on='product_id',
        how='left'
    )
    
    return recommendations

# Item-Item Collaborative Filtering
def item_item_recommendations(user_id, user_item_matrix, item_user_matrix, df, similarity_method='cosine', n_recommendations=5):
    """Generate recommendations using Item-Item Collaborative Filtering"""
    
    # Check if user exists and has ratings
    if user_id not in user_item_matrix.index:
        st.error(f"User {user_id} not found in the dataset!")
        return pd.DataFrame()
    
    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0]
    
    if len(rated_items) == 0:
        st.warning(f"User {user_id} has no ratings. Cannot generate item-based recommendations.")
        return pd.DataFrame()
    
    st.info(f"User has rated {len(rated_items)} items")
    
    # Calculate item similarity matrix (only for items that have been rated)
    # Filter to only items with at least some ratings to avoid all-zero vectors
    item_user_filtered = item_user_matrix[item_user_matrix.sum(axis=1) > 0]
    
    if similarity_method == 'cosine':
        # Calculate cosine similarity between items
        item_similarity = cosine_similarity(item_user_filtered)
        item_similarity_df = pd.DataFrame(item_similarity,
                                        index=item_user_filtered.index,
                                        columns=item_user_filtered.index)
    else:  # pearson
        # Calculate Pearson correlation similarity between items
        item_similarity = pearson_similarity(item_user_filtered)
        item_similarity_df = pd.DataFrame(item_similarity,
                                        index=item_user_filtered.index,
                                        columns=item_user_filtered.index)
    
    # Calculate recommendation scores for all unrated items
    recommendation_scores = {}
    
    for item_id in item_similarity_df.index:
        if item_id not in rated_items.index:  # Only consider unrated items
            weighted_sum = 0
            similarity_sum = 0
            
            for rated_item_id, rating in rated_items.items():
                if rated_item_id in item_similarity_df.index:
                    similarity = item_similarity_df.loc[item_id, rated_item_id]
                    if not np.isnan(similarity) and similarity > 0.1:  # Threshold for meaningful similarity
                        weighted_sum += similarity * rating
                        similarity_sum += abs(similarity)
            
            if similarity_sum > 0:
                recommendation_scores[item_id] = weighted_sum / similarity_sum
    
    if len(recommendation_scores) == 0:
        st.warning("No item-based recommendations could be generated.")
        return pd.DataFrame()
    
    # Convert to DataFrame and sort
    recommendation_df = pd.DataFrame(list(recommendation_scores.items()), 
                                   columns=['product_id', 'score'])
    
    # Filter positive scores and sort
    recommendation_df = recommendation_df[recommendation_df['score'] > 0]
    top_recommendations = recommendation_df.sort_values('score', ascending=False).head(n_recommendations)
    
    if len(top_recommendations) == 0:
        st.warning("No item-based recommendations with positive scores found.")
        return pd.DataFrame()
    
    # Merge with product details
    recommendations = top_recommendations.merge(
        df[['product_id', 'name', 'description', 'images']].drop_duplicates(subset=['product_id']),
        on='product_id',
        how='left'
    )
    
    return recommendations

# Evaluation Functions
def split_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets for evaluation, ensuring users have at least 2 ratings"""
    # Filter users with at least 2 ratings
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    df_filtered = df[df['user_id'].isin(valid_users)]
    
    if len(df_filtered) == 0:
        st.error("No users with at least 2 ratings available for evaluation.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Perform stratified split
    try:
        train_df, test_df = train_test_split(
            df_filtered, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df_filtered['user_id']
        )
        return train_df, test_df
    except ValueError as e:
        st.error(f"Error in data splitting: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def predict_rating(user_id, item_id, train_matrix, model_type='user_user', similarity_method='cosine'):
    """Predict rating for a specific user-item pair"""
    
    if user_id not in train_matrix.index or item_id not in train_matrix.columns:
        return np.nan
    
    if model_type == 'user_user':
        return predict_user_user_rating(user_id, item_id, train_matrix, similarity_method)
    else:  # item_item
        return predict_item_item_rating(user_id, item_id, train_matrix, similarity_method)

def predict_user_user_rating(user_id, item_id, train_matrix, similarity_method='cosine'):
    """Predict rating using User-User CF"""
    
    # Calculate similarity matrix
    if similarity_method == 'cosine':
        user_similarity = cosine_similarity(train_matrix)
        similarity_df = pd.DataFrame(user_similarity, 
                                   index=train_matrix.index, 
                                   columns=train_matrix.index)
    else:  # pearson
        user_similarity = pearson_similarity(train_matrix)
        similarity_df = pd.DataFrame(user_similarity,
                                   index=train_matrix.index,
                                   columns=train_matrix.index)
    
    # Get similar users who have rated this item
    similar_users = similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users.drop(user_id, errors='ignore')
    similar_users = similar_users[similar_users > 0].dropna()
    
    if len(similar_users) == 0:
        return train_matrix.mean().mean()  # Global average as fallback
    
    # Calculate weighted average
    weighted_sum = 0
    similarity_sum = 0
    
    for similar_user_id, similarity in similar_users.head(50).items():
        rating = train_matrix.loc[similar_user_id, item_id]
        if rating > 0:
            weighted_sum += similarity * rating
            similarity_sum += abs(similarity)
    
    if similarity_sum > 0:
        return weighted_sum / similarity_sum
    else:
        return train_matrix.mean().mean()

def predict_item_item_rating(user_id, item_id, train_matrix, similarity_method='cosine'):
    """Predict rating using Item-Item CF"""
    
    # Get item-user matrix
    item_user_matrix = train_matrix.T
    
    # Calculate similarity matrix
    if similarity_method == 'cosine':
        item_similarity = cosine_similarity(item_user_matrix)
        similarity_df = pd.DataFrame(item_similarity,
                                   index=item_user_matrix.index,
                                   columns=item_user_matrix.index)
    else:  # pearson
        item_similarity = pearson_similarity(item_user_matrix)
        similarity_df = pd.DataFrame(item_similarity,
                                   index=item_user_matrix.index,
                                   columns=item_user_matrix.index)
    
    # Get user's ratings
    user_ratings = train_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0]
    
    if len(rated_items) == 0:
        return train_matrix.mean().mean()
    
    # Calculate weighted average
    weighted_sum = 0
    similarity_sum = 0
    
    for rated_item_id, rating in rated_items.items():
        if rated_item_id in similarity_df.index and item_id in similarity_df.columns:
            similarity = similarity_df.loc[item_id, rated_item_id]
            if not np.isnan(similarity) and similarity > 0:
                weighted_sum += similarity * rating
                similarity_sum += abs(similarity)
    
    if similarity_sum > 0:
        return weighted_sum / similarity_sum
    else:
        return train_matrix.mean().mean()

def evaluate_model(train_df, test_df, model_type='user_user', similarity_method='cosine'):
    """Evaluate model performance using MSE, RMSE, MAE, NMAE"""
    
    if train_df.empty or test_df.empty:
        return None, None, None, None
    
    # Create train matrix
    train_matrix = train_df.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
    
    # Predict ratings for test set
    predictions = []
    actual_ratings = []
    
    progress_bar = st.progress(0)
    total_predictions = len(test_df)
    
    for idx, (_, row) in enumerate(test_df.iterrows()):
        user_id = row['user_id']
        item_id = row['product_id']
        actual_rating = row['rating']
        
        predicted_rating = predict_rating(user_id, item_id, train_matrix, model_type, similarity_method)
        
        if not np.isnan(predicted_rating):
            predictions.append(predicted_rating)
            actual_ratings.append(actual_rating)
        
        # Update progress
        progress_bar.progress((idx + 1) / total_predictions)
    
    progress_bar.empty()
    
    if len(predictions) == 0:
        return None, None, None, None
    
    predictions = np.array(predictions)
    actual_ratings = np.array(actual_ratings)
    
    # Calculate metrics
    mse = mean_squared_error(actual_ratings, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_ratings, predictions)
    
    # NMAE (Normalized MAE) - normalize by rating range
    rating_range = actual_ratings.max() - actual_ratings.min()
    nmae = mae / rating_range if rating_range > 0 else 0
    
    return mse, rmse, mae, nmae

# Display recommendations
def display_recommendations(recommendations, model_name, user_id, n_recommendations):
    """Display recommendations in a nice format"""
    st.subheader(f"🎯 Top {n_recommendations} Recommendations for User {user_id}")
    st.caption(f"Model: {model_name}")
    
    if recommendations.empty:
        st.warning("No recommendations available for this user with the selected model.")
        return
    
    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
        with st.container():
            col1, col2 = st.columns([1, 4])
            
            with col1:
                if pd.notna(row['images']):
                    try:
                        # Parse the images string as a list and take the first URL
                        image_list = ast.literal_eval(str(row['images']))
                        if isinstance(image_list, list) and len(image_list) > 0:
                            st.image(image_list[0], width=120)
                        else:
                            st.info("📷 No image")
                    except (ValueError, SyntaxError):
                        st.info("📷 Invalid image")
                else:
                    st.info("📷 No image")
            
            with col2:
                st.markdown(f"**#{idx} {row['name']}**")
                st.markdown(f"*Score: {row['score']:.4f}*")
                if pd.notna(row['description']):
                    st.write(f"📝 {row['description'][:200]}...")
                st.markdown("---")

# Display evaluation results
def display_evaluation_results(results_df):
    """Display evaluation results in a nice format"""
    st.subheader("📊 Model Evaluation Results")
    
    # Color coding for better/worse performance
    def color_metric(val, metric_name):
        if metric_name in ['MSE', 'RMSE', 'MAE', 'NMAE']:
            # Lower is better for error metrics
            if val < results_df[metric_name].median():
                return 'background-color: #d4edda'  # Light green
            else:
                return 'background-color: #f8d7da'  # Light red
        elif metric_name == 'Time (s)':
            # Lower is better for time
            if val < results_df[metric_name].median():
                return 'background-color: #d4edda'  # Light green
            else:
                return 'background-color: #f8d7da'  # Light red
        return ''
    
    # Style the dataframe
    styled_df = results_df.style.applymap(lambda x: color_metric(x, 'MSE'), subset=['MSE'])\
                                .applymap(lambda x: color_metric(x, 'RMSE'), subset=['RMSE'])\
                                .applymap(lambda x: color_metric(x, 'MAE'), subset=['MAE'])\
                                .applymap(lambda x: color_metric(x, 'NMAE'), subset=['NMAE'])\
                                .applymap(lambda x: color_metric(x, 'Time (s)'), subset=['Time (s)'])\
                                .format({'MSE': '{:.4f}', 'RMSE': '{:.4f}', 'MAE': '{:.4f}', 'NMAE': '{:.4f}', 'Time (s)': '{:.2f}'})
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Best model summary
    best_mse = results_df.loc[results_df['MSE'].idxmin(), 'Model']
    best_rmse = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
    best_mae = results_df.loc[results_df['MAE'].idxmin(), 'Model']
    best_nmae = results_df.loc[results_df['NMAE'].idxmin(), 'Model']
    best_time = results_df.loc[results_df['Time (s)'].idxmin(), 'Model']
    
    st.subheader("🏆 Best Performing Models")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Best MSE", best_mse, f"{results_df['MSE'].min():.4f}")
    with col2:
        st.metric("Best RMSE", best_rmse, f"{results_df['RMSE'].min():.4f}")
    with col3:
        st.metric("Best MAE", best_mae, f"{results_df['MAE'].min():.4f}")
    with col4:
        st.metric("Best NMAE", best_nmae, f"{results_df['NMAE'].min():.4f}")
    with col5:
        st.metric("Fastest Time", best_time, f"{results_df['Time (s)'].min():.2f}s")
    
    # Interpretation
    st.subheader("📖 Metrics Interpretation")
    st.write("""
    - **MSE (Mean Squared Error)**: Squares the errors, penalizes large errors more heavily
    - **RMSE (Root Mean Squared Error)**: Square root of MSE, same unit as original ratings
    - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual ratings
    - **NMAE (Normalized MAE)**: MAE normalized by rating range (0-1 scale)
    - **Time (s)**: Time taken to evaluate the model in seconds
    
    **Lower values = Better performance** for all metrics
    """)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Collaborative Filtering Recommendation System", 
                       page_icon="🛍️", 
                       layout="wide")
    
    st.title("🛍️ Collaborative Filtering Recommendation System")
    st.markdown("### Compare 4 Memory-Based Models")
    
    # Load data
    df, user_item_matrix, item_user_matrix = load_data()
    
    if df is None:
        return
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Controls")
    
    # Model selection
    model_options = {
        "User-User Cosine": ("user_user", "cosine"),
        "User-User Pearson": ("user_user", "pearson"), 
        "Item-Item Cosine": ("item_item", "cosine"),
        "Item-Item Pearson": ("item_item", "pearson")
    }
    
    selected_model = st.sidebar.selectbox("Select Recommendation Model", list(model_options.keys()))
    model_type, similarity_method = model_options[selected_model]
    
    # User selection
    user_ids = sorted(df['user_id'].unique())
    selected_user = st.sidebar.selectbox("Select User ID", user_ids)
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider("Number of Recommendations", 1, 20, 5)
    
    # Dataset info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.info(f"Users: {len(user_ids)}")
    st.sidebar.info(f"Products: {df['product_id'].nunique()}")
    st.sidebar.info(f"Ratings: {len(df)}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Generate Recommendations", type="primary"):
            with st.spinner(f"Generating recommendations using {selected_model}..."):
                try:
                    # Add data debugging
                    st.info(f"Dataset shape: {df.shape}")
                    st.info(f"User-item matrix shape: {user_item_matrix.shape}")
                    st.info(f"Selected user: {selected_user}")
                    
                    # Check user's existing ratings
                    user_existing_ratings = user_item_matrix.loc[selected_user]
                    num_ratings = (user_existing_ratings > 0).sum()
                    st.info(f"User {selected_user} has {num_ratings} existing ratings")
                    
                    if num_ratings == 0:
                        st.error(f"User {selected_user} has no ratings! Please select a different user.")
                        
                        # Show users with ratings
                        users_with_ratings = []
                        for uid in user_ids[:10]:  # Check first 10 users
                            if (user_item_matrix.loc[uid] > 0).sum() > 0:
                                users_with_ratings.append(uid)
                        
                        if users_with_ratings:
                            st.info(f"Try these users instead: {users_with_ratings}")
                    else:
                        if model_type == "user_user":
                            recommendations = user_user_recommendations(
                                selected_user, user_item_matrix, df, 
                                similarity_method, n_recommendations
                            )
                        else:  # item_item
                            recommendations = item_item_recommendations(
                                selected_user, user_item_matrix, item_user_matrix, df,
                                similarity_method, n_recommendations
                            )
                        
                        if not recommendations.empty:
                            display_recommendations(recommendations, selected_model, selected_user, n_recommendations)
                        else:
                            st.warning("No recommendations generated. This could be due to:")
                            st.write("- Sparse data (too few similar users/items)")
                            st.write("- All similar users/items have zero similarity")
                            st.write("- User has rated too few items")
                            st.write("- Try a different model or user")
                    
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
                    st.error("Please check your data format and try again.")
    
    with col2:
        st.markdown("### 🔍 Model Comparison")
        st.markdown("""
        **User-User Models:**
        - Find users with similar preferences
        - Recommend items liked by similar users
        
        **Item-Item Models:**
        - Find items similar to user's preferences
        - Recommend based on item relationships
        
        **Similarity Measures:**
        - **Cosine:** Angle between rating vectors
        - **Pearson:** Linear correlation coefficient
        """)
        
        # Show user's current ratings
        if st.checkbox("Show User's Current Ratings"):
            user_ratings = user_item_matrix.loc[selected_user]
            rated_items = user_ratings[user_ratings > 0]
            
            if len(rated_items) > 0:
                st.markdown(f"### User {selected_user}'s Ratings")
                user_items_df = pd.DataFrame({
                    'Product ID': rated_items.index,
                    'Rating': rated_items.values
                }).head(10)
                st.dataframe(user_items_df, use_container_width=True)
            else:
                st.info("This user has no ratings in the dataset.")
    
    # Evaluation section
    st.markdown("---")
    st.header("📊 Model Evaluation")
    
    eval_col1, eval_col2 = st.columns([2, 1])
    
    with eval_col2:
        st.subheader("⚙️ Evaluation Settings")
        test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        sample_size = st.slider("Sample Size for Evaluation", 100, 50000, 500)
        st.info(f"Will evaluate on {sample_size} random samples")
        
        if st.button("🔍 Evaluate All Models", type="secondary"):
            st.session_state.run_evaluation = True
    
    with eval_col1:
        if st.session_state.get('run_evaluation', False):
            st.subheader("🔄 Running Model Evaluation...")
            
            # Sample data for faster evaluation
            df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
            
            # Split data
            train_df, test_df = split_data(df_sample, test_size=test_size)
            
            if train_df.empty or test_df.empty:
                st.error("Evaluation failed: Not enough valid data after filtering users with single ratings.")
            else:
                st.info(f"Training set: {len(train_df)} ratings")
                st.info(f"Test set: {len(test_df)} ratings")
                
                # Evaluate all models
                models = [
                    ('User-User Cosine', 'user_user', 'cosine'),
                    ('User-User Pearson', 'user_user', 'pearson'),
                    ('Item-Item Cosine', 'item_item', 'cosine'),
                    ('Item-Item Pearson', 'item_item', 'pearson')
                ]
                
                results = []
                
                for model_name, model_type, similarity_method in models:
                    st.write(f"Evaluating {model_name}...")
                    
                    try:
                        # Measure evaluation time
                        start_time = time.time()
                        mse, rmse, mae, nmae = evaluate_model(
                            train_df, test_df, model_type, similarity_method
                        )
                        end_time = time.time()
                        eval_time = end_time - start_time
                        
                        if mse is not None:
                            results.append({
                                'Model': model_name,
                                'MSE': mse,
                                'RMSE': rmse,
                                'MAE': mae,
                                'NMAE': nmae,
                                'Time (s)': eval_time
                            })
                            st.success(f"✅ {model_name} completed in {eval_time:.2f} seconds")
                        else:
                            st.warning(f"⚠️ {model_name} failed - no valid predictions")
                            
                    except Exception as e:
                        st.error(f"❌ {model_name} error: {str(e)}")
                
                if results:
                    results_df = pd.DataFrame(results)
                    display_evaluation_results(results_df)
                    
                    # Save results to session state
                    st.session_state.evaluation_results = results_df
                else:
                    st.error("No models could be evaluated successfully")
                
                # Reset evaluation flag
                st.session_state.run_evaluation = False
    
    # Comparison section
    st.markdown("---")
    if st.checkbox("🔄 Compare All Models"):
        st.markdown("### Model Comparison Results")
        
        comparison_cols = st.columns(2)
        
        with comparison_cols[0]:
            st.markdown("#### User-Based Models")
            
            # User-User Cosine
            try:
                uu_cosine = user_user_recommendations(selected_user, user_item_matrix, df, 'cosine', 3)
                st.markdown("**User-User Cosine:**")
                if not uu_cosine.empty:
                    for _, row in uu_cosine.iterrows():
                        st.write(f"• {row['name']} (Score: {row['score']:.3f})")
                else:
                    st.write("No recommendations")
            except:
                st.write("Error in User-User Cosine")
            
            st.markdown("---")
            
            # User-User Pearson
            try:
                uu_pearson = user_user_recommendations(selected_user, user_item_matrix, df, 'pearson', 3)
                st.markdown("**User-User Pearson:**")
                if not uu_pearson.empty:
                    for _, row in uu_pearson.iterrows():
                        st.write(f"• {row['name']} (Score: {row['score']:.3f})")
                else:
                    st.write("No recommendations")
            except:
                st.write("Error in User-User Pearson")
        
        with comparison_cols[1]:
            st.markdown("#### Item-Based Models")
            
            # Item-Item Cosine
            try:
                ii_cosine = item_item_recommendations(selected_user, user_item_matrix, item_user_matrix, df, 'cosine', 3)
                st.markdown("**Item-Item Cosine:**")
                if not ii_cosine.empty:
                    for _, row in ii_cosine.iterrows():
                        st.write(f"• {row['name']} (Score: {row['score']:.3f})")
                else:
                    st.write("No recommendations")
            except:
                st.write("Error in Item-Item Cosine")
            
            st.markdown("---")
            
            # Item-Item Pearson
            try:
                ii_pearson = item_item_recommendations(selected_user, user_item_matrix, item_user_matrix, df, 'pearson', 3)
                st.markdown("**Item-Item Pearson:**")
                if not ii_pearson.empty:
                    for _, row in ii_pearson.iterrows():
                        st.write(f"• {row['name']} (Score: {row['score']:.3f})")
                else:
                    st.write("No recommendations")
            except:
                st.write("Error in Item-Item Pearson")

    # Display previous evaluation results if available
    if 'evaluation_results' in st.session_state:
        st.markdown("---")
        st.subheader("📈 Previous Evaluation Results")
        display_evaluation_results(st.session_state.evaluation_results)

if __name__ == "__main__":
    main()