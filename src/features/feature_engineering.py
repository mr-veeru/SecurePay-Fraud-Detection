"""
SecurePay Fraud Detection - Feature Engineering
----------------------------------------------
This module provides functions for feature engineering and selection.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import print_header, load_creditcard_data, apply_mappings

def load_raw_data():
    """Load the raw fraud detection dataset."""
    print("Loading raw data...")
    
    # Try to locate the data file
    for path in [
        "data/creditcard.csv",  # Standard path
        "../data/creditcard.csv",  # Running from src directory
        "../../data/creditcard.csv"  # Running from deeper directory
    ]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"Loaded data from {path} with {len(df)} rows and {len(df.columns)} columns")
                
                # Remove potential index or unnamed columns to prevent data leakage
                unnamed_cols = [col for col in df.columns if 'unnamed' in col.lower() or 'index' in col.lower()]
                if unnamed_cols:
                    print(f"Dropping {len(unnamed_cols)} potential index columns to prevent data leakage: {unnamed_cols}")
                    df = df.drop(columns=unnamed_cols)
                
                return df
            except Exception as e:
                print(f"Error loading data from {path}: {e}")
    
    # If no file found, try to download the dataset
    print("No local data file found. Trying to download the dataset...")
    try:
        # Include code to download from a URL if appropriate
        pass
    except:
        print("Failed to download the dataset.")
    
    print("No data found. Please download the creditcard dataset and place it in data/")
    return None

def create_synthetic_fraud_labels(df):
    """
    Create synthetic fraud labels based on transaction characteristics
    when no ground truth labels are available.
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Identify amount column - typically named 'Amount' in creditcard.csv
    amount_col = 'Amount' if 'Amount' in df.columns else None
    
    # If Amount column not found, try to identify it based on distribution
    if amount_col is None:
        for col in df.columns:
            # Amount columns typically have a highly skewed distribution with mostly small values
            if df[col].dtype in ['float64', 'int64'] and df[col].min() >= 0 and df[col].mean() < df[col].std():
                amount_col = col
                print(f"Using {col} as the transaction amount column")
                break
    
    if amount_col is None:
        print("Warning: Could not identify transaction amount column")
        # Create random labels with 0.5% fraud ratio if no amount column found
        np.random.seed(42)
        df['is_fraud'] = np.random.choice([0, 1], size=len(df), p=[0.995, 0.005])
        return df
    
    # Create is_fraud column based on heuristics (large amounts, unusual times if available)
    # Mark transactions with unusually high amounts as potentially fraudulent
    amount_threshold = df[amount_col].quantile(0.99)  # Top 1% of amounts
    
    # Initialize fraud flags
    df['is_fraud'] = 0
    
    # Flag high-value transactions as potentially fraudulent
    df.loc[df[amount_col] > amount_threshold, 'is_fraud'] = 1
    
    # If time column exists, also flag unusual time patterns
    time_cols = [col for col in df.columns if 'time' in col.lower()]
    if time_cols:
        time_col = time_cols[0]
        # Flag unusual times (e.g., late night transactions) as potentially fraudulent
        time_values = df[time_col].values
        unusual_times = (time_values % 86400) < 14400  # Transactions between midnight and 4am
        df.loc[unusual_times & (df[amount_col] > df[amount_col].median()), 'is_fraud'] = 1
    
    # Ensure a reasonable fraud rate (~0.5-2%)
    fraud_rate = df['is_fraud'].mean()
    if fraud_rate > 0.02:
        # If too many frauds, keep only the most extreme ones
        fraud_indices = df[df['is_fraud'] == 1].index
        keep_n = int(len(df) * 0.01)  # Aim for 1% fraud rate
        drop_indices = np.random.choice(
            fraud_indices, 
            size=max(0, len(fraud_indices) - keep_n), 
            replace=False
        )
        df.loc[drop_indices, 'is_fraud'] = 0
    
    print(f"Created synthetic fraud labels. Fraud rate: {df['is_fraud'].mean():.4f}")
    return df

def check_data_quality(df):
    """Check for data quality issues and fix them when possible."""
    print("Checking data quality...")
    
    # Get basic statistics
    n_rows, n_cols = df.shape
    print(f"Dataset shape: {n_rows} rows, {n_cols} columns")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    if total_missing > 0:
        print(f"Found {total_missing} missing values across {(missing_values > 0).sum()} columns")
        columns_with_missing = missing_values[missing_values > 0].index.tolist()
        for col in columns_with_missing:
            missing_ratio = missing_values[col] / n_rows
            print(f"  Column '{col}': {missing_values[col]} missing values ({missing_ratio:.2%})")
            
            # Decide how to handle missing values based on ratio
            if missing_ratio < 0.05:  # Less than 5% missing
                if df[col].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    df[col] = df[col].fillna(df[col].median())
                    print(f"    Filled missing values with median ({df[col].median()})")
                else:
                    # Fill categorical columns with mode
                    df[col] = df[col].fillna(df[col].mode()[0])
                    print(f"    Filled missing values with mode ({df[col].mode()[0]})")
            elif missing_ratio < 0.3:  # Between 5% and 30% missing
                if df[col].dtype in ['int64', 'float64']:
                    # Create a flag for missingness and then fill with median
                    missing_flag_col = f"{col}_missing"
                    df[missing_flag_col] = df[col].isnull().astype(int)
                    df[col] = df[col].fillna(df[col].median())
                    print(f"    Created missing flag column '{missing_flag_col}' and filled with median")
                else:
                    # For categorical, treat missing as its own category
                    df[col] = df[col].fillna("MISSING")
                    print(f"    Filled missing values with 'MISSING' category")
            else:
                # Too many missing values, consider dropping the column
                print(f"    WARNING: Column '{col}' has {missing_ratio:.2%} missing values. Consider dropping it.")
    else:
        print("No missing values found")
    
    # Check for constant or near-constant columns (low variance)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    near_constant_cols = []
    
    for col in numeric_cols:
        if col == 'is_fraud':  # Skip target column 
            continue
            
        # Calculate variance relative to the mean to handle different scales
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        # Avoid division by zero
        if mean_val != 0 and std_val is not None:
            relative_variance = std_val / (abs(mean_val) + 1e-8)
            if relative_variance < 0.01:  # Very low relative variance
                near_constant_cols.append((col, relative_variance))
    
    if near_constant_cols:
        print(f"Found {len(near_constant_cols)} near-constant columns:")
        for col, var in near_constant_cols:
            unique_count = df[col].nunique()
            print(f"  Column '{col}': relative variance = {var:.6f}, unique values = {unique_count}")
            if unique_count <= 1:
                print(f"    Dropping constant column '{col}'")
                df = df.drop(columns=[col])
    
    # Return the dataframe with quality issues fixed
    return df

def create_additional_features(df):
    """Create additional features that might help in fraud detection."""
    print("\nCreating additional features...")
    
    df_new = df.copy()
    new_feature_count = 0
    
    # Transaction amount related features - simplify code by using next()
    amt_col = next((col for col in ['amt', 'amount'] if col in df.columns), None)
    
    if amt_col:
        # Log transform of amount to reduce skewness
        df_new['log_amount'] = np.log1p(df[amt_col])
        new_feature_count += 1
        
        # Calculate quantiles once instead of multiple times
        q33 = df[amt_col].quantile(0.33)
        q66 = df[amt_col].quantile(0.66)
        
        # Create amount bins
        df_new['amount_bin_low'] = (df[amt_col] <= q33).astype(int)
        df_new['amount_bin_medium'] = ((df[amt_col] > q33) & (df[amt_col] <= q66)).astype(int)
        df_new['amount_bin_high'] = (df[amt_col] > q66).astype(int)
        new_feature_count += 3
    
    # Time-based features if a timestamp column exists - simplify code by using next()
    time_col = next((col for col in ['trans_date_trans_time', 'datetime', 'timestamp'] if col in df.columns), None)
    
    if time_col and pd.api.types.is_numeric_dtype(df[time_col]):
        try:
            # Already converted to numeric by check_data_quality
            df_new['hour_of_day'] = df[time_col] % 24
            df_new['day_of_week'] = (df[time_col] // 24) % 7
            new_feature_count += 2
        except Exception as e:
            print(f"Could not create time-based features: {e}")
    
    # Location-based features (if available)
    if all(col in df.columns for col in ['lat', 'long']):
        # Distance from origin
        df_new['distance_from_origin'] = np.sqrt(df['lat']**2 + df['long']**2)
        new_feature_count += 1
    
    # Category-based features - simplify code by using next()
    cat_col = next((col for col in ['category', 'merchant_cat', 'merchant_category'] if col in df.columns), None)
    
    if cat_col:
        # Calculate category risk based on fraud rate
        category_risk = df.groupby(cat_col)['is_fraud'].mean().to_dict()
        df_new['category_risk'] = df[cat_col].map(category_risk)
        new_feature_count += 1
    
    print(f"Created {new_feature_count} new features.")
    print(f"New dataframe shape: {df_new.shape}")
    
    return df_new

def select_features(df, target_col='is_fraud', method='mutual_info', k=15):
    """Select the most important features using different methods."""
    print(f"\nSelecting top {k} features using {method}...")
    
    # Filter to only include numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    if target_col not in numeric_df.columns:
        print(f"Warning: Target column '{target_col}' is not numeric. Feature selection may not work properly.")
        # Try to convert the target if it's not in numeric columns
        try:
            numeric_df[target_col] = df[target_col]
        except:
            print(f"Error: Could not include target column '{target_col}' in feature selection.")
            return df, df.columns.tolist()
    
    X = numeric_df.drop(target_col, axis=1)
    y = numeric_df[target_col]
    
    # If we have too few features, skip selection
    if X.shape[1] <= k:
        print(f"Only {X.shape[1]} numeric features available, which is less than or equal to k={k}.")
        print("Skipping feature selection and returning all numeric features.")
        return numeric_df, X.columns.tolist()
    
    try:
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
        else:
            print("Invalid method. Using mutual_info as default.")
            selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
        
        X_new = selector.fit_transform(X, y)
        
        # Get the selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        # Get the scores
        scores = selector.scores_
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': scores
        })
        feature_scores = feature_scores.sort_values('Score', ascending=False)
        
        print("Top features:")
        print(feature_scores.head(k))
        
        try:
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.barh(feature_scores.head(k)['Feature'], feature_scores.head(k)['Score'])
            plt.xlabel('Score')
            plt.ylabel('Feature')
            plt.title(f'Top {k} Features by {method.capitalize()} Score')
            plt.tight_layout()
            
            # Create directory if it doesn't exist
            os.makedirs('models/figures', exist_ok=True)
            plt.savefig(f'models/figures/feature_importance_{method}.png')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create feature importance plot: {e}")
        
        # Create a new dataframe with only the selected features and the target
        # Also include the target column from the original dataframe
        selected_df = df[selected_features + [target_col]]
        
        return selected_df, selected_features
    
    except Exception as e:
        print(f"Error during feature selection: {e}")
        print("Returning original numeric features.")
        return numeric_df, X.columns.tolist()

def handle_class_imbalance(df, method='auto'):
    """
    Balance the classes in the dataset using resampling techniques.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset with features and target
    method : str
        Resampling method to use ('smote', 'adasyn', 'random', or 'auto')
        
    Returns:
    --------
    pandas DataFrame
        Balanced dataset
    """
    print("\nHandling class imbalance...")
    
    # Check if target column exists
    if 'is_fraud' not in df.columns:
        print("Error: 'is_fraud' column not found. Cannot balance classes.")
        return df
    
    # Get class distribution
    class_counts = df['is_fraud'].value_counts()
    print("Original class distribution:")
    print(class_counts)
    
    # Check if there's any imbalance worth addressing
    if len(class_counts) <= 1:
        print("Only one class present. Cannot balance.")
        return df
    
    imbalance_ratio = class_counts.iloc[0] / class_counts.iloc[1]
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio < 3:
        print("Class imbalance is not severe. Returning original data.")
        return df
    
    # Make a copy of the dataframe
    df_copy = df.copy()
    
    # Convert date/time columns to numeric before resampling
    date_columns = []
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            # Check if column appears to be a date
            if df_copy[col].astype(str).str.contains('-').any():
                try:
                    # Try parsing a few values as datetime
                    sample = df_copy[col].dropna().iloc[0]
                    pd.to_datetime(sample)
                    date_columns.append(col)
                    print(f"Converting date column '{col}' to numeric for resampling")
                except:
                    pass
    
    # Convert date columns to numeric features
    for col in date_columns:
        try:
            # Extract numeric components
            datetime_series = pd.to_datetime(df_copy[col], errors='coerce')
            # Create year, month, day columns
            col_prefix = f"{col}_"
            df_copy[f"{col_prefix}year"] = datetime_series.dt.year
            df_copy[f"{col_prefix}month"] = datetime_series.dt.month
            df_copy[f"{col_prefix}day"] = datetime_series.dt.day
            # Drop the original column
            df_copy = df_copy.drop(columns=[col])
            print(f"Extracted date features from '{col}'")
        except Exception as e:
            print(f"Error converting date column '{col}': {e}")
    
    # Convert remaining categorical columns to numeric
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            print(f"Converting categorical column '{col}' to numeric")
            # Use category codes for categorical variables
            df_copy[col] = pd.Categorical(df_copy[col]).codes
    
    # Split features and target
    X = df_copy.drop('is_fraud', axis=1)
    y = df_copy['is_fraud']
    
    # Try the specified method
    methods = ['smote', 'adasyn', 'random'] if method == 'auto' else [method.lower()]
    
    for method_name in methods:
        try:
            if method_name == 'smote':
                from imblearn.over_sampling import SMOTE
                resampler = SMOTE(random_state=42)
            elif method_name == 'adasyn':
                from imblearn.over_sampling import ADASYN
                resampler = ADASYN(random_state=42)
            elif method_name == 'random':
                from imblearn.over_sampling import RandomOverSampler
                resampler = RandomOverSampler(random_state=42)
            else:
                print(f"Unknown method: {method_name}")
                continue
                
            print(f"Attempting to balance classes using {method_name.upper()}...")
            X_resampled, y_resampled = resampler.fit_resample(X, y)
            
            # Reconstruct DataFrame
            df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            df_resampled['is_fraud'] = y_resampled
            
            print(f"Class distribution after {method_name.upper()}:")
            print(df_resampled['is_fraud'].value_counts())
            return df_resampled
        except Exception as e:
            print(f"Error using {method_name.upper()}: {e}")
    
    # If all techniques failed, return the original dataframe
    print("All balancing techniques failed. Returning original imbalanced data.")
    return df

def remove_collinear_features(df, threshold=0.95):
    """
    Remove highly correlated features to prevent data leakage and multicollinearity.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset with features
    threshold : float, default=0.95
        Correlation threshold above which to remove features
    
    Returns:
    --------
    pandas DataFrame
        Dataset with collinear features removed
    """
    print(f"Checking for collinear features (threshold={threshold})...")
    
    # Get only numeric columns for correlation calculation
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # If we have a target column, exclude it from feature removal
    target_cols = ['is_fraud', 'Class'] 
    numeric_features = [col for col in numeric_cols if col not in target_cols]
    
    # Skip if we don't have enough numeric features
    if len(numeric_features) < 2:
        print("Not enough numeric features to check for collinearity")
        return df
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_features].corr().abs()
    
    # Create a mask for the upper triangle of the correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    if to_drop:
        print(f"Removing {len(to_drop)} collinear features: {', '.join(to_drop)}")
        
        # For each feature to drop, print what it's highly correlated with
        for col in to_drop:
            correlated_features = [
                f"{other_col} ({corr_matrix.loc[col, other_col]:.2f})" 
                for other_col in numeric_features 
                if other_col != col and corr_matrix.loc[col, other_col] > threshold
            ]
            print(f"  {col} is highly correlated with: {', '.join(correlated_features)}")
        
        # Drop the identified collinear features
        df_filtered = df.drop(columns=to_drop)
        print(f"Reduced feature count from {df.shape[1]} to {df_filtered.shape[1]}")
        return df_filtered
    else:
        print("No collinear features found")
        return df

def check_target_leakage(df, target_col='is_fraud'):
    """
    Check for features that may cause target leakage.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset with features and target
    target_col : str
        The name of the target column
        
    Returns:
    --------
    list
        List of potentially leaky features
    """
    if target_col not in df.columns:
        print("Target column not found in dataframe")
        return []
        
    print("\nChecking for potential target leakage...")
    
    # Get correlation with target - only for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    # Calculate correlations
    high_corr_features = []
    leaky_threshold = 0.7  # Threshold for suspiciously high correlation
    
    if len(numeric_cols) > 0:
        corrs = {}
        for col in numeric_cols:
            if col != target_col:
                corr = abs(df[col].corr(df[target_col]))
                corrs[col] = corr
                if corr > leaky_threshold:
                    high_corr_features.append((col, corr))
    
        # Sort by correlation and print warnings
        if high_corr_features:
            print("WARNING: The following features have suspiciously high correlation with the target:")
            for feature, corr in sorted(high_corr_features, key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {corr:.4f} - Consider investigating or removing this feature")
                
        # Check for index columns or monotonically increasing features
        for col in numeric_cols:
            if df[col].is_monotonic_increasing or (df[col] == df.index).all():
                print(f"WARNING: {col} is monotonically increasing or matches index - this may indicate an ID column")
                high_corr_features.append((col, 1.0))
                
        # Check near duplicate rows which can cause test set leakage
        duplicate_threshold = 0.9
        duplicate_rows = df.duplicated(subset=numeric_cols, keep=False).sum()
        if duplicate_rows > 0:
            duplicate_pct = duplicate_rows / len(df) * 100
            print(f"WARNING: Found {duplicate_rows} duplicate rows ({duplicate_pct:.2f}%) which may lead to test set leakage")
    
    # Also check direct target leakage from columns that might directly reveal the target
    potential_leaky_names = ['fraud', 'is_fraud', 'target', 'label', 'class', 'flag', 'warning', 'risk', 'score']
    name_leakage = []
    
    for col in df.columns:
        if col == target_col:
            continue
            
        # Check if column name contains suspicious terms
        if any(leaky_term in col.lower() for leaky_term in potential_leaky_names):
            name_leakage.append(col)
    
    if name_leakage:
        print("WARNING: The following columns have suspicious names that suggest potential leakage:")
        for col in name_leakage:
            print(f"  {col}")
        high_corr_features.extend([(col, 1.0) for col in name_leakage])
    
    # Look for unnamed or index-like columns
    unnamed_cols = [col for col in df.columns if 'unnamed' in col.lower() or 'index' in col.lower()]
    if unnamed_cols:
        print("WARNING: The following columns appear to be index or unnamed columns that may cause leakage:")
        for col in unnamed_cols:
            print(f"  {col}")
        high_corr_features.extend([(col, 1.0) for col in unnamed_cols])
    
    # Combine and return all suspicious features
    leaky_features = [feature for feature, _ in high_corr_features]
    return leaky_features

def main():
    """Main function to run the feature engineering process."""
    print_header("SecurePay Fraud Detection - Feature Engineering")
    
    # Load the data
    df = load_raw_data()
    if df is None:
        print("Data loading failed. Exiting.")
        return
    
    # Check data quality and remove collinear features in one step
    df = check_data_quality(df)
    df = remove_collinear_features(df)
    
    # Apply standard mappings for gender and state
    df = apply_mappings(df)
    
    # Check for potential target leakage
    leaky_features = check_target_leakage(df)
    if leaky_features:
        print(f"\nRemoving {len(leaky_features)} features suspected of target leakage")
        df = df.drop(columns=leaky_features, errors='ignore')
    
    # Balance the classes
    df_balanced = handle_class_imbalance(df)
    
    # Create additional features
    df_engineered = create_additional_features(df_balanced)
    
    # Perform correlation analysis
    print("\nPerforming correlation analysis...")
    numeric_cols = df_engineered.select_dtypes(include=['number']).columns
    print(f"Using {len(numeric_cols)} numeric features for correlation analysis")
    
    target_col = 'is_fraud'
    if target_col in numeric_cols:
        corr_matrix = df_engineered[numeric_cols].corr()
        try:
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                        fmt='.2f', square=True, linewidths=0.5)
            plt.title('Feature Correlation Matrix')
            os.makedirs('models/figures', exist_ok=True)
            plt.savefig('models/figures/correlation_matrix.png')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create correlation plot: {e}")
        
        # Get correlations with target
        target_corr = corr_matrix[target_col].sort_values(ascending=False)
        print(f"\nTop correlations with {target_col}:")
        print(target_corr.head(10))
    else:
        print(f"Warning: Target column '{target_col}' is not numeric and was excluded from correlation analysis")
    
    # Select features
    selected_df, selected_features = select_features(df_engineered, target_col=target_col, k=15)
    
    # Save the processed data
    print("\nSaving processed data to selected_features.csv...")
    os.makedirs('data/processed', exist_ok=True)
    selected_df.to_csv('data/processed/selected_features.csv', index=False)
    
    # Perform PCA
    try:
        print("\nPerforming PCA with 2 components...")
        numeric_df = df_engineered.select_dtypes(include=['number'])
        if target_col in numeric_df.columns:
            X = numeric_df.drop(target_col, axis=1)
            y = numeric_df[target_col]
            
            # First standardize the data for PCA
            from sklearn.preprocessing import StandardScaler
            X_scaled = StandardScaler().fit_transform(X)
            
            # Then apply PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create PCA plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
            plt.colorbar(scatter, label=target_col)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('PCA of Fraud Detection Features')
            plt.savefig('models/figures/pca_visualization.png')
            plt.close()
            
            print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        else:
            print(f"Warning: Target column '{target_col}' not in numeric columns")
    except Exception as e:
        print(f"Error during PCA: {e}")
    
    # Save the fully engineered data
    print("\nSaving processed data to engineered_features.csv...")
    df_engineered.to_csv('data/processed/engineered_features.csv', index=False)
    
    print("\nFeature engineering process completed.")

if __name__ == "__main__":
    main() 