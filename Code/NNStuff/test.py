import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class VoiceSexualityPerceptionPipeline:
    def __init__(self, random_state=42):
        """
        Initialize the Voice-Based Sexuality Perception Pipeline
        
        This pipeline handles two separate neural networks:
        1. Acoustic features → Self-reported sexuality (1-10 scale)
        2. Acoustic features → Perceived sexuality (1-10 scale)
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler_acoustic = StandardScaler()
        self.scaler_sexuality = MinMaxScaler(feature_range=(0, 1))  # For 1-10 scale
        
    def load_acoustic_data(self, csv_file_path, sexuality_column='self_reported_sexuality', 
                          speaker_id_column='speaker_id'):
        """
        Load acoustic features with self-reported sexuality data
        
        Args:
            csv_file_path (str): Path to CSV with acoustic features and self-reported sexuality
            sexuality_column (str): Column name for self-reported sexuality (1-10 scale)
            speaker_id_column (str): Column name for speaker identification
        """
        print("Loading acoustic features data...")
        self.acoustic_df = pd.read_csv(csv_file_path)
        self.sexuality_column = sexuality_column
        self.speaker_id_column = speaker_id_column
        
        print(f"Acoustic data shape: {self.acoustic_df.shape}")
        print(f"Sexuality range: {self.acoustic_df[sexuality_column].min()} to {self.acoustic_df[sexuality_column].max()}")
        print(f"Number of unique speakers: {self.acoustic_df[speaker_id_column].nunique()}")
        
        # Handle missing values in acoustic features
        acoustic_cols = [col for col in self.acoustic_df.columns 
                        if col not in [sexuality_column, speaker_id_column]]
        
        print(f"Number of acoustic features: {len(acoustic_cols)}")
        
        if self.acoustic_df[acoustic_cols].isnull().sum().sum() > 0:
            print("Handling missing values in acoustic features...")
            self.acoustic_df[acoustic_cols] = self.acoustic_df[acoustic_cols].fillna(
                self.acoustic_df[acoustic_cols].median()
            )
        
        return self.acoustic_df
    
    def load_perception_data(self, csv_file_path, perceived_sexuality_column='perceived_sexuality',
                           rater_id_column='rater_id', speaker_id_column='speaker_id'):
        """
        Load perception data (raters' judgments of speaker sexuality)
        
        Args:
            csv_file_path (str): Path to CSV with perception data
            perceived_sexuality_column (str): Column name for perceived sexuality ratings
            rater_id_column (str): Column name for rater identification
            speaker_id_column (str): Column name for speaker identification
        """
        print("Loading perception data...")
        self.perception_df = pd.read_csv(csv_file_path)
        self.perceived_sexuality_column = perceived_sexuality_column
        self.rater_id_column = rater_id_column
        
        print(f"Perception data shape: {self.perception_df.shape}")
        print(f"Number of unique raters: {self.perception_df[rater_id_column].nunique()}")
        print(f"Number of unique speakers rated: {self.perception_df[speaker_id_column].nunique()}")
        print(f"Perceived sexuality range: {self.perception_df[perceived_sexuality_column].min()} to {self.perception_df[perceived_sexuality_column].max()}")
        
        # Calculate average perceived sexuality per speaker
        print("Calculating average perceived sexuality per speaker...")
        self.avg_perception_df = self.perception_df.groupby(speaker_id_column).agg({
            perceived_sexuality_column: ['mean', 'std', 'count']
        }).round(3)
        
        self.avg_perception_df.columns = ['avg_perceived_sexuality', 'std_perceived_sexuality', 'num_raters']
        self.avg_perception_df = self.avg_perception_df.reset_index()
        
        print(f"Average perceived sexuality range: {self.avg_perception_df['avg_perceived_sexuality'].min():.2f} to {self.avg_perception_df['avg_perceived_sexuality'].max():.2f}")
        
        return self.perception_df, self.avg_perception_df
    
    def merge_datasets(self):
        """Merge acoustic features with averaged perception data"""
        print("Merging acoustic and perception datasets...")
        
        self.merged_df = pd.merge(
            self.acoustic_df, 
            self.avg_perception_df, 
            on=self.speaker_id_column, 
            how='inner'
        )
        
        print(f"Merged dataset shape: {self.merged_df.shape}")
        print(f"Speakers with both acoustic and perception data: {len(self.merged_df)}")
        
        return self.merged_df
    
    def prepare_data_for_model(self, target_type='self_reported'):
        """
        Prepare data for neural network training
        
        Args:
            target_type (str): 'self_reported' or 'perceived'
        """
        print(f"Preparing data for {target_type} sexuality prediction...")
        
        # Select acoustic features (exclude metadata and target columns)
        exclude_cols = [self.speaker_id_column, self.sexuality_column, 
                       'avg_perceived_sexuality', 'std_perceived_sexuality', 'num_raters']
        
        acoustic_features = [col for col in self.merged_df.columns if col not in exclude_cols]
        
        X = self.merged_df[acoustic_features].values
        
        if target_type == 'self_reported':
            y = self.merged_df[self.sexuality_column].values
        else:  # perceived
            y = self.merged_df['avg_perceived_sexuality'].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Target range: {y.min():.2f} to {y.max():.2f}")
        
        # Scale features
        X_scaled = self.scaler_acoustic.fit_transform(X)
        
        # Normalize sexuality scores to 0-1 range for better training
        y_scaled = (y - 1) / 9  # Convert 1-10 scale to 0-1 scale
        
        return X_scaled, y_scaled, y  # Return both scaled and original targets
    
    def create_sexuality_model(self, input_size, model_complexity='medium'):
        """
        Create neural network model for sexuality prediction
        
        Args:
            input_size (int): Number of input features
            model_complexity (str): 'simple', 'medium', or 'complex'
        """
        if model_complexity == 'simple':
            model = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(input_size,)),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation='sigmoid')  # Sigmoid for 0-1 range
            ])
        elif model_complexity == 'medium':
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(input_size,)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
        else:  # complex
            model = keras.Sequential([
                layers.Dense(256, activation='relu', input_shape=(input_size,)),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
        
        # Use MAE as primary metric since it's more interpretable for 1-10 scale
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def perform_cross_validation(self, X, y, y_original, target_type, n_splits=5, 
                               model_complexity='medium'):
        """Perform k-fold cross validation"""
        print(f"\nPerforming {n_splits}-fold cross validation for {target_type} sexuality...")
        
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        cv_scores = {'mae': [], 'rmse': [], 'r2': [], 'corr': []}
        fold_histories = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\nTraining Fold {fold + 1}/{n_splits}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            y_val_original = y_original[val_idx]
            
            # Create and train model
            model = self.create_sexuality_model(X.shape[1], model_complexity)
            
            # Callbacks
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True
            )
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
            )
            
            # Train model
            history = model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=200,
                batch_size=min(32, len(X_train_fold) // 4),  # Adaptive batch size
                verbose=0,
                callbacks=[early_stopping, reduce_lr]
            )
            
            # Evaluate fold
            y_pred_scaled = model.predict(X_val_fold, verbose=0)
            y_pred_original = y_pred_scaled.flatten() * 9 + 1  # Convert back to 1-10 scale
            
            # Calculate metrics
            mae = mean_absolute_error(y_val_original, y_pred_original)
            rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))
            r2 = r2_score(y_val_original, y_pred_original)
            corr, _ = pearsonr(y_val_original, y_pred_original)
            
            cv_scores['mae'].append(mae)
            cv_scores['rmse'].append(rmse)
            cv_scores['r2'].append(r2)
            cv_scores['corr'].append(corr)
            fold_histories.append(history.history)
            
            print(f"Fold {fold + 1} - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}, Corr: {corr:.3f}")
        
        # Print summary
        print(f"\n{target_type.upper()} Cross-validation Results:")
        for metric, scores in cv_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"{metric.upper()}: {mean_score:.3f} (±{std_score:.3f})")
        
        return cv_scores, fold_histories
    
    def train_final_model(self, X, y, y_original, target_type, test_size=0.2, 
                         model_complexity='medium'):
        """Train final model on train-test split"""
        print(f"\nTraining final {target_type} model with 80-20 split...")
        
        X_train, X_test, y_train, y_test, y_test_orig = train_test_split(
            X, y, y_original, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Create and train final model
        final_model = self.create_sexuality_model(X.shape[1], model_complexity)
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=25, restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=12, min_lr=1e-6
        )
        
        # Train model
        history = final_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=300,
            batch_size=min(32, len(X_train) // 4),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Final evaluation
        y_pred_scaled = final_model.predict(X_test, verbose=0)
        y_pred_orig = y_pred_scaled.flatten() * 9 + 1
        
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        r2 = r2_score(y_test_orig, y_pred_orig)
        corr, p_value = pearsonr(y_test_orig, y_pred_orig)
        
        print(f"\nFinal {target_type.upper()} Model Results:")
        print(f"Test MAE: {mae:.3f}")
        print(f"Test RMSE: {rmse:.3f}")
        print(f"Test R²: {r2:.3f}")
        print(f"Test Correlation: {corr:.3f} (p={p_value:.4f})")
        
        return final_model, history, (X_test, y_test_orig, y_pred_orig)
    
    def compare_models(self, self_reported_results, perceived_results):
        """Compare self-reported vs perceived sexuality prediction models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON: SELF-REPORTED vs PERCEIVED SEXUALITY")
        print("="*60)
        
        # Extract test results
        _, y_self_true, y_self_pred = self_reported_results[2]
        _, y_perc_true, y_perc_pred = perceived_results[2]
        
        # Calculate correlation between actual self-reported and perceived ratings
        actual_corr, p_val = pearsonr(y_self_true, y_perc_true)
        print(f"Correlation between actual self-reported and perceived: {actual_corr:.3f} (p={p_val:.4f})")
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Self-reported predictions
        axes[0,0].scatter(y_self_true, y_self_pred, alpha=0.6, color='blue')
        axes[0,0].plot([1, 10], [1, 10], 'r--', alpha=0.8)
        axes[0,0].set_xlabel('Actual Self-Reported Sexuality')
        axes[0,0].set_ylabel('Predicted Self-Reported Sexuality')
        axes[0,0].set_title('Self-Reported Sexuality Prediction')
        axes[0,0].grid(True, alpha=0.3)
        
        # Perceived predictions
        axes[0,1].scatter(y_perc_true, y_perc_pred, alpha=0.6, color='green')
        axes[0,1].plot([1, 10], [1, 10], 'r--', alpha=0.8)
        axes[0,1].set_xlabel('Actual Perceived Sexuality')
        axes[0,1].set_ylabel('Predicted Perceived Sexuality')
        axes[0,1].set_title('Perceived Sexuality Prediction')
        axes[0,1].grid(True, alpha=0.3)
        
        # Self-reported vs Perceived (actual)
        axes[1,0].scatter(y_self_true, y_perc_true, alpha=0.6, color='purple')
        axes[1,0].plot([1, 10], [1, 10], 'r--', alpha=0.8)
        axes[1,0].set_xlabel('Self-Reported Sexuality')
        axes[1,0].set_ylabel('Perceived Sexuality (Avg)')
        axes[1,0].set_title(f'Self-Reported vs Perceived\n(r={actual_corr:.3f})')
        axes[1,0].grid(True, alpha=0.3)
        
        # Prediction errors comparison
        self_errors = np.abs(y_self_true - y_self_pred)
        perc_errors = np.abs(y_perc_true - y_perc_pred)
        
        axes[1,1].hist(self_errors, bins=20, alpha=0.6, label='Self-Reported', color='blue')
        axes[1,1].hist(perc_errors, bins=20, alpha=0.6, label='Perceived', color='green')
        axes[1,1].set_xlabel('Absolute Prediction Error')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Prediction Error Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return actual_corr
    
    def analyze_rater_agreement(self):
        """Analyze agreement between raters"""
        if not hasattr(self, 'perception_df'):
            print("No perception data loaded for rater agreement analysis")
            return
        
        print("\n" + "="*50)
        print("RATER AGREEMENT ANALYSIS")
        print("="*50)
        
        # Calculate inter-rater statistics
        speaker_ratings = self.perception_df.groupby(self.speaker_id_column)[self.perceived_sexuality_column].agg(['mean', 'std', 'count'])
        
        print(f"Average standard deviation across speakers: {speaker_ratings['std'].mean():.3f}")
        print(f"Speakers with high disagreement (std > 2): {(speaker_ratings['std'] > 2).sum()}")
        print(f"Average number of raters per speaker: {speaker_ratings['count'].mean():.1f}")
        
        # Plot rater agreement
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(speaker_ratings['std'].dropna(), bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Standard Deviation of Ratings')
        plt.ylabel('Number of Speakers')
        plt.title('Inter-Rater Agreement Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(speaker_ratings['mean'], speaker_ratings['std'], alpha=0.6)
        plt.xlabel('Mean Perceived Sexuality')
        plt.ylabel('Standard Deviation')
        plt.title('Mean Rating vs Agreement')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return speaker_ratings
    
    def run_complete_analysis(self, acoustic_csv, perception_csv, 
                            acoustic_sexuality_col='self_reported_sexuality',
                            perceived_sexuality_col='perceived_sexuality',
                            speaker_id_col='speaker_id',
                            rater_id_col='rater_id'):
        """Run the complete analysis pipeline"""
        print("="*70)
        print("VOICE-BASED SEXUALITY PERCEPTION ANALYSIS")
        print("="*70)
        
        # Load data
        self.load_acoustic_data(acoustic_csv, acoustic_sexuality_col, speaker_id_col)
        self.load_perception_data(perception_csv, perceived_sexuality_col, rater_id_col, speaker_id_col)
        
        # Analyze rater agreement
        self.analyze_rater_agreement()
        
        # Merge datasets
        self.merge_datasets()
        
        results = {}
        
        # Model 1: Acoustic features → Self-reported sexuality
        print("\n" + "="*50)
        print("MODEL 1: PREDICTING SELF-REPORTED SEXUALITY")
        print("="*50)
        
        X_self, y_self_scaled, y_self_orig = self.prepare_data_for_model('self_reported')
        cv_scores_self, _ = self.perform_cross_validation(X_self, y_self_scaled, y_self_orig, 'self_reported')
        model_self, history_self, test_results_self = self.train_final_model(
            X_self, y_self_scaled, y_self_orig, 'self_reported'
        )
        
        results['self_reported'] = {
            'cv_scores': cv_scores_self,
            'model': model_self,
            'history': history_self,
            'test_results': test_results_self
        }
        
        # Model 2: Acoustic features → Perceived sexuality
        print("\n" + "="*50)
        print("MODEL 2: PREDICTING PERCEIVED SEXUALITY")
        print("="*50)
        
        X_perc, y_perc_scaled, y_perc_orig = self.prepare_data_for_model('perceived')
        cv_scores_perc, _ = self.perform_cross_validation(X_perc, y_perc_scaled, y_perc_orig, 'perceived')
        model_perc, history_perc, test_results_perc = self.train_final_model(
            X_perc, y_perc_scaled, y_perc_orig, 'perceived'
        )
        
        results['perceived'] = {
            'cv_scores': cv_scores_perc,
            'model': model_perc,
            'history': history_perc,
            'test_results': test_results_perc
        }
        
        # Compare models
        actual_corr = self.compare_models(
            (model_self, history_self, test_results_self),
            (model_perc, history_perc, test_results_perc)
        )
        
        results['comparison'] = {'self_vs_perceived_correlation': actual_corr}
        
        print(f"\nAnalysis complete! Key finding: Self-reported vs Perceived correlation = {actual_corr:.3f}")
        
        return results

# Example usage:
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = VoiceSexualityPerceptionPipeline(random_state=42)
    
    # Run complete analysis
    # Update these paths with your actual data files
    results = pipeline.run_complete_analysis(
        acoustic_csv='acoustic_features.csv',      # CSV with acoustic features + self-reported sexuality
        perception_csv='perception_ratings.csv',   # CSV with rater judgments
        acoustic_sexuality_col='self_reported_sexuality',  # 1-10 scale
        perceived_sexuality_col='perceived_sexuality',     # 1-10 scale  
        speaker_id_col='speaker_id',
        rater_id_col='rater_id'
    )