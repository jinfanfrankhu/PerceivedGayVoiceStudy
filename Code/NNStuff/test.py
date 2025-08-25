import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class NeuralNetworkPipeline:
    def __init__(self, csv_file_path, target_column, task_type='classification', 
                 input_size=50, random_state=42):
        """
        Initialize the Neural Network Pipeline
        
        Args:
            csv_file_path (str): Path to the CSV file
            target_column (str): Name of the target column
            task_type (str): 'classification' or 'regression'
            input_size (int): Expected input size (will be adjusted based on actual data)
            random_state (int): Random state for reproducibility
        """
        self.csv_file_path = csv_file_path
        self.target_column = target_column
        self.task_type = task_type
        self.input_size = input_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder() if task_type == 'classification' else None
        
    def load_data(self):
        """Load data from CSV file"""
        print("Loading data from CSV...")
        self.df = pd.read_csv(self.csv_file_path)
        print(f"Data shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Handle missing values
        if self.df.isnull().sum().sum() > 0:
            print("Handling missing values...")
            # For numeric columns, fill with mean
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
            
            # For categorical columns, fill with mode
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != self.target_column:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        return self.df
    
    def preprocess_data(self):
        """Preprocess the data for neural network training"""
        print("Preprocessing data...")
        
        # Separate features and target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Handle categorical features (one-hot encoding)
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"One-hot encoding categorical columns: {list(categorical_cols)}")
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Update input size based on actual feature count
        self.actual_input_size = X.shape[1]
        print(f"Actual input size: {self.actual_input_size} (originally set to {self.input_size})")
        
        # Encode target variable if classification
        if self.task_type == 'classification':
            y = self.label_encoder.fit_transform(y)
            self.num_classes = len(np.unique(y))
            print(f"Number of classes: {self.num_classes}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def create_model(self):
        """Create a neural network model"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.actual_input_size,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
        ])
        
        if self.task_type == 'classification':
            if self.num_classes == 2:
                # Binary classification
                model.add(layers.Dense(1, activation='sigmoid'))
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
            else:
                # Multi-class classification
                model.add(layers.Dense(self.num_classes, activation='softmax'))
                model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
        else:
            # Regression
            model.add(layers.Dense(1, activation='linear'))
            model.compile(
                optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae']
            )
        
        return model
    
    def perform_cross_validation(self, X, y, n_splits=5):
        """Perform k-fold cross validation"""
        print(f"Performing {n_splits}-fold cross validation...")
        
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        cv_scores = []
        fold_histories = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\nTraining Fold {fold + 1}/{n_splits}")
            
            # Split data for this fold
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Create and train model
            model = self.create_model()
            
            # Early stopping callback
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=100,
                batch_size=32,
                verbose=0,
                callbacks=[early_stopping]
            )
            
            # Evaluate fold
            if self.task_type == 'classification':
                y_pred = model.predict(X_val_fold)
                if self.num_classes == 2:
                    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
                    score = accuracy_score(y_val_fold, y_pred_binary)
                else:
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    score = accuracy_score(y_val_fold, y_pred_classes)
                print(f"Fold {fold + 1} Accuracy: {score:.4f}")
            else:
                y_pred = model.predict(X_val_fold)
                score = r2_score(y_val_fold, y_pred)
                print(f"Fold {fold + 1} R² Score: {score:.4f}")
            
            cv_scores.append(score)
            fold_histories.append(history.history)
        
        print(f"\nCross-validation results:")
        print(f"Mean Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        
        return cv_scores, fold_histories
    
    def train_final_model(self, X, y, test_size=0.2):
        """Train final model on train-test split"""
        print(f"\nTraining final model with {int((1-test_size)*100)}-{int(test_size*100)} train-test split...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y if self.task_type == 'classification' else None
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Create and train final model
        self.final_model = self.create_model()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train model
        history = self.final_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=150,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Final evaluation
        print("\nFinal Model Evaluation:")
        if self.task_type == 'classification':
            y_pred = self.final_model.predict(X_test)
            if self.num_classes == 2:
                y_pred_binary = (y_pred > 0.5).astype(int).flatten()
                accuracy = accuracy_score(y_test, y_pred_binary)
                print(f"Test Accuracy: {accuracy:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred_binary))
            else:
                y_pred_classes = np.argmax(y_pred, axis=1)
                accuracy = accuracy_score(y_test, y_pred_classes)
                print(f"Test Accuracy: {accuracy:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred_classes))
        else:
            y_pred = self.final_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"Test MSE: {mse:.4f}")
            print(f"Test R² Score: {r2:.4f}")
        
        return history, X_test, y_test
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot metrics
        if self.task_type == 'classification':
            axes[1].plot(history.history['accuracy'], label='Training Accuracy')
            axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axes[1].set_ylabel('Accuracy')
        else:
            axes[1].plot(history.history['mae'], label='Training MAE')
            axes[1].plot(history.history['val_mae'], label='Validation MAE')
            axes[1].set_ylabel('MAE')
        
        axes[1].set_title('Model Performance')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        # Load data
        self.load_data()
        
        # Preprocess data
        X, y = self.preprocess_data()
        
        # Perform cross-validation
        cv_scores, fold_histories = self.perform_cross_validation(X, y)
        
        # Train final model
        history, X_test, y_test = self.train_final_model(X, y)
        
        # Plot results
        self.plot_training_history(history)
        
        return {
            'cv_scores': cv_scores,
            'final_model': self.final_model,
            'history': history,
            'test_data': (X_test, y_test)
        }

# Example usage:
if __name__ == "__main__":
    # Initialize the pipeline
    # Replace 'your_data.csv' with your actual CSV file path
    # Replace 'target_column_name' with your actual target column name
    
    pipeline = NeuralNetworkPipeline(
        csv_file_path='your_data.csv',  # Update this path
        target_column='target',         # Update this column name
        task_type='classification',     # or 'regression'
        input_size=50,                  # This will be automatically adjusted
        random_state=42
    )
    
    # Run the complete pipeline
    results = pipeline.run_pipeline()
    
    print("\nPipeline completed successfully!")
    print(f"Cross-validation scores: {results['cv_scores']}")
    print(f"Mean CV score: {np.mean(results['cv_scores']):.4f}")