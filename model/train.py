# Enhanced version with comprehensive cross-validation and advanced feature engineering

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    StratifiedKFold, validation_curve, cross_validate
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import joblib
import hashlib
import sys
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any, List
import warnings
import re
warnings.filterwarnings('ignore')

# Import enhanced feature engineering components
try:
    from features.feature_engineer import AdvancedFeatureEngineer, create_enhanced_pipeline, analyze_feature_importance
    from features.sentiment_analyzer import SentimentAnalyzer
    from features.readability_analyzer import ReadabilityAnalyzer
    from features.entity_analyzer import EntityAnalyzer
    from features.linguistic_analyzer import LinguisticAnalyzer
    ENHANCED_FEATURES_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Enhanced feature engineering components loaded successfully")
except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Enhanced features not available, falling back to basic TF-IDF: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def preprocess_text_function(texts):
    """
    Standalone function for text preprocessing - pickle-safe
    """
    def clean_single_text(text):
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Remove non-alphabetic characters except spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z\s.!?]', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip().lower()
    
    # Process all texts
    processed = []
    for text in texts:
        processed.append(clean_single_text(text))
    
    return processed


class ProgressTracker:
    """Progress tracking with time estimation"""
    
    def __init__(self, total_steps: int, description: str = "Training"):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.description = description
        self.step_times = []
        
    def update(self, step_name: str = ""):
        """Update progress and print status"""
        self.current_step += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calculate progress percentage
        progress_pct = (self.current_step / self.total_steps) * 100
        
        # Estimate remaining time
        if self.current_step > 0:
            avg_time_per_step = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = avg_time_per_step * remaining_steps
            eta = timedelta(seconds=int(eta_seconds))
        else:
            eta = "calculating..."
            
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * self.current_step // self.total_steps)
        bar = '█' * filled_length + '▒' * (bar_length - filled_length)
        
        # Print progress (this will be visible in Streamlit logs)
        status_msg = f"\r{self.description}: [{bar}] {progress_pct:.1f}% | Step {self.current_step}/{self.total_steps}"
        if step_name:
            status_msg += f" | {step_name}"
        if eta != "calculating...":
            status_msg += f" | ETA: {eta}"
            
        print(status_msg, end='', flush=True)
        
        # Also output JSON for Streamlit parsing (if needed)
        progress_json = {
            "type": "progress",
            "step": self.current_step,
            "total": self.total_steps,
            "percentage": progress_pct,
            "eta": str(eta) if eta != "calculating..." else None,
            "step_name": step_name,
            "elapsed": elapsed
        }
        print(f"\nPROGRESS_JSON: {json.dumps(progress_json)}")
        
        # Store step time for better estimation
        if len(self.step_times) >= 3:  # Keep last 3 step times for moving average
            self.step_times.pop(0)
        self.step_times.append(current_time - (self.start_time + sum(self.step_times)))
        
    def finish(self):
        """Complete progress tracking"""
        total_time = time.time() - self.start_time
        print(f"\n{self.description} completed in {timedelta(seconds=int(total_time))}")


def estimate_training_time(dataset_size: int, enable_tuning: bool = True, cv_folds: int = 5, 
                          use_enhanced_features: bool = False) -> Dict:
    """Estimate training time based on dataset characteristics and feature complexity"""
    
    # Base time estimates (in seconds) based on empirical testing
    base_times = {
        'preprocessing': max(0.1, dataset_size * 0.001),  # ~1ms per sample
        'vectorization': max(0.5, dataset_size * 0.01),   # ~10ms per sample
        'feature_selection': max(0.2, dataset_size * 0.005), # ~5ms per sample
        'simple_training': max(1.0, dataset_size * 0.02),  # ~20ms per sample
        'evaluation': max(0.5, dataset_size * 0.01),       # ~10ms per sample
    }
    
    # Enhanced feature engineering time multipliers
    if use_enhanced_features:
        base_times['preprocessing'] *= 2.5  # More complex preprocessing
        base_times['vectorization'] *= 1.5  # Additional feature extraction
        base_times['feature_selection'] *= 2.0  # More features to select from
        base_times['enhanced_feature_extraction'] = max(2.0, dataset_size * 0.05)  # New step
    
    # Hyperparameter tuning multipliers
    tuning_multipliers = {
        'logistic_regression': 8 if enable_tuning else 1,  # 8 param combinations
        'random_forest': 12 if enable_tuning else 1,       # 12 param combinations
    }
    
    # Cross-validation multiplier
    cv_multiplier = cv_folds if dataset_size > 100 else 1
    
    # Calculate estimates
    estimates = {}
    
    # Preprocessing steps
    estimates['data_loading'] = 0.5
    estimates['preprocessing'] = base_times['preprocessing']
    estimates['vectorization'] = base_times['vectorization']
    
    if use_enhanced_features:
        estimates['enhanced_feature_extraction'] = base_times['enhanced_feature_extraction']
    
    estimates['feature_selection'] = base_times['feature_selection']
    
    # Model training (now includes CV)
    for model_name, multiplier in tuning_multipliers.items():
        model_time = base_times['simple_training'] * multiplier * cv_multiplier
        estimates[f'{model_name}_training'] = model_time
        estimates[f'{model_name}_evaluation'] = base_times['evaluation']
    
    # Cross-validation overhead
    estimates['cross_validation'] = base_times['simple_training'] * cv_folds * 0.5
    
    # Model saving
    estimates['model_saving'] = 1.0
    
    # Total estimate
    total_estimate = sum(estimates.values())
    
    # Add buffer for overhead (more for enhanced features)
    buffer_multiplier = 1.4 if use_enhanced_features else 1.2
    total_estimate *= buffer_multiplier
    
    return {
        'detailed_estimates': estimates,
        'total_seconds': total_estimate,
        'total_formatted': str(timedelta(seconds=int(total_estimate))),
        'dataset_size': dataset_size,
        'enable_tuning': enable_tuning,
        'cv_folds': cv_folds,
        'use_enhanced_features': use_enhanced_features
    }


class CrossValidationManager:
    """Advanced cross-validation management with comprehensive metrics"""
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv_results = {}
        
    def create_cv_strategy(self, X, y) -> StratifiedKFold:
        """Create appropriate CV strategy based on data characteristics"""
        # Calculate appropriate CV folds for small datasets
        n_samples = len(X)
        min_samples_per_fold = 3  # Minimum samples per fold
        max_folds = n_samples // min_samples_per_fold
        
        # Adjust folds based on data size and class distribution
        unique_classes = np.unique(y)
        min_class_count = min([np.sum(y == cls) for cls in unique_classes])
        
        # Ensure each fold has at least one sample from each class
        max_folds_by_class = min_class_count
        
        actual_folds = max(2, min(self.cv_folds, max_folds, max_folds_by_class))
        
        logger.info(f"Using {actual_folds} CV folds (requested: {self.cv_folds})")
        
        return StratifiedKFold(
            n_splits=actual_folds,
            shuffle=True,
            random_state=self.random_state
        )
    
    def perform_cross_validation(self, pipeline, X, y, cv_strategy=None) -> Dict:
        """Perform comprehensive cross-validation with multiple metrics"""
        
        if cv_strategy is None:
            cv_strategy = self.create_cv_strategy(X, y)
        
        logger.info(f"Starting cross-validation with {cv_strategy.n_splits} folds...")
        
        # Define scoring metrics
        scoring_metrics = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted',
            'roc_auc': 'roc_auc'
        }
        
        try:
            # Perform cross-validation
            cv_scores = cross_validate(
                pipeline, X, y,
                cv=cv_strategy,
                scoring=scoring_metrics,
                return_train_score=True,
                n_jobs=1,  # Use single job for stability
                verbose=0
            )
            
            # Process results
            cv_results = {
                'n_splits': cv_strategy.n_splits,
                'test_scores': {},
                'train_scores': {},
                'fold_results': []
            }
            
            # Calculate statistics for each metric
            for metric_name in scoring_metrics.keys():
                test_key = f'test_{metric_name}'
                train_key = f'train_{metric_name}'
                
                if test_key in cv_scores:
                    test_scores = cv_scores[test_key]
                    cv_results['test_scores'][metric_name] = {
                        'mean': float(np.mean(test_scores)),
                        'std': float(np.std(test_scores)),
                        'min': float(np.min(test_scores)),
                        'max': float(np.max(test_scores)),
                        'scores': test_scores.tolist()
                    }
                
                if train_key in cv_scores:
                    train_scores = cv_scores[train_key]
                    cv_results['train_scores'][metric_name] = {
                        'mean': float(np.mean(train_scores)),
                        'std': float(np.std(train_scores)),
                        'min': float(np.min(train_scores)),
                        'max': float(np.max(train_scores)),
                        'scores': train_scores.tolist()
                    }
            
            # Store individual fold results
            for fold_idx in range(cv_strategy.n_splits):
                fold_result = {
                    'fold': fold_idx + 1,
                    'test_scores': {},
                    'train_scores': {}
                }
                
                for metric_name in scoring_metrics.keys():
                    test_key = f'test_{metric_name}'
                    train_key = f'train_{metric_name}'
                    
                    if test_key in cv_scores:
                        fold_result['test_scores'][metric_name] = float(cv_scores[test_key][fold_idx])
                    if train_key in cv_scores:
                        fold_result['train_scores'][metric_name] = float(cv_scores[train_key][fold_idx])
                
                cv_results['fold_results'].append(fold_result)
            
            # Calculate overfitting indicators
            if 'accuracy' in cv_results['test_scores'] and 'accuracy' in cv_results['train_scores']:
                train_mean = cv_results['train_scores']['accuracy']['mean']
                test_mean = cv_results['test_scores']['accuracy']['mean']
                cv_results['overfitting_score'] = float(train_mean - test_mean)
            
            # Calculate stability metrics
            if 'accuracy' in cv_results['test_scores']:
                test_std = cv_results['test_scores']['accuracy']['std']
                test_mean = cv_results['test_scores']['accuracy']['mean']
                cv_results['stability_score'] = float(1 - (test_std / test_mean)) if test_mean > 0 else 0
            
            logger.info(f"Cross-validation completed successfully")
            logger.info(f"Mean test accuracy: {cv_results['test_scores'].get('accuracy', {}).get('mean', 'N/A'):.4f}")
            logger.info(f"Mean test F1: {cv_results['test_scores'].get('f1', {}).get('mean', 'N/A'):.4f}")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {
                'error': str(e),
                'n_splits': cv_strategy.n_splits if cv_strategy else self.cv_folds,
                'fallback': True
            }
    
    def compare_cv_results(self, results1: Dict, results2: Dict, metric: str = 'f1') -> Dict:
        """Compare cross-validation results between two models"""
        
        try:
            if 'error' in results1 or 'error' in results2:
                return {'error': 'Cannot compare results with errors'}
            
            scores1 = results1['test_scores'][metric]['scores']
            scores2 = results2['test_scores'][metric]['scores']
            
            # Paired t-test
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(scores1, scores2)
            
            comparison = {
                'metric': metric,
                'model1_mean': results1['test_scores'][metric]['mean'],
                'model2_mean': results2['test_scores'][metric]['mean'],
                'model1_std': results1['test_scores'][metric]['std'],
                'model2_std': results2['test_scores'][metric]['std'],
                'difference': results2['test_scores'][metric]['mean'] - results1['test_scores'][metric]['mean'],
                'paired_ttest': {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                },
                'effect_size': float(abs(t_stat) / np.sqrt(len(scores1))) if len(scores1) > 0 else 0
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"CV comparison failed: {e}")
            return {'error': str(e)}


class EnhancedModelTrainer:
    """Production-ready model trainer with enhanced feature engineering and comprehensive CV"""

    def __init__(self, use_enhanced_features: bool = None):
        # Auto-detect enhanced features if not specified
        if use_enhanced_features is None:
            self.use_enhanced_features = ENHANCED_FEATURES_AVAILABLE
        else:
            self.use_enhanced_features = use_enhanced_features and ENHANCED_FEATURES_AVAILABLE
        
        self.setup_paths()
        self.setup_training_config()
        self.setup_models()
        self.progress_tracker = None
        self.cv_manager = CrossValidationManager()
        
        # Enhanced feature tracking
        self.feature_engineer = None
        self.feature_importance_results = {}

    def setup_paths(self):
        """Setup all necessary paths with proper permissions"""
        self.base_dir = Path("/tmp")
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "model"
        self.results_dir = self.base_dir / "results"
        self.features_dir = self.base_dir / "features"  # New for enhanced features

        # Create directories with proper permissions
        for dir_path in [self.data_dir, self.model_dir, self.results_dir, self.features_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            # Ensure write permissions
            try:
                dir_path.chmod(0o755)
            except:
                pass

        # File paths
        self.data_path = self.data_dir / "combined_dataset.csv"
        self.model_path = Path("/tmp/model.pkl")
        self.vectorizer_path = Path("/tmp/vectorizer.pkl")
        self.pipeline_path = Path("/tmp/pipeline.pkl")
        self.metadata_path = Path("/tmp/metadata.json")
        self.evaluation_path = self.results_dir / "evaluation_results.json"
        
        # Enhanced feature paths
        self.feature_engineer_path = Path("/tmp/feature_engineer.pkl")
        self.feature_importance_path = self.results_dir / "feature_importance.json"

    def setup_training_config(self):
        """Setup training configuration with enhanced feature parameters"""
        self.test_size = 0.2
        self.validation_size = 0.1
        self.random_state = 42
        self.cv_folds = 5
        
        # Enhanced feature configuration
        if self.use_enhanced_features:
            self.max_features = 7500  # Increased for enhanced features
            self.feature_selection_k = 3000  # More features to select from
            logger.info("Using enhanced feature engineering pipeline")
        else:
            self.max_features = 5000  # Standard TF-IDF
            self.feature_selection_k = 2000
            logger.info("Using standard TF-IDF feature pipeline")
        
        # Common parameters
        self.min_df = 1
        self.max_df = 0.95
        self.ngram_range = (1, 2)
        self.max_iter = 500
        self.class_weight = 'balanced'

    def setup_models(self):
        """Setup model configurations for comparison"""
        self.models = {
            'logistic_regression': {
                'model': LogisticRegression(
                    max_iter=self.max_iter,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'param_grid': {
                    'model__C': [0.1, 1, 10],
                    'model__penalty': ['l2']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=50,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'param_grid': {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [10, None]
                }
            }
        }

    def load_and_validate_data(self) -> Tuple[bool, Optional[pd.DataFrame], str]:
        """Load and validate training data"""
        try:
            logger.info("Loading training data...")
            if self.progress_tracker:
                self.progress_tracker.update("Loading data")

            if not self.data_path.exists():
                return False, None, f"Data file not found: {self.data_path}"

            # Load data
            df = pd.read_csv(self.data_path)

            # Basic validation
            if df.empty:
                return False, None, "Dataset is empty"

            required_columns = ['text', 'label']
            missing_columns = [
                col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, None, f"Missing required columns: {missing_columns}"

            # Remove missing values
            initial_count = len(df)
            df = df.dropna(subset=required_columns)
            if len(df) < initial_count:
                logger.warning(
                    f"Removed {initial_count - len(df)} rows with missing values")

            # Validate text content
            df = df[df['text'].astype(str).str.len() > 10]

            # Validate labels
            unique_labels = df['label'].unique()
            if len(unique_labels) < 2:
                return False, None, f"Need at least 2 classes, found: {unique_labels}"

            # Check minimum sample size for CV
            min_samples_for_cv = self.cv_folds * 2
            if len(df) < min_samples_for_cv:
                logger.warning(f"Dataset size ({len(df)}) is small for {self.cv_folds}-fold CV")
                self.cv_manager.cv_folds = max(2, len(df) // 3)
                logger.info(f"Adjusted CV folds to {self.cv_manager.cv_folds}")

            # Check class balance
            label_counts = df['label'].value_counts()
            min_class_ratio = label_counts.min() / label_counts.max()
            if min_class_ratio < 0.1:
                logger.warning(
                    f"Severe class imbalance detected: {min_class_ratio:.3f}")

            logger.info(
                f"Data validation successful: {len(df)} samples, {len(unique_labels)} classes")
            logger.info(f"Class distribution: {label_counts.to_dict()}")

            return True, df, "Data loaded successfully"

        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg

    def create_preprocessing_pipeline(self, use_enhanced: bool = None) -> Pipeline:
        """Create preprocessing pipeline with optional enhanced features"""
        
        if use_enhanced is None:
            use_enhanced = self.use_enhanced_features
        
        if self.progress_tracker:
            feature_type = "enhanced" if use_enhanced else "standard"
            self.progress_tracker.update(f"Creating {feature_type} pipeline")
        
        if use_enhanced and ENHANCED_FEATURES_AVAILABLE:
            logger.info("Creating enhanced feature engineering pipeline...")
            
            # Create enhanced feature engineer
            feature_engineer = AdvancedFeatureEngineer(
                enable_sentiment=True,
                enable_readability=True,
                enable_entities=True,
                enable_linguistic=True,
                feature_selection_k=self.feature_selection_k,
                tfidf_max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df
            )
            
            # Create pipeline with enhanced features
            pipeline = Pipeline([
                ('enhanced_features', feature_engineer),
                ('model', None)  # Will be set during training
            ])
            
            # Store reference for later use
            self.feature_engineer = feature_engineer
            
        else:
            logger.info("Creating standard TF-IDF pipeline...")
            
            # Use the standalone function instead of lambda
            text_preprocessor = FunctionTransformer(
                func=preprocess_text_function,
                validate=False
            )

            # TF-IDF vectorization with optimized parameters
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                ngram_range=self.ngram_range,
                stop_words='english',
                sublinear_tf=True,
                norm='l2'
            )

            # Feature selection
            feature_selector = SelectKBest(
                score_func=chi2,
                k=min(self.feature_selection_k, self.max_features)
            )

            # Create standard pipeline
            pipeline = Pipeline([
                ('preprocess', text_preprocessor),
                ('vectorize', vectorizer),
                ('feature_select', feature_selector),
                ('model', None)  # Will be set during training
            ])

        return pipeline

    def comprehensive_evaluation(self, model, X_test, y_test, X_train=None, y_train=None) -> Dict:
        """Comprehensive model evaluation with enhanced feature analysis"""
        
        if self.progress_tracker:
            self.progress_tracker.update("Evaluating model")
            
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Basic metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted')),
            'f1': float(f1_score(y_test, y_pred, average='weighted')),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Cross-validation on full dataset
        if X_train is not None and y_train is not None:
            # Combine train and test for full dataset CV
            X_full = np.concatenate([X_train, X_test])
            y_full = np.concatenate([y_train, y_test])
            
            logger.info("Performing cross-validation on full dataset...")
            cv_results = self.cv_manager.perform_cross_validation(model, X_full, y_full)
            metrics['cross_validation'] = cv_results
            
            # Log CV results
            if 'test_scores' in cv_results and 'f1' in cv_results['test_scores']:
                cv_f1_mean = cv_results['test_scores']['f1']['mean']
                cv_f1_std = cv_results['test_scores']['f1']['std']
                logger.info(f"CV F1 Score: {cv_f1_mean:.4f} (±{cv_f1_std:.4f})")
        
        # Enhanced feature analysis
        if self.use_enhanced_features and self.feature_engineer is not None:
            try:
                # Get feature importance if available
                if hasattr(self.feature_engineer, 'get_feature_importance'):
                    feature_importance = self.feature_engineer.get_feature_importance(top_k=20)
                    metrics['top_features'] = feature_importance
                
                # Get feature metadata
                if hasattr(self.feature_engineer, 'get_feature_metadata'):
                    feature_metadata = self.feature_engineer.get_feature_metadata()
                    metrics['feature_metadata'] = feature_metadata
                    
                    logger.info(f"Enhanced features used: {feature_metadata['total_features']}")
                    logger.info(f"Feature breakdown: {feature_metadata['feature_types']}")
                
            except Exception as e:
                logger.warning(f"Enhanced feature analysis failed: {e}")
        
        # Training accuracy for overfitting detection
        try:
            if X_train is not None and y_train is not None:
                y_train_pred = model.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                metrics['train_accuracy'] = float(train_accuracy)
                metrics['overfitting_score'] = float(
                    train_accuracy - metrics['accuracy'])
        except Exception as e:
            logger.warning(f"Overfitting detection failed: {e}")

        return metrics

    def hyperparameter_tuning_with_cv(self, pipeline, X_train, y_train, model_name: str) -> Tuple[Any, Dict]:
        """Perform hyperparameter tuning with nested cross-validation"""
        
        if self.progress_tracker:
            feature_type = "enhanced" if self.use_enhanced_features else "standard"
            self.progress_tracker.update(f"Tuning {model_name} with {feature_type} features")

        try:
            # Set the model in the pipeline
            pipeline.set_params(model=self.models[model_name]['model'])

            # Skip hyperparameter tuning for very small datasets
            if len(X_train) < 20:
                logger.info(f"Skipping hyperparameter tuning for {model_name} due to small dataset")
                pipeline.fit(X_train, y_train)
                
                # Still perform CV evaluation
                cv_results = self.cv_manager.perform_cross_validation(pipeline, X_train, y_train)
                
                return pipeline, {
                    'best_params': 'default_parameters',
                    'best_score': cv_results.get('test_scores', {}).get('f1', {}).get('mean', 'not_calculated'),
                    'best_estimator': pipeline,
                    'cross_validation': cv_results,
                    'note': 'Hyperparameter tuning skipped for small dataset'
                }

            # Get parameter grid
            param_grid = self.models[model_name]['param_grid']

            # Create CV strategy
            cv_strategy = self.cv_manager.create_cv_strategy(X_train, y_train)
            
            # Create GridSearchCV with nested cross-validation
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv_strategy,
                scoring='f1_weighted',
                n_jobs=1,  # Single job for stability
                verbose=0,  # Reduce verbosity for speed
                return_train_score=True  # For overfitting analysis
            )

            # Fit grid search
            logger.info(f"Starting hyperparameter tuning for {model_name}...")
            grid_search.fit(X_train, y_train)

            # Perform additional CV on best model
            logger.info(f"Performing final CV evaluation for {model_name}...")
            best_cv_results = self.cv_manager.perform_cross_validation(
                grid_search.best_estimator_, X_train, y_train, cv_strategy
            )

            # Extract results
            tuning_results = {
                'best_params': grid_search.best_params_,
                'best_score': float(grid_search.best_score_),
                'best_estimator': grid_search.best_estimator_,
                'cv_folds_used': cv_strategy.n_splits,
                'cross_validation': best_cv_results,
                'grid_search_results': {
                    'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                    'std_test_scores': grid_search.cv_results_['std_test_score'].tolist(),
                    'mean_train_scores': grid_search.cv_results_.get('mean_train_score', []).tolist() if 'mean_train_score' in grid_search.cv_results_ else [],
                    'params': grid_search.cv_results_['params']
                }
            }

            logger.info(f"Hyperparameter tuning completed for {model_name}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            logger.info(f"Best params: {grid_search.best_params_}")
            
            if 'test_scores' in best_cv_results and 'f1' in best_cv_results['test_scores']:
                final_f1 = best_cv_results['test_scores']['f1']['mean']
                final_f1_std = best_cv_results['test_scores']['f1']['std']
                logger.info(f"Final CV F1: {final_f1:.4f} (±{final_f1_std:.4f})")

            return grid_search.best_estimator_, tuning_results

        except Exception as e:
            logger.error(f"Hyperparameter tuning failed for {model_name}: {str(e)}")
            # Return basic model if tuning fails
            try:
                pipeline.set_params(model=self.models[model_name]['model'])
                pipeline.fit(X_train, y_train)
                
                # Perform basic CV
                cv_results = self.cv_manager.perform_cross_validation(pipeline, X_train, y_train)
                
                return pipeline, {
                    'error': str(e), 
                    'fallback': 'simple_training',
                    'cross_validation': cv_results
                }
            except Exception as e2:
                logger.error(f"Fallback training also failed for {model_name}: {str(e2)}")
                raise Exception(f"Both hyperparameter tuning and fallback training failed: {str(e)} | {str(e2)}")

    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train and evaluate multiple models with enhanced features and comprehensive CV"""
        
        results = {}

        for model_name in self.models.keys():
            logger.info(f"Training {model_name} with {'enhanced' if self.use_enhanced_features else 'standard'} features...")

            try:
                # Create pipeline (enhanced or standard)
                pipeline = self.create_preprocessing_pipeline()

                # Hyperparameter tuning with CV
                best_model, tuning_results = self.hyperparameter_tuning_with_cv(
                    pipeline, X_train, y_train, model_name
                )

                # Comprehensive evaluation (includes additional CV)
                evaluation_metrics = self.comprehensive_evaluation(
                    best_model, X_test, y_test, X_train, y_train
                )

                # Store results
                results[model_name] = {
                    'model': best_model,
                    'tuning_results': tuning_results,
                    'evaluation_metrics': evaluation_metrics,
                    'training_time': datetime.now().isoformat(),
                    'feature_type': 'enhanced' if self.use_enhanced_features else 'standard'
                }

                # Log results
                test_f1 = evaluation_metrics['f1']
                cv_results = evaluation_metrics.get('cross_validation', {})
                cv_f1_mean = cv_results.get('test_scores', {}).get('f1', {}).get('mean', 'N/A')
                cv_f1_std = cv_results.get('test_scores', {}).get('f1', {}).get('std', 'N/A')
                
                logger.info(f"Model {model_name} - Test F1: {test_f1:.4f}, "
                            f"CV F1: {cv_f1_mean:.4f if cv_f1_mean != 'N/A' else cv_f1_mean} "
                            f"(±{cv_f1_std:.4f if cv_f1_std != 'N/A' else cv_f1_std})")

            except Exception as e:
                logger.error(f"Training failed for {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}

        return results

    def select_best_model(self, results: Dict) -> Tuple[str, Any, Dict]:
        """Select the best performing model based on CV results"""
        
        if self.progress_tracker:
            self.progress_tracker.update("Selecting best model")

        best_model_name = None
        best_model = None
        best_score = -1
        best_metrics = None

        for model_name, result in results.items():
            if 'error' in result:
                continue

            # Prioritize CV F1 score if available, fallback to test F1
            cv_results = result['evaluation_metrics'].get('cross_validation', {})
            if 'test_scores' in cv_results and 'f1' in cv_results['test_scores']:
                f1_score = cv_results['test_scores']['f1']['mean']
                score_type = "CV F1"
            else:
                f1_score = result['evaluation_metrics']['f1']
                score_type = "Test F1"

            if f1_score > best_score:
                best_score = f1_score
                best_model_name = model_name
                best_model = result['model']
                best_metrics = result['evaluation_metrics']

        if best_model_name is None:
            raise ValueError("No models trained successfully")

        logger.info(f"Best model: {best_model_name} with {score_type} score: {best_score:.4f}")
        return best_model_name, best_model, best_metrics

    def save_model_artifacts(self, model, model_name: str, metrics: Dict, results: Dict) -> bool:
        """Save model artifacts and enhanced metadata with feature engineering results"""
        try:
            if self.progress_tracker:
                self.progress_tracker.update("Saving model")

            # Save the full pipeline with error handling
            try:
                joblib.dump(model, self.pipeline_path)
                logger.info(f"✅ Saved pipeline to {self.pipeline_path}")
            except Exception as e:
                logger.error(f"Failed to save pipeline: {e}")
                # Try alternative path
                alt_pipeline_path = Path("/tmp") / "pipeline.pkl"
                joblib.dump(model, alt_pipeline_path)
                logger.info(f"✅ Saved pipeline to {alt_pipeline_path}")

            # Save enhanced feature engineer if available
            if self.use_enhanced_features and self.feature_engineer is not None:
                try:
                    self.feature_engineer.save_pipeline(self.feature_engineer_path)
                    logger.info(f"✅ Saved feature engineer to {self.feature_engineer_path}")
                except Exception as e:
                    logger.warning(f"Could not save feature engineer: {e}")

            # Save individual components for backward compatibility
            try:
                if hasattr(model, 'named_steps'):
                    if 'model' in model.named_steps:
                        joblib.dump(model.named_steps['model'], self.model_path)
                        logger.info(f"✅ Saved model component to {self.model_path}")
                    
                    # Save vectorizer (standard pipeline) or enhanced features reference
                    if 'vectorize' in model.named_steps:
                        joblib.dump(model.named_steps['vectorize'], self.vectorizer_path)
                        logger.info(f"✅ Saved vectorizer to {self.vectorizer_path}")
                    elif 'enhanced_features' in model.named_steps:
                        # Save reference to enhanced features
                        enhanced_ref = {
                            'type': 'enhanced_features',
                            'feature_engineer_path': str(self.feature_engineer_path),
                            'metadata': self.feature_engineer.get_feature_metadata() if self.feature_engineer else {}
                        }
                        joblib.dump(enhanced_ref, self.vectorizer_path)
                        logger.info(f"✅ Saved enhanced features reference to {self.vectorizer_path}")
                        
            except Exception as e:
                logger.warning(f"Could not save individual components: {e}")

            # Generate data hash
            data_hash = hashlib.md5(str(datetime.now()).encode()).hexdigest()

            # Extract CV results
            cv_results = metrics.get('cross_validation', {})
            
            # Create enhanced metadata with feature engineering information
            metadata = {
                'model_version': f"v1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'model_type': model_name,
                'feature_engineering': {
                    'type': 'enhanced' if self.use_enhanced_features else 'standard',
                    'enhanced_features_available': ENHANCED_FEATURES_AVAILABLE,
                    'enhanced_features_used': self.use_enhanced_features
                },
                'data_version': data_hash,
                'test_accuracy': metrics['accuracy'],
                'test_f1': metrics['f1'],
                'test_precision': metrics['precision'],
                'test_recall': metrics['recall'],
                'test_roc_auc': metrics['roc_auc'],
                'overfitting_score': metrics.get('overfitting_score', 'Unknown'),
                'timestamp': datetime.now().isoformat(),
                'training_config': {
                    'test_size': self.test_size,
                    'cv_folds': self.cv_folds,
                    'max_features': self.max_features,
                    'ngram_range': self.ngram_range,
                    'feature_selection_k': self.feature_selection_k,
                    'use_enhanced_features': self.use_enhanced_features
                }
            }
            
            # Add enhanced feature metadata
            if self.use_enhanced_features:
                feature_metadata = metrics.get('feature_metadata', {})
                if feature_metadata:
                    metadata['enhanced_features'] = {
                        'total_features': feature_metadata.get('total_features', 0),
                        'feature_types': feature_metadata.get('feature_types', {}),
                        'configuration': feature_metadata.get('configuration', {})
                    }
                
                # Add top features if available
                top_features = metrics.get('top_features', {})
                if top_features:
                    metadata['top_features'] = dict(list(top_features.items())[:10])  # Top 10 features
                    
                    # Save detailed feature importance
                    try:
                        feature_analysis = {
                            'top_features': top_features,
                            'feature_metadata': feature_metadata,
                            'timestamp': datetime.now().isoformat(),
                            'model_version': metadata['model_version']
                        }
                        
                        with open(self.feature_importance_path, 'w') as f:
                            json.dump(feature_analysis, f, indent=2)
                        logger.info(f"✅ Saved feature importance analysis to {self.feature_importance_path}")
                        
                    except Exception as e:
                        logger.warning(f"Could not save feature importance: {e}")
            
            # Add comprehensive CV results to metadata
            if cv_results and 'test_scores' in cv_results:
                metadata['cross_validation'] = {
                    'n_splits': cv_results.get('n_splits', self.cv_folds),
                    'test_scores': cv_results['test_scores'],
                    'train_scores': cv_results.get('train_scores', {}),
                    'overfitting_score': cv_results.get('overfitting_score', 'Unknown'),
                    'stability_score': cv_results.get('stability_score', 'Unknown'),
                    'individual_fold_results': cv_results.get('fold_results', [])
                }
                
                # Add summary statistics
                if 'f1' in cv_results['test_scores']:
                    metadata['cv_f1_mean'] = cv_results['test_scores']['f1']['mean']
                    metadata['cv_f1_std'] = cv_results['test_scores']['f1']['std']
                    metadata['cv_f1_min'] = cv_results['test_scores']['f1']['min']
                    metadata['cv_f1_max'] = cv_results['test_scores']['f1']['max']
                
                if 'accuracy' in cv_results['test_scores']:
                    metadata['cv_accuracy_mean'] = cv_results['test_scores']['accuracy']['mean']
                    metadata['cv_accuracy_std'] = cv_results['test_scores']['accuracy']['std']
            
            # Add model comparison results if available
            if len(results) > 1:
                model_comparison = {}
                for other_model_name, other_result in results.items():
                    if other_model_name != model_name and 'error' not in other_result:
                        other_cv = other_result['evaluation_metrics'].get('cross_validation', {})
                        if cv_results and other_cv:
                            comparison = self.cv_manager.compare_cv_results(cv_results, other_cv)
                            model_comparison[other_model_name] = comparison
                
                if model_comparison:
                    metadata['model_comparison'] = model_comparison

            # Save metadata with error handling
            try:
                with open(self.metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"✅ Saved enhanced metadata to {self.metadata_path}")
            except Exception as e:
                logger.warning(f"Could not save metadata: {e}")

            # Log feature engineering summary
            if self.use_enhanced_features and feature_metadata:
                logger.info(f"✅ Enhanced features summary:")
                logger.info(f"   Total features: {feature_metadata.get('total_features', 0)}")
                for feature_type, count in feature_metadata.get('feature_types', {}).items():
                    logger.info(f"   {feature_type}: {count}")

            logger.info(f"✅ Model artifacts saved successfully with {'enhanced' if self.use_enhanced_features else 'standard'} features")
            return True

        except Exception as e:
            logger.error(f"Failed to save model artifacts: {str(e)}")
            # Try to save at least the core pipeline
            try:
                joblib.dump(model, Path("/tmp/pipeline_backup.pkl"))
                logger.info("✅ Saved backup pipeline")
                return True
            except Exception as e2:
                logger.error(f"Failed to save backup pipeline: {str(e2)}")
                return False

    def train_model(self, data_path: str = None, force_enhanced: bool = None) -> Tuple[bool, str]:
        """Main training function with enhanced feature engineering pipeline"""
        try:
            # Override enhanced features setting if specified
            if force_enhanced is not None:
                original_setting = self.use_enhanced_features
                self.use_enhanced_features = force_enhanced and ENHANCED_FEATURES_AVAILABLE
                if force_enhanced and not ENHANCED_FEATURES_AVAILABLE:
                    logger.warning("Enhanced features requested but not available, using standard features")
            
            feature_type = "enhanced" if self.use_enhanced_features else "standard"
            logger.info(f"Starting {feature_type} model training with cross-validation...")

            # Override data path if provided
            if data_path:
                self.data_path = Path(data_path)

            # Load and validate data
            success, df, message = self.load_and_validate_data()
            if not success:
                return False, message

            # Estimate training time and setup progress tracker
            time_estimate = estimate_training_time(
                len(df), 
                enable_tuning=True, 
                cv_folds=self.cv_folds,
                use_enhanced_features=self.use_enhanced_features
            )
            
            print(f"\n📊 Enhanced Training Configuration:")
            print(f"Dataset size: {len(df)} samples")
            print(f"Feature engineering: {feature_type.title()}")
            print(f"Cross-validation folds: {self.cv_folds}")
            print(f"Estimated time: {time_estimate['total_formatted']}")
            print(f"Models to train: {len(self.models)}")
            print(f"Hyperparameter tuning: Enabled")
            if self.use_enhanced_features:
                print(f"Enhanced features: Sentiment, Readability, Entities, Linguistic")
            print()

            # Setup progress tracker (adjusted for enhanced features)
            base_steps = 4 + (len(self.models) * 3) + 1  # Basic steps
            enhanced_steps = 2 if self.use_enhanced_features else 0  # Feature engineering steps
            total_steps = base_steps + enhanced_steps
            self.progress_tracker = ProgressTracker(total_steps, f"{feature_type.title()} Training Progress")

            # Prepare data
            X = df['text'].values
            y = df['label'].values

            # Train-test split with smart handling for small datasets
            self.progress_tracker.update("Splitting data")
            
            # Ensure minimum test size for very small datasets
            if len(X) < 10:
                test_size = max(0.1, 1/len(X))  # At least 1 sample for test
            else:
                test_size = self.test_size
                
            # Check if stratification is possible
            label_counts = pd.Series(y).value_counts()
            min_class_count = label_counts.min()
            can_stratify = min_class_count >= 2 and len(y) >= 4
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                stratify=y if can_stratify else None,
                random_state=self.random_state
            )

            logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test")
            
            # Additional validation for very small datasets
            if len(X_train) < 3:
                logger.warning(f"Very small training set: {len(X_train)} samples. CV results may be unreliable.")
            if len(X_test) < 1:
                return False, "Cannot create test set. Dataset too small."

            # Train and evaluate models with enhanced features
            results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test)

            # Select best model
            best_model_name, best_model, best_metrics = self.select_best_model(results)

            # Save model artifacts with enhanced feature information
            if not self.save_model_artifacts(best_model, best_model_name, best_metrics, results):
                return False, "Failed to save model artifacts"

            # Finish progress tracking
            self.progress_tracker.finish()

            # Create success message with enhanced feature information
            cv_results = best_metrics.get('cross_validation', {})
            cv_info = ""
            if 'test_scores' in cv_results and 'f1' in cv_results['test_scores']:
                cv_f1_mean = cv_results['test_scores']['f1']['mean']
                cv_f1_std = cv_results['test_scores']['f1']['std']
                cv_info = f", CV F1: {cv_f1_mean:.4f} (±{cv_f1_std:.4f})"

            # Enhanced features summary
            feature_info = ""
            if self.use_enhanced_features:
                feature_metadata = best_metrics.get('feature_metadata', {})
                if feature_metadata:
                    total_features = feature_metadata.get('total_features', 0)
                    feature_info = f", Enhanced Features: {total_features}"

            success_message = (
                f"{feature_type.title()} model training completed successfully. "
                f"Best model: {best_model_name} "
                f"(Test F1: {best_metrics['f1']:.4f}, Test Accuracy: {best_metrics['accuracy']:.4f}{cv_info}{feature_info})"
            )

            logger.info(success_message)
            return True, success_message

        except Exception as e:
            if self.progress_tracker:
                print()  # New line after progress bar
            error_message = f"Enhanced model training failed: {str(e)}"
            logger.error(error_message)
            return False, error_message


def main():
    """Main execution function with enhanced feature engineering support"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train fake news detection model with enhanced features')
    parser.add_argument('--data_path', type=str, help='Path to training data CSV file')
    parser.add_argument('--config_path', type=str, help='Path to training configuration JSON file')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--enhanced_features', action='store_true', help='Force use of enhanced features')
    parser.add_argument('--standard_features', action='store_true', help='Force use of standard TF-IDF features only')
    args = parser.parse_args()
    
    # Determine feature engineering mode
    use_enhanced = None
    if args.enhanced_features and args.standard_features:
        logger.warning("Both --enhanced_features and --standard_features specified. Using auto-detection.")
    elif args.enhanced_features:
        use_enhanced = True
        logger.info("Enhanced features explicitly requested")
    elif args.standard_features:
        use_enhanced = False
        logger.info("Standard features explicitly requested")
    
    trainer = EnhancedModelTrainer(use_enhanced_features=use_enhanced)
    
    # Apply CV folds from command line
    if args.cv_folds:
        trainer.cv_folds = args.cv_folds
        trainer.cv_manager.cv_folds = args.cv_folds
    
    # Load custom configuration if provided
    if args.config_path and Path(args.config_path).exists():
        try:
            with open(args.config_path, 'r') as f:
                config = json.load(f)
            
            # Apply configuration
            trainer.test_size = config.get('test_size', trainer.test_size)
            trainer.cv_folds = config.get('cv_folds', trainer.cv_folds)
            trainer.cv_manager.cv_folds = trainer.cv_folds
            trainer.max_features = config.get('max_features', trainer.max_features)
            trainer.ngram_range = tuple(config.get('ngram_range', trainer.ngram_range))
            
            # Enhanced feature configuration
            if 'enhanced_features' in config and use_enhanced is None:
                trainer.use_enhanced_features = config['enhanced_features'] and ENHANCED_FEATURES_AVAILABLE
            
            # Filter models if specified
            selected_models = config.get('selected_models')
            if selected_models and len(selected_models) < len(trainer.models):
                all_models = trainer.models.copy()
                trainer.models = {k: v for k, v in all_models.items() if k in selected_models}
            
            # Update feature selection based on max_features
            trainer.feature_selection_k = min(trainer.feature_selection_k, trainer.max_features)
            
            logger.info(f"Applied custom configuration with {trainer.cv_folds} CV folds")
            if trainer.use_enhanced_features:
                logger.info("Enhanced features enabled via configuration")
            
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}, using defaults")
    
    success, message = trainer.train_model(data_path=args.data_path)

    if success:
        print(f"✅ {message}")
        
        # Print feature engineering summary
        if trainer.use_enhanced_features and trainer.feature_engineer:
            try:
                metadata = trainer.feature_engineer.get_feature_metadata()
                print(f"\n📈 Enhanced Feature Engineering Summary:")
                print(f"Total features generated: {metadata['total_features']}")
                for feature_type, count in metadata['feature_types'].items():
                    print(f"  {feature_type}: {count}")
            except Exception as e:
                logger.warning(f"Could not display feature summary: {e}")
    else:
        print(f"❌ {message}")
        exit(1)


if __name__ == "__main__":
    main()