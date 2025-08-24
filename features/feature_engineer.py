# File: features/feature_engineer.py
# Enhanced Feature Engineering Pipeline for Priority 6

import json
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.sparse import hstack, csr_matrix
from typing import Dict, List, Any, Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.preprocessing import StandardScaler, FunctionTransformer

import warnings
warnings.filterwarnings('ignore')

# Import feature analyzers
from features.sentiment_analyzer import SentimentAnalyzer
from features.readability_analyzer import ReadabilityAnalyzer
from features.entity_analyzer import EntityAnalyzer
from features.linguistic_analyzer import LinguisticAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering pipeline combining multiple NLP feature extractors
    for enhanced fake news detection performance.
    """
    
    def __init__(self, 
                 enable_sentiment: bool = True,
                 enable_readability: bool = True,
                 enable_entities: bool = True,
                 enable_linguistic: bool = True,
                 feature_selection_k: int = 5000,
                 tfidf_max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 3),
                 min_df: int = 2,
                 max_df: float = 0.95):
        """
        Initialize the advanced feature engineering pipeline.
        
        Args:
            enable_sentiment: Enable sentiment analysis features
            enable_readability: Enable readability/complexity features
            enable_entities: Enable named entity recognition features
            enable_linguistic: Enable advanced linguistic features
            feature_selection_k: Number of features to select
            tfidf_max_features: Maximum TF-IDF features
            ngram_range: N-gram range for TF-IDF
            min_df: Minimum document frequency for TF-IDF
            max_df: Maximum document frequency for TF-IDF
        """
        self.enable_sentiment = enable_sentiment
        self.enable_readability = enable_readability
        self.enable_entities = enable_entities
        self.enable_linguistic = enable_linguistic
        self.feature_selection_k = feature_selection_k
        self.tfidf_max_features = tfidf_max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        # Initialize feature extractors
        self.sentiment_analyzer = SentimentAnalyzer() if enable_sentiment else None
        self.readability_analyzer = ReadabilityAnalyzer() if enable_readability else None
        self.entity_analyzer = EntityAnalyzer() if enable_entities else None
        self.linguistic_analyzer = LinguisticAnalyzer() if enable_linguistic else None
        
        # Initialize TF-IDF components
        self.tfidf_vectorizer = None
        self.feature_selector = None
        self.feature_scaler = None
        
        # Feature metadata
        self.feature_names_ = []
        self.feature_importance_ = {}
        self.is_fitted_ = False
        
    def fit(self, X, y=None):
        """
        Fit the feature engineering pipeline.
        
        Args:
            X: Text data (array-like of strings)
            y: Target labels (optional, for supervised feature selection)
        """
        logger.info("Fitting advanced feature engineering pipeline...")
        
        # Convert to array if needed
        if isinstance(X, pd.Series):
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)
        
        # Validate input
        if len(X) == 0:
            raise ValueError("Cannot fit on empty data")
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english',
            sublinear_tf=True,
            norm='l2',
            lowercase=True
        )
        
        # Fit TF-IDF on text data
        logger.info("Fitting TF-IDF vectorizer...")
        tfidf_features = self.tfidf_vectorizer.fit_transform(X)
        
        # Extract additional features
        additional_features = self._extract_additional_features(X, fit=True)
        
        # Combine all features
        if additional_features.shape[1] > 0:
            all_features = hstack([tfidf_features, additional_features])
        else:
            all_features = tfidf_features
            
        logger.info(f"Total features before selection: {all_features.shape[1]}")
        
        # Feature selection
        if y is not None and self.feature_selection_k < all_features.shape[1]:
            logger.info(f"Performing feature selection (k={self.feature_selection_k})...")
            
            # Use chi2 for text features and mutual information for numerical features
            self.feature_selector = SelectKBest(
                score_func=chi2, 
                k=min(self.feature_selection_k, all_features.shape[1])
            )
            
            # Ensure non-negative features for chi2
            if hasattr(all_features, 'toarray'):
                features_dense = all_features.toarray()
            else:
                features_dense = all_features
                
            # Make features non-negative for chi2
            features_dense = np.maximum(features_dense, 0)
            
            self.feature_selector.fit(features_dense, y)
            selected_features = self.feature_selector.transform(features_dense)
            
            logger.info(f"Selected {selected_features.shape[1]} features")
        else:
            selected_features = all_features
            
        # Scale numerical features (additional features only)
        if additional_features.shape[1] > 0:
            self.feature_scaler = StandardScaler()
            # Only scale the additional features part
            additional_selected = selected_features[:, -additional_features.shape[1]:]
            self.feature_scaler.fit(additional_selected)
        
        # Generate feature names
        self._generate_feature_names()
        
        # Calculate feature importance if possible
        if y is not None and self.feature_selector is not None:
            self._calculate_feature_importance()
        
        self.is_fitted_ = True
        logger.info("Feature engineering pipeline fitted successfully")
        
        return self
    
    def transform(self, X):
        """
        Transform text data into enhanced feature vectors.
        
        Args:
            X: Text data (array-like of strings)
            
        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted before transforming")
        
        # Convert to array if needed
        if isinstance(X, pd.Series):
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)
        
        # Extract TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform(X)
        
        # Extract additional features
        additional_features = self._extract_additional_features(X, fit=False)
        
        # Combine features
        if additional_features.shape[1] > 0:
            all_features = hstack([tfidf_features, additional_features])
        else:
            all_features = tfidf_features
        
        # Apply feature selection
        if self.feature_selector is not None:
            if hasattr(all_features, 'toarray'):
                features_dense = all_features.toarray()
            else:
                features_dense = all_features
                
            # Ensure non-negative for consistency
            features_dense = np.maximum(features_dense, 0)
            selected_features = self.feature_selector.transform(features_dense)
        else:
            selected_features = all_features
        
        # Scale additional features if scaler exists
        if self.feature_scaler is not None and additional_features.shape[1] > 0:
            # Scale only the additional features part
            tfidf_selected = selected_features[:, :-additional_features.shape[1]]
            additional_selected = selected_features[:, -additional_features.shape[1]:]
            additional_scaled = self.feature_scaler.transform(additional_selected)
            
            # Combine back
            if hasattr(tfidf_selected, 'toarray'):
                tfidf_selected = tfidf_selected.toarray()
            
            final_features = np.hstack([tfidf_selected, additional_scaled])
        else:
            if hasattr(selected_features, 'toarray'):
                final_features = selected_features.toarray()
            else:
                final_features = selected_features
        
        return final_features
    
    def _extract_additional_features(self, X, fit=False):
        """Extract additional features beyond TF-IDF"""
        feature_arrays = []
        
        try:
            # Sentiment features
            if self.sentiment_analyzer is not None:
                logger.info("Extracting sentiment features...")
                if fit:
                    sentiment_features = self.sentiment_analyzer.fit_transform(X)
                else:
                    sentiment_features = self.sentiment_analyzer.transform(X)
                feature_arrays.append(sentiment_features)
            
            # Readability features
            if self.readability_analyzer is not None:
                logger.info("Extracting readability features...")
                if fit:
                    readability_features = self.readability_analyzer.fit_transform(X)
                else:
                    readability_features = self.readability_analyzer.transform(X)
                feature_arrays.append(readability_features)
            
            # Entity features
            if self.entity_analyzer is not None:
                logger.info("Extracting entity features...")
                if fit:
                    entity_features = self.entity_analyzer.fit_transform(X)
                else:
                    entity_features = self.entity_analyzer.transform(X)
                feature_arrays.append(entity_features)
            
            # Linguistic features
            if self.linguistic_analyzer is not None:
                logger.info("Extracting linguistic features...")
                if fit:
                    linguistic_features = self.linguistic_analyzer.fit_transform(X)
                else:
                    linguistic_features = self.linguistic_analyzer.transform(X)
                feature_arrays.append(linguistic_features)
            
            # Combine all additional features
            if feature_arrays:
                additional_features = np.hstack(feature_arrays)
                logger.info(f"Extracted {additional_features.shape[1]} additional features")
            else:
                additional_features = np.empty((len(X), 0))
                
        except Exception as e:
            logger.warning(f"Error extracting additional features: {e}")
            additional_features = np.empty((len(X), 0))
        
        return additional_features
    
    def _generate_feature_names(self):
        """Generate comprehensive feature names"""
        self.feature_names_ = []
        
        # TF-IDF feature names
        if self.tfidf_vectorizer is not None:
            tfidf_names = [f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]
            self.feature_names_.extend(tfidf_names)
        
        # Additional feature names
        if self.sentiment_analyzer is not None:
            self.feature_names_.extend(self.sentiment_analyzer.get_feature_names())
        
        if self.readability_analyzer is not None:
            self.feature_names_.extend(self.readability_analyzer.get_feature_names())
        
        if self.entity_analyzer is not None:
            self.feature_names_.extend(self.entity_analyzer.get_feature_names())
        
        if self.linguistic_analyzer is not None:
            self.feature_names_.extend(self.linguistic_analyzer.get_feature_names())
        
        # Apply feature selection to names if applicable
        if self.feature_selector is not None:
            selected_indices = self.feature_selector.get_support()
            self.feature_names_ = [name for i, name in enumerate(self.feature_names_) if selected_indices[i]]
    
    def _calculate_feature_importance(self):
        """Calculate feature importance scores"""
        if self.feature_selector is not None:
            scores = self.feature_selector.scores_
            selected_indices = self.feature_selector.get_support()
            
            # Get scores for selected features
            selected_scores = scores[selected_indices]
            
            # Create importance dictionary
            self.feature_importance_ = {
                name: float(score) for name, score in zip(self.feature_names_, selected_scores)
            }
            
            # Sort by importance
            self.feature_importance_ = dict(
                sorted(self.feature_importance_.items(), key=lambda x: x[1], reverse=True)
            )
    
    def get_feature_names(self):
        """Get names of output features"""
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted first")
        return self.feature_names_
    
    def get_feature_importance(self, top_k=None):
        """Get feature importance scores"""
        if not self.feature_importance_:
            return {}
        
        if top_k is not None:
            return dict(list(self.feature_importance_.items())[:top_k])
        
        return self.feature_importance_
    
    def get_feature_metadata(self):
        """Get comprehensive feature metadata"""
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted first")
        
        metadata = {
            'total_features': len(self.feature_names_),
            'feature_types': {
                'tfidf_features': sum(1 for name in self.feature_names_ if name.startswith('tfidf_')),
                'sentiment_features': sum(1 for name in self.feature_names_ if name.startswith('sentiment_')),
                'readability_features': sum(1 for name in self.feature_names_ if name.startswith('readability_')),
                'entity_features': sum(1 for name in self.feature_names_ if name.startswith('entity_')),
                'linguistic_features': sum(1 for name in self.feature_names_ if name.startswith('linguistic_'))
            },
            'configuration': {
                'enable_sentiment': self.enable_sentiment,
                'enable_readability': self.enable_readability,
                'enable_entities': self.enable_entities,
                'enable_linguistic': self.enable_linguistic,
                'feature_selection_k': self.feature_selection_k,
                'tfidf_max_features': self.tfidf_max_features,
                'ngram_range': self.ngram_range
            },
            'feature_importance_available': bool(self.feature_importance_),
            'timestamp': datetime.now().isoformat()
        }
        
        return metadata
    
    def save_pipeline(self, filepath):
        """Save the fitted pipeline"""
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted before saving")
        
        save_data = {
            'feature_engineer': self,
            'metadata': self.get_feature_metadata(),
            'feature_names': self.feature_names_,
            'feature_importance': self.feature_importance_
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Feature engineering pipeline saved to {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath):
        """Load a fitted pipeline"""
        save_data = joblib.load(filepath)
        feature_engineer = save_data['feature_engineer']
        
        logger.info(f"Feature engineering pipeline loaded from {filepath}")
        return feature_engineer


def create_enhanced_pipeline(X_train, y_train, 
                           enable_sentiment=True,
                           enable_readability=True, 
                           enable_entities=True,
                           enable_linguistic=True,
                           feature_selection_k=5000):
    """
    Create and fit an enhanced feature engineering pipeline.
    
    Args:
        X_train: Training text data
        y_train: Training labels
        enable_sentiment: Enable sentiment analysis features
        enable_readability: Enable readability features
        enable_entities: Enable entity features
        enable_linguistic: Enable linguistic features
        feature_selection_k: Number of features to select
        
    Returns:
        Fitted AdvancedFeatureEngineer instance
    """
    logger.info("Creating enhanced feature engineering pipeline...")
    
    # Create feature engineer
    feature_engineer = AdvancedFeatureEngineer(
        enable_sentiment=enable_sentiment,
        enable_readability=enable_readability,
        enable_entities=enable_entities,
        enable_linguistic=enable_linguistic,
        feature_selection_k=feature_selection_k
    )
    
    # Fit the pipeline
    feature_engineer.fit(X_train, y_train)
    
    # Log feature information
    metadata = feature_engineer.get_feature_metadata()
    logger.info(f"Enhanced pipeline created with {metadata['total_features']} features")
    logger.info(f"Feature breakdown: {metadata['feature_types']}")
    
    return feature_engineer


def analyze_feature_importance(feature_engineer, top_k=20):
    """
    Analyze and display feature importance.
    
    Args:
        feature_engineer: Fitted AdvancedFeatureEngineer instance
        top_k: Number of top features to analyze
        
    Returns:
        Dictionary with feature analysis results
    """
    if not feature_engineer.is_fitted_:
        raise ValueError("Feature engineer must be fitted first")
    
    # Get feature importance
    importance = feature_engineer.get_feature_importance(top_k=top_k)
    metadata = feature_engineer.get_feature_metadata()
    
    # Analyze feature types in top features
    top_features = list(importance.keys())
    feature_type_counts = {}
    
    for feature in top_features:
        if feature.startswith('tfidf_'):
            feature_type = 'tfidf'
        elif feature.startswith('sentiment_'):
            feature_type = 'sentiment'
        elif feature.startswith('readability_'):
            feature_type = 'readability'
        elif feature.startswith('entity_'):
            feature_type = 'entity'
        elif feature.startswith('linguistic_'):
            feature_type = 'linguistic'
        else:
            feature_type = 'other'
        
        feature_type_counts[feature_type] = feature_type_counts.get(feature_type, 0) + 1
    
    analysis = {
        'top_features': importance,
        'feature_type_distribution': feature_type_counts,
        'total_features': metadata['total_features'],
        'feature_breakdown': metadata['feature_types'],
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    return analysis