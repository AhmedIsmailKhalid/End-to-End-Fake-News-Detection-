# Enhanced version with comprehensive cross-validation and advanced feature engineering


import json
import shutil
import joblib
import asyncio
import logging
import hashlib
import schedule
import threading
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import time as time_module
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any, List
from monitor.monitor_drift import AdvancedDriftMonitor

import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, cross_validate, train_test_split
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectKBest, chi2

# Import enhanced feature engineering components
try:
    from features.feature_engineer import AdvancedFeatureEngineer, create_enhanced_pipeline, analyze_feature_importance
    from features.sentiment_analyzer import SentimentAnalyzer
    from features.readability_analyzer import ReadabilityAnalyzer
    from features.entity_analyzer import EntityAnalyzer
    from features.linguistic_analyzer import LinguisticAnalyzer
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    logging.warning(f"Enhanced features not available in retrain.py, falling back to basic TF-IDF: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/model_retraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log enhanced feature availability
if ENHANCED_FEATURES_AVAILABLE:
    logger.info("Enhanced feature engineering components loaded for retraining")
else:
    logger.warning("Enhanced features not available - using standard TF-IDF for retraining")


def preprocess_text_function(texts):
    """Standalone function for text preprocessing - pickle-safe"""
    import re
    
    def clean_single_text(text):
        text = str(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[^a-zA-Z\s.!?]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    processed = []
    for text in texts:
        processed.append(clean_single_text(text))
    
    return processed


class CVModelComparator:
    """Advanced model comparison using cross-validation and statistical tests with enhanced features"""
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        
    def create_cv_strategy(self, X, y) -> StratifiedKFold:
        """Create appropriate CV strategy based on data characteristics"""
        n_samples = len(X)
        min_samples_per_fold = 3
        max_folds = n_samples // min_samples_per_fold
        
        unique_classes = np.unique(y)
        min_class_count = min([np.sum(y == cls) for cls in unique_classes])
        max_folds_by_class = min_class_count
        
        actual_folds = max(2, min(self.cv_folds, max_folds, max_folds_by_class))
        
        logger.info(f"Using {actual_folds} CV folds for enhanced model comparison")
        
        return StratifiedKFold(
            n_splits=actual_folds,
            shuffle=True,
            random_state=self.random_state
        )
    
    def perform_model_cv_evaluation(self, model, X, y, cv_strategy=None) -> Dict:
        """Perform comprehensive CV evaluation of a model with enhanced features"""
        
        if cv_strategy is None:
            cv_strategy = self.create_cv_strategy(X, y)
        
        logger.info(f"Performing enhanced CV evaluation with {cv_strategy.n_splits} folds...")
        
        scoring_metrics = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted',
            'roc_auc': 'roc_auc'
        }
        
        try:
            cv_scores = cross_validate(
                model, X, y,
                cv=cv_strategy,
                scoring=scoring_metrics,
                return_train_score=True,
                n_jobs=1,
                verbose=0
            )
            
            cv_results = {
                'n_splits': cv_strategy.n_splits,
                'test_scores': {},
                'train_scores': {},
                'fold_results': [],
                'feature_engineering_type': self._detect_feature_type(model)
            }
            
            # Process results for each metric
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
                        'scores': train_scores.tolist()
                    }
            
            # Individual fold results
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
            
            # Calculate overfitting and stability scores
            if 'accuracy' in cv_results['test_scores'] and 'accuracy' in cv_results['train_scores']:
                train_mean = cv_results['train_scores']['accuracy']['mean']
                test_mean = cv_results['test_scores']['accuracy']['mean']
                cv_results['overfitting_score'] = float(train_mean - test_mean)
                
                test_std = cv_results['test_scores']['accuracy']['std']
                cv_results['stability_score'] = float(1 - (test_std / test_mean)) if test_mean > 0 else 0
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Enhanced CV evaluation failed: {e}")
            return {'error': str(e), 'n_splits': cv_strategy.n_splits}
    
    def _detect_feature_type(self, model) -> str:
        """Detect whether model uses enhanced or standard features"""
        try:
            if hasattr(model, 'named_steps'):
                if 'enhanced_features' in model.named_steps:
                    return 'enhanced'
                elif 'vectorize' in model.named_steps:
                    return 'standard_tfidf'
            return 'unknown'
        except:
            return 'unknown'
    
    def compare_models_with_cv(self, model1, model2, X, y, model1_name="Production", model2_name="Candidate") -> Dict:
        """Compare two models using cross-validation with enhanced feature awareness"""
        
        logger.info(f"Comparing {model1_name} vs {model2_name} models using enhanced CV...")
        
        try:
            cv_strategy = self.create_cv_strategy(X, y)
            
            # Evaluate both models with same CV folds
            results1 = self.perform_model_cv_evaluation(model1, X, y, cv_strategy)
            results2 = self.perform_model_cv_evaluation(model2, X, y, cv_strategy)
            
            if 'error' in results1 or 'error' in results2:
                return {
                    'error': 'One or both models failed CV evaluation',
                    'model1_results': results1,
                    'model2_results': results2
                }
            
            # Statistical comparison with feature type awareness
            comparison_results = {
                'model1_name': model1_name,
                'model2_name': model2_name,
                'cv_folds': cv_strategy.n_splits,
                'model1_cv_results': results1,
                'model2_cv_results': results2,
                'statistical_tests': {},
                'metric_comparisons': {},
                'feature_engineering_comparison': {
                    'model1_features': results1.get('feature_engineering_type', 'unknown'),
                    'model2_features': results2.get('feature_engineering_type', 'unknown'),
                    'feature_upgrade': self._assess_feature_upgrade(results1, results2)
                }
            }
            
            # Compare each metric
            for metric in ['accuracy', 'f1', 'precision', 'recall']:
                if (metric in results1['test_scores'] and 
                    metric in results2['test_scores']):
                    
                    scores1 = results1['test_scores'][metric]['scores']
                    scores2 = results2['test_scores'][metric]['scores']
                    
                    metric_comparison = self._compare_metric_scores(
                        scores1, scores2, metric, model1_name, model2_name
                    )
                    comparison_results['metric_comparisons'][metric] = metric_comparison
            
            # Enhanced promotion decision logic
            promotion_decision = self._make_enhanced_promotion_decision(comparison_results)
            comparison_results['promotion_decision'] = promotion_decision
            
            logger.info(f"Enhanced model comparison completed: {promotion_decision['reason']}")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Enhanced model comparison failed: {e}")
            return {'error': str(e)}
    
    def _assess_feature_upgrade(self, results1: Dict, results2: Dict) -> Dict:
        """Assess if there's a feature engineering upgrade"""
        feature1 = results1.get('feature_engineering_type', 'unknown')
        feature2 = results2.get('feature_engineering_type', 'unknown')
        
        upgrade_assessment = {
            'is_upgrade': False,
            'upgrade_type': 'none',
            'description': 'No feature engineering change detected'
        }
        
        if feature1 == 'standard_tfidf' and feature2 == 'enhanced':
            upgrade_assessment.update({
                'is_upgrade': True,
                'upgrade_type': 'standard_to_enhanced',
                'description': 'Upgrade from standard TF-IDF to enhanced feature engineering'
            })
        elif feature1 == 'enhanced' and feature2 == 'standard_tfidf':
            upgrade_assessment.update({
                'is_upgrade': False,
                'upgrade_type': 'enhanced_to_standard',
                'description': 'Downgrade from enhanced features to standard TF-IDF'
            })
        elif feature1 == feature2 and feature1 != 'unknown':
            upgrade_assessment.update({
                'is_upgrade': False,
                'upgrade_type': 'same_features',
                'description': f'Both models use {feature1} features'
            })
        
        return upgrade_assessment
    
    def _make_enhanced_promotion_decision(self, comparison_results: Dict) -> Dict:
        """Enhanced promotion decision that considers feature engineering upgrades"""
        f1_comparison = comparison_results['metric_comparisons'].get('f1', {})
        accuracy_comparison = comparison_results['metric_comparisons'].get('accuracy', {})
        feature_comparison = comparison_results['feature_engineering_comparison']
        
        promote_candidate = False
        promotion_reason = ""
        confidence = 0.0
        
        # Factor in feature engineering improvements
        feature_upgrade = feature_comparison.get('feature_upgrade', {})
        is_feature_upgrade = feature_upgrade.get('is_upgrade', False)
        
        # Enhanced decision logic
        if f1_comparison.get('significant_improvement', False):
            promote_candidate = True
            promotion_reason = f"Significant F1 improvement: {f1_comparison.get('improvement', 0):.4f}"
            confidence = 0.8
            
            if is_feature_upgrade:
                promotion_reason += " with enhanced feature engineering"
                confidence = 0.9
                
        elif is_feature_upgrade and f1_comparison.get('improvement', 0) > 0.005:
            # Lower threshold for promotion when upgrading features
            promote_candidate = True
            promotion_reason = f"Feature engineering upgrade with F1 improvement: {f1_comparison.get('improvement', 0):.4f}"
            confidence = 0.7
            
        elif (f1_comparison.get('improvement', 0) > 0.01 and 
              accuracy_comparison.get('improvement', 0) > 0.01):
            promote_candidate = True
            promotion_reason = "Practical improvement in both F1 and accuracy"
            confidence = 0.6
            
            if is_feature_upgrade:
                promotion_reason += " with enhanced features"
                confidence = 0.75
                
        elif f1_comparison.get('improvement', 0) > 0.02:
            promote_candidate = True
            promotion_reason = f"Large F1 improvement: {f1_comparison.get('improvement', 0):.4f}"
            confidence = 0.7
        else:
            if is_feature_upgrade:
                promotion_reason = f"Feature upgrade available but insufficient performance gain ({f1_comparison.get('improvement', 0):.4f})"
            else:
                promotion_reason = "No significant improvement detected"
            confidence = 0.3
        
        return {
            'promote_candidate': promote_candidate,
            'reason': promotion_reason,
            'confidence': confidence,
            'feature_engineering_factor': is_feature_upgrade,
            'feature_upgrade_details': feature_upgrade
        }
    
    def _compare_metric_scores(self, scores1: list, scores2: list, metric: str, 
                              model1_name: str, model2_name: str) -> Dict:
        """Compare metric scores between two models using statistical tests"""
        
        try:
            # Basic statistics
            mean1, mean2 = np.mean(scores1), np.mean(scores2)
            std1, std2 = np.std(scores1), np.std(scores2)
            improvement = mean2 - mean1
            
            comparison = {
                'metric': metric,
                f'{model1_name.lower()}_mean': float(mean1),
                f'{model2_name.lower()}_mean': float(mean2),
                f'{model1_name.lower()}_std': float(std1),
                f'{model2_name.lower()}_std': float(std2),
                'improvement': float(improvement),
                'relative_improvement': float(improvement / mean1 * 100) if mean1 > 0 else 0,
                'tests': {}
            }
            
            # Paired t-test
            try:
                t_stat, p_value = stats.ttest_rel(scores2, scores1)
                comparison['tests']['paired_ttest'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
            except Exception as e:
                logger.warning(f"Paired t-test failed for {metric}: {e}")
            
            # Wilcoxon signed-rank test (non-parametric alternative)
            try:
                w_stat, w_p_value = stats.wilcoxon(scores2, scores1, alternative='greater')
                comparison['tests']['wilcoxon'] = {
                    'statistic': float(w_stat),
                    'p_value': float(w_p_value),
                    'significant': w_p_value < 0.05
                }
            except Exception as e:
                logger.warning(f"Wilcoxon test failed for {metric}: {e}")
            
            # Effect size (Cohen's d)
            try:
                pooled_std = np.sqrt(((len(scores1) - 1) * std1**2 + (len(scores2) - 1) * std2**2) / 
                                   (len(scores1) + len(scores2) - 2))
                cohens_d = improvement / pooled_std if pooled_std > 0 else 0
                comparison['effect_size'] = float(cohens_d)
            except Exception:
                comparison['effect_size'] = 0
            
            # Practical significance
            practical_threshold = 0.01  # 1% improvement threshold
            comparison['practical_significance'] = abs(improvement) > practical_threshold
            comparison['significant_improvement'] = (
                improvement > practical_threshold and 
                comparison['tests'].get('paired_ttest', {}).get('significant', False)
            )
            
            return comparison
            
        except Exception as e:
            logger.error(f"Metric comparison failed for {metric}: {e}")
            return {'metric': metric, 'error': str(e)}
    
    def _calculate_decision_confidence(self, comparison_results: Dict) -> float:
        """Calculate confidence in the promotion decision with enhanced features consideration"""
        
        try:
            confidence_factors = []
            
            # Check F1 improvement significance
            f1_comp = comparison_results['metric_comparisons'].get('f1', {})
            if f1_comp.get('significant_improvement', False):
                confidence_factors.append(0.4)
            elif f1_comp.get('improvement', 0) > 0.01:
                confidence_factors.append(0.2)
            
            # Feature engineering upgrade bonus
            feature_upgrade = comparison_results.get('feature_engineering_comparison', {}).get('feature_upgrade', {})
            if feature_upgrade.get('is_upgrade', False):
                confidence_factors.append(0.3)  # Significant bonus for feature upgrades
            
            # Check consistency across metrics
            improved_metrics = 0
            total_metrics = 0
            for metric_comp in comparison_results['metric_comparisons'].values():
                if isinstance(metric_comp, dict) and 'improvement' in metric_comp:
                    total_metrics += 1
                    if metric_comp['improvement'] > 0:
                        improved_metrics += 1
            
            if total_metrics > 0:
                consistency_score = improved_metrics / total_metrics
                confidence_factors.append(consistency_score * 0.3)
            
            # Check effect sizes
            effect_sizes = []
            for metric_comp in comparison_results['metric_comparisons'].values():
                if isinstance(metric_comp, dict) and 'effect_size' in metric_comp:
                    effect_sizes.append(abs(metric_comp['effect_size']))
            
            if effect_sizes:
                avg_effect_size = np.mean(effect_sizes)
                if avg_effect_size > 0.5:  # Large effect
                    confidence_factors.append(0.2)
                elif avg_effect_size > 0.2:  # Medium effect
                    confidence_factors.append(0.1)
            
            # Calculate final confidence
            total_confidence = sum(confidence_factors)
            return min(1.0, max(0.0, total_confidence))
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5


class EnhancedModelRetrainer:
    """Production-ready model retraining with enhanced feature engineering and comprehensive CV"""
    
    def __init__(self):
        self.setup_paths()
        self.setup_retraining_config()
        self.setup_statistical_tests()
        self.cv_comparator = CVModelComparator()
        
        # Enhanced feature engineering settings
        self.enhanced_features_available = ENHANCED_FEATURES_AVAILABLE
        self.use_enhanced_features = ENHANCED_FEATURES_AVAILABLE  # Default to enhanced if available
        
        logger.info(f"Enhanced retraining initialized with features: {'enhanced' if self.use_enhanced_features else 'standard'}")
    
    def setup_paths(self):
        """Setup all necessary paths"""
        self.base_dir = Path("/tmp")
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "model"
        self.logs_dir = self.base_dir / "logs"
        self.backup_dir = self.base_dir / "backups"
        self.features_dir = self.base_dir / "features"  # For enhanced features
        
        # Create directories
        for dir_path in [self.data_dir, self.model_dir, self.logs_dir, self.backup_dir, self.features_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Current production files
        self.prod_model_path = self.model_dir / "model.pkl"
        self.prod_vectorizer_path = self.model_dir / "vectorizer.pkl"
        self.prod_pipeline_path = self.model_dir / "pipeline.pkl"
        self.prod_feature_engineer_path = self.features_dir / "feature_engineer.pkl"
        
        # Candidate files
        self.candidate_model_path = self.model_dir / "model_candidate.pkl"
        self.candidate_vectorizer_path = self.model_dir / "vectorizer_candidate.pkl"
        self.candidate_pipeline_path = self.model_dir / "pipeline_candidate.pkl"
        self.candidate_feature_engineer_path = self.features_dir / "feature_engineer_candidate.pkl"
        
        # Data files
        self.combined_data_path = self.data_dir / "combined_dataset.csv"
        self.scraped_data_path = self.data_dir / "scraped_real.csv"
        self.generated_data_path = self.data_dir / "generated_fake.csv"
        
        # Metadata and logs
        self.metadata_path = Path("/tmp/metadata.json")
        self.retraining_log_path = self.logs_dir / "retraining_log.json"
        self.comparison_log_path = self.logs_dir / "model_comparison.json"
        self.feature_analysis_log_path = self.logs_dir / "feature_analysis.json"
    
    def setup_retraining_config(self):
        """Setup enhanced retraining configuration"""
        self.min_new_samples = 50
        self.improvement_threshold = 0.01  # 1% improvement required
        self.significance_level = 0.05
        self.cv_folds = 5
        self.test_size = 0.2
        self.random_state = 42
        self.max_retries = 3
        self.backup_retention_days = 30
        
        # Enhanced feature configuration
        self.enhanced_feature_config = {
            'enable_sentiment': True,
            'enable_readability': True,
            'enable_entities': True,
            'enable_linguistic': True,
            'feature_selection_k': 3000,
            'tfidf_max_features': 7500,
            'ngram_range': (1, 3),
            'min_df': 2,
            'max_df': 0.95
        }
    
    def setup_statistical_tests(self):
        """Setup statistical test configurations"""
        self.statistical_tests = {
            'paired_ttest': {'alpha': 0.05, 'name': "Paired T-Test"},
            'wilcoxon': {'alpha': 0.05, 'name': "Wilcoxon Signed-Rank Test"},
            'mcnemar': {'alpha': 0.05, 'name': "McNemar's Test"}
        }
    
    def detect_production_feature_type(self) -> str:
        """Detect what type of features the production model uses"""
        try:
            # Check if enhanced feature engineer exists
            if self.prod_feature_engineer_path.exists():
                return 'enhanced'
            
            # Check pipeline structure
            if self.prod_pipeline_path.exists():
                pipeline = joblib.load(self.prod_pipeline_path)
                if hasattr(pipeline, 'named_steps'):
                    if 'enhanced_features' in pipeline.named_steps:
                        return 'enhanced'
                    elif 'vectorize' in pipeline.named_steps:
                        return 'standard_tfidf'
            
            # Check metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    feature_info = metadata.get('feature_engineering', {})
                    if feature_info.get('type') == 'enhanced':
                        return 'enhanced'
            
            return 'standard_tfidf'
            
        except Exception as e:
            logger.warning(f"Could not detect production feature type: {e}")
            return 'unknown'

    def load_existing_metadata(self) -> Optional[Dict]:
        """Load existing model metadata with enhanced feature information"""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Log feature engineering information
                feature_info = metadata.get('feature_engineering', {})
                logger.info(f"Loaded metadata: {metadata.get('model_version', 'Unknown')} with {feature_info.get('type', 'unknown')} features")
                return metadata
            else:
                logger.warning("No existing metadata found")
                return None
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            return None
    
    def load_production_model(self) -> Tuple[bool, Optional[Any], str]:
        """Load current production model with enhanced feature support"""
        try:
            # Detect production feature type
            prod_feature_type = self.detect_production_feature_type()
            logger.info(f"Production model uses: {prod_feature_type} features")
            
            # Try to load pipeline first (preferred)
            if self.prod_pipeline_path.exists():
                model = joblib.load(self.prod_pipeline_path)
                logger.info("Loaded production pipeline")
                return True, model, f"Pipeline loaded successfully ({prod_feature_type} features)"
            
            # Fallback to individual components
            elif self.prod_model_path.exists() and self.prod_vectorizer_path.exists():
                model = joblib.load(self.prod_model_path)
                vectorizer = joblib.load(self.prod_vectorizer_path)
                logger.info("Loaded production model and vectorizer components")
                return True, (model, vectorizer), f"Model components loaded successfully ({prod_feature_type} features)"
            
            else:
                return False, None, "No production model found"
                
        except Exception as e:
            error_msg = f"Failed to load production model: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def load_new_data(self) -> Tuple[bool, Optional[pd.DataFrame], str]:
        """Load and combine all available data"""
        try:
            logger.info("Loading training data for enhanced retraining...")
            
            dataframes = []
            
            # Load combined dataset (base)
            if self.combined_data_path.exists():
                df_combined = pd.read_csv(self.combined_data_path)
                dataframes.append(df_combined)
                logger.info(f"Loaded combined dataset: {len(df_combined)} samples")
            
            # Load scraped real news
            if self.scraped_data_path.exists():
                df_scraped = pd.read_csv(self.scraped_data_path)
                if 'label' not in df_scraped.columns:
                    df_scraped['label'] = 0  # Real news
                dataframes.append(df_scraped)
                logger.info(f"Loaded scraped data: {len(df_scraped)} samples")
            
            # Load generated fake news
            if self.generated_data_path.exists():
                df_generated = pd.read_csv(self.generated_data_path)
                if 'label' not in df_generated.columns:
                    df_generated['label'] = 1  # Fake news
                dataframes.append(df_generated)
                logger.info(f"Loaded generated data: {len(df_generated)} samples")
            
            if not dataframes:
                return False, None, "No data files found"
            
            # Combine all data
            df = pd.concat(dataframes, ignore_index=True)
            
            # Data cleaning and validation
            df = self.clean_and_validate_data(df)
            
            if len(df) < 100:
                return False, None, f"Insufficient data after cleaning: {len(df)} samples"
            
            logger.info(f"Total training data: {len(df)} samples")
            return True, df, f"Successfully loaded {len(df)} samples"
            
        except Exception as e:
            error_msg = f"Failed to load data: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the training data"""
        initial_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['text'], keep='first')
        
        # Remove null values
        df = df.dropna(subset=['text', 'label'])
        
        # Validate text quality
        df = df[df['text'].astype(str).str.len() > 10]
        
        # Validate labels
        df = df[df['label'].isin([0, 1])]
        
        # Remove excessive length texts
        df = df[df['text'].astype(str).str.len() < 10000]
        
        logger.info(f"Data cleaning: {initial_count} -> {len(df)} samples")
        return df
    
    def create_enhanced_pipeline(self, use_enhanced_features: bool = None) -> Pipeline:
        """Create enhanced ML pipeline with feature engineering"""
        
        if use_enhanced_features is None:
            use_enhanced_features = self.use_enhanced_features
        
        if use_enhanced_features and ENHANCED_FEATURES_AVAILABLE:
            logger.info("Creating enhanced feature engineering pipeline for retraining...")
            
            # Create enhanced feature engineer with configuration
            feature_engineer = AdvancedFeatureEngineer(
                enable_sentiment=self.enhanced_feature_config['enable_sentiment'],
                enable_readability=self.enhanced_feature_config['enable_readability'],
                enable_entities=self.enhanced_feature_config['enable_entities'],
                enable_linguistic=self.enhanced_feature_config['enable_linguistic'],
                feature_selection_k=self.enhanced_feature_config['feature_selection_k'],
                tfidf_max_features=self.enhanced_feature_config['tfidf_max_features'],
                ngram_range=self.enhanced_feature_config['ngram_range'],
                min_df=self.enhanced_feature_config['min_df'],
                max_df=self.enhanced_feature_config['max_df']
            )
            
            # Create pipeline with enhanced features
            pipeline = Pipeline([
                ('enhanced_features', feature_engineer),
                ('model', LogisticRegression(
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=self.random_state
                ))
            ])
            
            return pipeline
            
        else:
            logger.info("Creating standard TF-IDF pipeline for retraining...")
            
            def preprocess_text(texts):
                return preprocess_text_function(texts)
            
            # Create standard pipeline
            pipeline = Pipeline([
                ('preprocess', FunctionTransformer(preprocess_text, validate=False)),
                ('vectorize', TfidfVectorizer(
                    max_features=10000,
                    min_df=2,
                    max_df=0.95,
                    ngram_range=(1, 3),
                    stop_words='english',
                    sublinear_tf=True
                )),
                ('feature_select', SelectKBest(chi2, k=5000)),
                ('model', LogisticRegression(
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=self.random_state
                ))
            ])
            
            return pipeline
    
    def train_candidate_model(self, df: pd.DataFrame) -> Tuple[bool, Optional[Any], Dict]:
        """Train candidate model with enhanced features and comprehensive CV evaluation"""
        try:
            logger.info("Training candidate model with enhanced feature engineering...")
            
            # Prepare data
            X = df['text'].values
            y = df['label'].values
            
            # Determine feature type to use for candidate
            candidate_feature_type = 'enhanced' if self.use_enhanced_features else 'standard'
            prod_feature_type = self.detect_production_feature_type()
            
            logger.info(f"Training candidate with {candidate_feature_type} features (production uses {prod_feature_type})")
            
            # Create and train pipeline
            pipeline = self.create_enhanced_pipeline(self.use_enhanced_features)
            
            # Perform cross-validation before final training
            logger.info("Performing cross-validation on candidate model...")
            cv_results = self.cv_comparator.perform_model_cv_evaluation(pipeline, X, y)
            
            # Train on full dataset for final model
            pipeline.fit(X, y)
            
            # Additional holdout evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
            )
            
            pipeline_holdout = self.create_enhanced_pipeline(self.use_enhanced_features)
            pipeline_holdout.fit(X_train, y_train)
            
            # Evaluate on holdout
            y_pred = pipeline_holdout.predict(X_test)
            y_pred_proba = pipeline_holdout.predict_proba(X_test)[:, 1]
            
            holdout_metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted')),
                'recall': float(recall_score(y_test, y_pred, average='weighted')),
                'f1': float(f1_score(y_test, y_pred, average='weighted')),
                'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
            }
            
            # Extract feature information if using enhanced features
            feature_analysis = {}
            if self.use_enhanced_features and hasattr(pipeline, 'named_steps'):
                feature_engineer = pipeline.named_steps.get('enhanced_features')
                if feature_engineer and hasattr(feature_engineer, 'get_feature_metadata'):
                    try:
                        feature_analysis = {
                            'feature_metadata': feature_engineer.get_feature_metadata(),
                            'feature_importance': feature_engineer.get_feature_importance(top_k=20) if hasattr(feature_engineer, 'get_feature_importance') else {},
                            'total_features': len(feature_engineer.get_feature_names()) if hasattr(feature_engineer, 'get_feature_names') else 0
                        }
                        logger.info(f"Enhanced features extracted: {feature_analysis['total_features']} total features")
                    except Exception as e:
                        logger.warning(f"Could not extract feature analysis: {e}")
            
            # Combine CV and holdout results
            evaluation_results = {
                'cross_validation': cv_results,
                'holdout_evaluation': holdout_metrics,
                'feature_analysis': feature_analysis,
                'feature_type': candidate_feature_type,
                'training_samples': len(X),
                'test_samples': len(X_test)
            }
            
            # Save candidate model
            joblib.dump(pipeline, self.candidate_pipeline_path)
            if hasattr(pipeline, 'named_steps'):
                if 'model' in pipeline.named_steps:
                    joblib.dump(pipeline.named_steps['model'], self.candidate_model_path)
                
                # Save enhanced features or vectorizer
                if 'enhanced_features' in pipeline.named_steps:
                    feature_engineer = pipeline.named_steps['enhanced_features']
                    if hasattr(feature_engineer, 'save_pipeline'):
                        feature_engineer.save_pipeline(self.candidate_feature_engineer_path)
                    
                    # Save reference as vectorizer for compatibility
                    enhanced_ref = {
                        'type': 'enhanced_features',
                        'feature_engineer_path': str(self.candidate_feature_engineer_path),
                        'metadata': feature_analysis.get('feature_metadata', {})
                    }
                    joblib.dump(enhanced_ref, self.candidate_vectorizer_path)
                    
                elif 'vectorize' in pipeline.named_steps:
                    joblib.dump(pipeline.named_steps['vectorize'], self.candidate_vectorizer_path)
            
            # Log results
            if 'test_scores' in cv_results and 'f1' in cv_results['test_scores']:
                cv_f1_mean = cv_results['test_scores']['f1']['mean']
                cv_f1_std = cv_results['test_scores']['f1']['std']
                logger.info(f"Candidate model CV F1: {cv_f1_mean:.4f} (±{cv_f1_std:.4f})")
            
            logger.info(f"Candidate model holdout F1: {holdout_metrics['f1']:.4f}")
            logger.info(f"Candidate model training completed with {candidate_feature_type} features")
            
            return True, pipeline, evaluation_results
            
        except Exception as e:
            error_msg = f"Candidate model training failed: {str(e)}"
            logger.error(error_msg)
            return False, None, {'error': error_msg}
    
    def compare_models_with_enhanced_cv_validation(self, prod_model, candidate_model, X, y) -> Dict:
        """Compare models using comprehensive cross-validation with enhanced feature awareness"""
        
        logger.info("Performing comprehensive model comparison with enhanced CV...")
        
        try:
            # Use the enhanced CV comparator for detailed analysis
            comparison_results = self.cv_comparator.compare_models_with_cv(
                prod_model, candidate_model, X, y, "Production", "Candidate"
            )
            
            if 'error' in comparison_results:
                return comparison_results
            
            # Additional legacy format for backward compatibility
            legacy_comparison = {
                'production_cv_results': comparison_results['model1_cv_results'],
                'candidate_cv_results': comparison_results['model2_cv_results'],
                'statistical_tests': comparison_results['statistical_tests'],
                'promotion_decision': comparison_results['promotion_decision']
            }
            
            # Extract key metrics for legacy format
            prod_cv = comparison_results['model1_cv_results']
            cand_cv = comparison_results['model2_cv_results']
            
            if 'test_scores' in prod_cv and 'test_scores' in cand_cv:
                if 'accuracy' in prod_cv['test_scores'] and 'accuracy' in cand_cv['test_scores']:
                    legacy_comparison.update({
                        'production_accuracy': prod_cv['test_scores']['accuracy']['mean'],
                        'candidate_accuracy': cand_cv['test_scores']['accuracy']['mean'],
                        'absolute_improvement': (cand_cv['test_scores']['accuracy']['mean'] - 
                                               prod_cv['test_scores']['accuracy']['mean']),
                        'relative_improvement': ((cand_cv['test_scores']['accuracy']['mean'] - 
                                                prod_cv['test_scores']['accuracy']['mean']) / 
                                               prod_cv['test_scores']['accuracy']['mean'] * 100)
                    })
            
            # Merge detailed and legacy formats
            final_results = {**comparison_results, **legacy_comparison}
            
            # Log summary with enhanced feature information
            f1_comp = comparison_results.get('metric_comparisons', {}).get('f1', {})
            feature_comp = comparison_results.get('feature_engineering_comparison', {})
            
            if f1_comp:
                logger.info(f"F1 improvement: {f1_comp.get('improvement', 0):.4f}")
                logger.info(f"Significant improvement: {f1_comp.get('significant_improvement', False)}")
            
            if feature_comp:
                feature_upgrade = feature_comp.get('feature_upgrade', {})
                logger.info(f"Feature engineering: {feature_upgrade.get('description', 'No change')}")
            
            promotion_decision = comparison_results.get('promotion_decision', {})
            logger.info(f"Promotion recommendation: {promotion_decision.get('promote_candidate', False)}")
            logger.info(f"Reason: {promotion_decision.get('reason', 'Unknown')}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Enhanced model comparison failed: {str(e)}")
            return {'error': str(e)}
    
    def create_backup(self) -> bool:
        """Create backup of current production model with enhanced features"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = self.backup_dir / f"backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup files
            files_to_backup = [
                (self.prod_model_path, backup_dir / "model.pkl"),
                (self.prod_vectorizer_path, backup_dir / "vectorizer.pkl"),
                (self.prod_pipeline_path, backup_dir / "pipeline.pkl"),
                (self.metadata_path, backup_dir / "metadata.json"),
                (self.prod_feature_engineer_path, backup_dir / "feature_engineer.pkl")  # Enhanced features
            ]
            
            for source, dest in files_to_backup:
                if source.exists():
                    shutil.copy2(source, dest)
            
            logger.info(f"Backup created: {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Backup creation failed: {str(e)}")
            return False
    
    def promote_candidate_model(self, candidate_model, candidate_metrics: Dict, comparison_results: Dict) -> bool:
        """Promote candidate model to production with enhanced metadata and feature support"""
        try:
            logger.info("Promoting candidate model to production with enhanced features...")
            
            # Create backup first
            if not self.create_backup():
                logger.error("Backup creation failed, aborting promotion")
                return False
            
            # Copy candidate files to production
            shutil.copy2(self.candidate_model_path, self.prod_model_path)
            shutil.copy2(self.candidate_vectorizer_path, self.prod_vectorizer_path)
            shutil.copy2(self.candidate_pipeline_path, self.prod_pipeline_path)
            
            # Copy enhanced feature engineer if it exists
            if self.candidate_feature_engineer_path.exists():
                shutil.copy2(self.candidate_feature_engineer_path, self.prod_feature_engineer_path)
                logger.info("Enhanced feature engineer promoted to production")
            
            # Update metadata with comprehensive enhanced feature information
            metadata = self.load_existing_metadata() or {}
            
            # Increment version
            old_version = metadata.get('model_version', 'v1.0')
            if old_version.startswith('v'):
                try:
                    major, minor = map(int, old_version[1:].split('.'))
                    new_version = f"v{major}.{minor + 1}"
                except:
                    new_version = f"v1.{int(datetime.now().timestamp()) % 1000}"
            else:
                new_version = f"v1.{int(datetime.now().timestamp()) % 1000}"
            
            # Extract metrics from candidate evaluation
            cv_results = candidate_metrics.get('cross_validation', {})
            holdout_results = candidate_metrics.get('holdout_evaluation', {})
            feature_analysis = candidate_metrics.get('feature_analysis', {})
            
            # Update metadata with comprehensive information
            metadata.update({
                'model_version': new_version,
                'model_type': 'enhanced_retrained_pipeline_cv',
                'previous_version': old_version,
                'promotion_timestamp': datetime.now().isoformat(),
                'retrain_trigger': 'enhanced_cv_validated_retrain',
                'training_samples': candidate_metrics.get('training_samples', 'Unknown'),
                'test_samples': candidate_metrics.get('test_samples', 'Unknown')
            })
            
            # Enhanced feature engineering metadata
            feature_type = candidate_metrics.get('feature_type', 'unknown')
            metadata['feature_engineering'] = {
                'type': feature_type,
                'enhanced_features_available': ENHANCED_FEATURES_AVAILABLE,
                'enhanced_features_used': feature_type == 'enhanced',
                'feature_upgrade': comparison_results.get('feature_engineering_comparison', {}).get('feature_upgrade', {})
            }
            
            # Add feature analysis if available
            if feature_analysis:
                feature_metadata = feature_analysis.get('feature_metadata', {})
                if feature_metadata:
                    metadata['enhanced_features'] = {
                        'total_features': feature_analysis.get('total_features', 0),
                        'feature_types': feature_metadata.get('feature_types', {}),
                        'configuration': feature_metadata.get('configuration', {})
                    }
                
                # Add top features
                top_features = feature_analysis.get('feature_importance', {})
                if top_features:
                    metadata['top_features'] = dict(list(top_features.items())[:10])
                    
                    # Save detailed feature analysis
                    try:
                        feature_analysis_data = {
                            'top_features': top_features,
                            'feature_metadata': feature_metadata,
                            'model_version': new_version,
                            'timestamp': datetime.now().isoformat(),
                            'feature_type': feature_type
                        }
                        
                        with open(self.feature_analysis_log_path, 'w') as f:
                            json.dump(feature_analysis_data, f, indent=2)
                        logger.info(f"Feature analysis saved to {self.feature_analysis_log_path}")
                        
                    except Exception as e:
                        logger.warning(f"Could not save feature analysis: {e}")
            
            # Add holdout evaluation results
            if holdout_results:
                metadata.update({
                    'test_accuracy': holdout_results.get('accuracy', 'Unknown'),
                    'test_f1': holdout_results.get('f1', 'Unknown'),
                    'test_precision': holdout_results.get('precision', 'Unknown'),
                    'test_recall': holdout_results.get('recall', 'Unknown'),
                    'test_roc_auc': holdout_results.get('roc_auc', 'Unknown')
                })
            
            # Add comprehensive CV results
            if cv_results and 'test_scores' in cv_results:
                metadata['cross_validation'] = {
                    'n_splits': cv_results.get('n_splits', self.cv_folds),
                    'test_scores': cv_results['test_scores'],
                    'train_scores': cv_results.get('train_scores', {}),
                    'overfitting_score': cv_results.get('overfitting_score', 'Unknown'),
                    'stability_score': cv_results.get('stability_score', 'Unknown'),
                    'individual_fold_results': cv_results.get('fold_results', []),
                    'feature_engineering_type': cv_results.get('feature_engineering_type', feature_type)
                }
                
                # Add CV summary statistics
                if 'f1' in cv_results['test_scores']:
                    metadata.update({
                        'cv_f1_mean': cv_results['test_scores']['f1']['mean'],
                        'cv_f1_std': cv_results['test_scores']['f1']['std'],
                        'cv_f1_min': cv_results['test_scores']['f1']['min'],
                        'cv_f1_max': cv_results['test_scores']['f1']['max']
                    })
            
            # Add enhanced model comparison results
            promotion_decision = comparison_results.get('promotion_decision', {})
            metadata['promotion_validation'] = {
                'decision_confidence': promotion_decision.get('confidence', 'Unknown'),
                'promotion_reason': promotion_decision.get('reason', 'Unknown'),
                'comparison_method': 'enhanced_cv_statistical_tests',
                'feature_engineering_factor': promotion_decision.get('feature_engineering_factor', False),
                'feature_upgrade_details': promotion_decision.get('feature_upgrade_details', {})
            }
            
            # Add enhanced statistical test results
            metric_comparisons = comparison_results.get('metric_comparisons', {})
            if metric_comparisons:
                metadata['statistical_validation'] = {}
                for metric, comparison in metric_comparisons.items():
                    if isinstance(comparison, dict):
                        metadata['statistical_validation'][metric] = {
                            'improvement': comparison.get('improvement', 0),
                            'significant_improvement': comparison.get('significant_improvement', False),
                            'effect_size': comparison.get('effect_size', 0),
                            'tests': comparison.get('tests', {})
                        }
            
            # Save updated metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Log promotion summary
            feature_info = ""
            if feature_type == 'enhanced':
                total_features = feature_analysis.get('total_features', 0)
                feature_info = f" with {total_features} enhanced features"
            
            logger.info(f"Model promoted successfully to {new_version}{feature_info}")
            logger.info(f"Promotion reason: {promotion_decision.get('reason', 'Enhanced CV validation passed')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Enhanced model promotion failed: {str(e)}")
            return False
    
    def log_retraining_session(self, results: Dict):
        """Log comprehensive retraining session results with enhanced feature information"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'session_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
                'retraining_type': 'enhanced_cv_features',
                'enhanced_features_used': self.use_enhanced_features,
                'enhanced_features_available': ENHANCED_FEATURES_AVAILABLE
            }
            
            # Load existing logs
            logs = []
            if self.retraining_log_path.exists():
                try:
                    with open(self.retraining_log_path, 'r') as f:
                        logs = json.load(f)
                except:
                    logs = []
            
            # Add new log
            logs.append(log_entry)
            
            # Keep only last 100 entries
            if len(logs) > 100:
                logs = logs[-100:]
            
            # Save logs
            with open(self.retraining_log_path, 'w') as f:
                json.dump(logs, f, indent=2)
            
            # Also save detailed comparison results
            if 'comparison_results' in results:
                comparison_logs = []
                if self.comparison_log_path.exists():
                    try:
                        with open(self.comparison_log_path, 'r') as f:
                            comparison_logs = json.load(f)
                    except:
                        comparison_logs = []
                
                comparison_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'session_id': log_entry['session_id'],
                    'comparison_details': results['comparison_results'],
                    'enhanced_features_info': {
                        'used': self.use_enhanced_features,
                        'available': ENHANCED_FEATURES_AVAILABLE,
                        'feature_comparison': results['comparison_results'].get('feature_engineering_comparison', {})
                    }
                }
                
                comparison_logs.append(comparison_entry)
                if len(comparison_logs) > 50:
                    comparison_logs = comparison_logs[-50:]
                
                with open(self.comparison_log_path, 'w') as f:
                    json.dump(comparison_logs, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to log enhanced retraining session: {str(e)}")
    
    def retrain_model(self) -> Tuple[bool, str]:
        """Main retraining function with enhanced feature engineering and comprehensive CV validation"""
        try:
            logger.info("Starting enhanced model retraining with advanced feature engineering...")
            
            # Load existing metadata
            existing_metadata = self.load_existing_metadata()
            
            # Load production model
            prod_success, prod_model, prod_msg = self.load_production_model()
            if not prod_success:
                logger.warning(f"No production model found: {prod_msg}")
                # Fall back to initial training
                try:
                    from model.train import main as train_main
                    train_main()
                    return True, "Initial enhanced training completed"
                except ImportError:
                    return False, "No production model and cannot import training module"
            
            # Load new data
            data_success, df, data_msg = self.load_new_data()
            if not data_success:
                return False, data_msg
            
            # Check if we have enough new data
            if len(df) < self.min_new_samples:
                return False, f"Insufficient new data: {len(df)} < {self.min_new_samples}"
            
            # Determine optimal feature engineering strategy
            prod_feature_type = self.detect_production_feature_type()
            candidate_feature_type = 'enhanced' if self.use_enhanced_features else 'standard'
            
            logger.info(f"Retraining strategy: {prod_feature_type} -> {candidate_feature_type}")
            
            # Train candidate model with enhanced features
            candidate_success, candidate_model, candidate_metrics = self.train_candidate_model(df)
            if not candidate_success:
                return False, f"Enhanced candidate training failed: {candidate_metrics.get('error', 'Unknown error')}"
            
            # Prepare data for model comparison
            X = df['text'].values
            y = df['label'].values
            
            # Comprehensive model comparison with enhanced CV
            comparison_results = self.compare_models_with_enhanced_cv_validation(
                prod_model, candidate_model, X, y
            )
            
            # Log results with enhanced information
            session_results = {
                'candidate_metrics': candidate_metrics,
                'comparison_results': comparison_results,
                'data_size': len(df),
                'cv_folds': self.cv_folds,
                'retraining_method': 'enhanced_cv_features',
                'feature_engineering': {
                    'production_type': prod_feature_type,
                    'candidate_type': candidate_feature_type,
                    'feature_upgrade': comparison_results.get('feature_engineering_comparison', {})
                }
            }
            
            self.log_retraining_session(session_results)
            
            # Enhanced decision based on CV comparison
            promotion_decision = comparison_results.get('promotion_decision', {})
            should_promote = promotion_decision.get('promote_candidate', False)
            
            if should_promote:
                # Promote candidate model
                promotion_success = self.promote_candidate_model(
                    candidate_model, candidate_metrics, comparison_results
                )
                
                if promotion_success:
                    # Extract improvement information
                    f1_comp = comparison_results.get('metric_comparisons', {}).get('f1', {})
                    improvement = f1_comp.get('improvement', 0)
                    confidence = promotion_decision.get('confidence', 0)
                    feature_upgrade = promotion_decision.get('feature_engineering_factor', False)
                    
                    feature_info = ""
                    if feature_upgrade:
                        feature_info = " with enhanced feature engineering upgrade"
                    elif candidate_feature_type == 'enhanced':
                        feature_info = " using enhanced features"
                    
                    success_msg = (
                        f"Enhanced model promoted successfully{feature_info}! "
                        f"F1 improvement: {improvement:.4f}, "
                        f"Confidence: {confidence:.2f}, "
                        f"Reason: {promotion_decision.get('reason', 'Enhanced CV validation passed')}"
                    )
                    logger.info(success_msg)
                    return True, success_msg
                else:
                    return False, "Enhanced model promotion failed"
            else:
                # Keep current model
                reason = promotion_decision.get('reason', 'No significant improvement detected')
                confidence = promotion_decision.get('confidence', 0)
                
                keep_msg = (
                    f"Keeping current model based on enhanced CV analysis. "
                    f"Reason: {reason}, "
                    f"Confidence: {confidence:.2f}"
                )
                logger.info(keep_msg)
                return True, keep_msg
            
        except Exception as e:
            error_msg = f"Enhanced model retraining failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def automated_retrain_with_validation(self) -> Tuple[bool, str]:
        """Automated retraining with enhanced validation and feature engineering"""
        try:
            logger.info("Starting automated enhanced retraining with validation...")
            
            # Use the main enhanced retraining method
            success, message = self.retrain_model()
            
            if success:
                logger.info("Automated enhanced retraining completed successfully")
                return True, f"Enhanced automated retraining: {message}"
            else:
                logger.error(f"Automated enhanced retraining failed: {message}")
                return False, f"Enhanced automated retraining failed: {message}"
        
        except Exception as e:
            logger.error(f"Automated enhanced retraining failed: {e}")
            return False, f"Automated enhanced retraining failed: {str(e)}"


class AutomatedRetrainingManager:
    """Manages automated retraining triggers and scheduling with enhanced features"""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path("/tmp")
        self.setup_automation_paths()
        self.setup_automation_config()
        self.drift_monitor = AdvancedDriftMonitor()
        self.retraining_active = False
        self.automation_thread = None
        self.last_check_time = None
        
        # Enhanced feature settings
        self.enhanced_features_available = ENHANCED_FEATURES_AVAILABLE
        
        self.automation_status = {
            'enabled': True,
            'last_automated_training': None,
            'total_automated_trainings': 0,
            'failed_attempts': 0,
            'enhanced_features_used': self.enhanced_features_available
        }
        
        logger.info(f"Automated retraining manager initialized with enhanced features: {self.enhanced_features_available}")
    
    def setup_automation_paths(self):
        """Setup automation-specific paths"""
        self.automation_dir = self.base_dir / "automation"
        self.automation_dir.mkdir(parents=True, exist_ok=True)
        
        self.automation_log_path = self.automation_dir / "automation_log.json"
        self.retraining_queue_path = self.automation_dir / "retraining_queue.json"
        self.automation_config_path = self.automation_dir / "automation_config.json"
    
    def setup_automation_config(self):
        """Setup automation configuration with enhanced feature considerations"""
        self.automation_config = {
            'monitoring_schedule': {
                'check_interval_minutes': 360,  # 6 hours
                'force_check_interval_hours': 24,
                'max_daily_retrainings': 3,
                'cooldown_hours_after_training': 6
            },
            'retraining_conditions': {
                'require_data_quality_check': True,
                'min_time_between_trainings': timedelta(hours=6),
                'max_consecutive_failures': 3,
                'emergency_override': True,
                'prefer_enhanced_features': True  # New setting
            },
            'notification_settings': {
                'notify_on_trigger': True,
                'notify_on_completion': True,
                'notify_on_failure': True,
                'notify_on_feature_upgrade': True  # New setting
            }
        }
        
        self.load_automation_config()
    
    def load_automation_config(self):
        """Load automation configuration from file"""
        try:
            if self.automation_config_path.exists():
                with open(self.automation_config_path, 'r') as f:
                    saved_config = json.load(f)
                
                # Update with saved settings
                self.automation_config.update(saved_config)
                logger.info("Loaded enhanced automation configuration")
        except Exception as e:
            logger.warning(f"Failed to load automation config: {e}")
    
    def save_automation_config(self):
        """Save automation configuration to file"""
        try:
            with open(self.automation_config_path, 'w') as f:
                # Convert timedelta objects to strings for JSON serialization
                config_to_save = json.loads(json.dumps(self.automation_config, default=str))
                json.dump(config_to_save, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save automation config: {e}")
    
    def start_automated_monitoring(self):
        """Start the automated monitoring and retraining system with enhanced features"""
        if self.retraining_active:
            logger.warning("Automated monitoring already active")
            return
        
        self.retraining_active = True
        
        # Schedule periodic checks
        check_interval = self.automation_config['monitoring_schedule']['check_interval_minutes']
        schedule.every(check_interval).minutes.do(self.perform_scheduled_check)
        
        # Schedule daily forced check
        schedule.every().day.at("02:00").do(self.perform_forced_check)
        
        # Start background thread
        self.automation_thread = threading.Thread(target=self.automation_loop, daemon=True)
        self.automation_thread.start()
        
        logger.info("Enhanced automated retraining monitoring started")
        self.log_automation_event("monitoring_started", "Enhanced automated monitoring system started")
    
    def stop_automated_monitoring(self):
        """Stop the automated monitoring system"""
        self.retraining_active = False
        schedule.clear()
        
        if self.automation_thread:
            self.automation_thread.join(timeout=10)
        
        logger.info("Enhanced automated retraining monitoring stopped")
        self.log_automation_event("monitoring_stopped", "Enhanced automated monitoring system stopped")
    
    def automation_loop(self):
        """Main automation loop"""
        while self.retraining_active:
            try:
                schedule.run_pending()
                time_module.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Enhanced automation loop error: {e}")
                time_module.sleep(300)  # Wait 5 minutes on error
    
    def perform_scheduled_check(self):
        """Perform scheduled retraining trigger check"""
        try:
            logger.info("Performing scheduled enhanced retraining trigger check")
            
            # Check if we're in cooldown period
            if self.is_in_cooldown_period():
                logger.info("Skipping check - in cooldown period after recent training")
                return
            
            # Check daily limit
            if self.exceeded_daily_retraining_limit():
                logger.info("Skipping check - daily retraining limit exceeded")
                return
            
            # Perform trigger evaluation
            trigger_results = self.drift_monitor.check_retraining_triggers()
            
            self.last_check_time = datetime.now()
            
            # Log the check
            self.log_automation_event("scheduled_check", "Performed scheduled enhanced trigger check", {
                'trigger_results': trigger_results,
                'should_retrain': trigger_results.get('should_retrain', False),
                'enhanced_features_available': self.enhanced_features_available
            })
            
            # Trigger retraining if needed
            if trigger_results.get('should_retrain', False):
                self.queue_retraining(trigger_results)
            
        except Exception as e:
            logger.error(f"Scheduled enhanced check failed: {e}")
            self.log_automation_event("check_failed", f"Scheduled enhanced check failed: {str(e)}")
    
    def perform_forced_check(self):
        """Perform forced daily check regardless of other conditions"""
        try:
            logger.info("Performing forced daily enhanced retraining check")
            
            # Always perform drift monitoring
            trigger_results = self.drift_monitor.check_retraining_triggers()
            
            self.log_automation_event("forced_check", "Performed forced daily enhanced check", {
                'trigger_results': trigger_results,
                'enhanced_features_available': self.enhanced_features_available
            })
            
            # Only trigger retraining for urgent cases during forced check
            if trigger_results.get('urgency') in ['critical', 'high']:
                self.queue_retraining(trigger_results, forced=True)
            
        except Exception as e:
            logger.error(f"Forced enhanced check failed: {e}")
            self.log_automation_event("forced_check_failed", f"Forced enhanced check failed: {str(e)}")
    
    def queue_retraining(self, trigger_results: Dict, forced: bool = False):
        """Queue a retraining job with enhanced feature considerations"""
        try:
            retraining_job = {
                'queued_at': datetime.now().isoformat(),
                'trigger_results': trigger_results,
                'urgency': trigger_results.get('urgency', 'medium'),
                'forced': forced,
                'status': 'queued',
                'attempts': 0,
                'enhanced_features_available': self.enhanced_features_available,
                'prefer_enhanced_features': self.automation_config['retraining_conditions'].get('prefer_enhanced_features', True)
            }
            
            # Load existing queue
            queue = self.load_retraining_queue()
            queue.append(retraining_job)
            
            # Save queue
            self.save_retraining_queue(queue)
            
            logger.info(f"Enhanced retraining queued with urgency: {trigger_results.get('urgency', 'medium')}")
            self.log_automation_event("retraining_queued", "Enhanced retraining job queued", retraining_job)
            
            # Execute immediately for critical cases
            if trigger_results.get('urgency') == 'critical' or forced:
                self.execute_queued_retraining()
            
        except Exception as e:
            logger.error(f"Failed to queue enhanced retraining: {e}")
            self.log_automation_event("queue_failed", f"Failed to queue enhanced retraining: {str(e)}")
    
    def execute_queued_retraining(self):
        """Execute queued retraining jobs with enhanced features"""
        try:
            queue = self.load_retraining_queue()
            
            # Sort by urgency (critical > high > medium > low)
            urgency_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            queue.sort(key=lambda x: urgency_order.get(x.get('urgency', 'medium'), 2))
            
            executed_jobs = []
            
            for job in queue:
                if job['status'] == 'queued':
                    success = self.execute_single_retraining(job)
                    
                    if success:
                        job['status'] = 'completed'
                        job['completed_at'] = datetime.now().isoformat()
                        self.automation_status['last_automated_training'] = datetime.now().isoformat()
                        self.automation_status['total_automated_trainings'] += 1
                        executed_jobs.append(job)
                        break  # Only execute one job at a time
                    else:
                        job['attempts'] += 1
                        if job['attempts'] >= 3:
                            job['status'] = 'failed'
                            job['failed_at'] = datetime.now().isoformat()
                        self.automation_status['failed_attempts'] += 1
            
            # Update queue
            remaining_queue = [job for job in queue if job['status'] == 'queued']
            self.save_retraining_queue(remaining_queue)
            
            return len(executed_jobs) > 0
            
        except Exception as e:
            logger.error(f"Failed to execute queued enhanced retraining: {e}")
            return False
    
    def execute_single_retraining(self, job: Dict) -> bool:
        """Execute a single retraining job with enhanced features"""
        try:
            logger.info(f"Starting automated enhanced retraining with urgency: {job.get('urgency', 'medium')}")
            
            job['started_at'] = datetime.now().isoformat()
            job['status'] = 'running'
            
            # Create enhanced retraining manager
            retrainer = EnhancedModelRetrainer()
            
            # Configure enhanced features based on job preferences
            prefer_enhanced = job.get('prefer_enhanced_features', True)
            retrainer.use_enhanced_features = prefer_enhanced and ENHANCED_FEATURES_AVAILABLE
            
            # Perform enhanced retraining with validation
            success, result = retrainer.automated_retrain_with_validation()
            
            if success:
                logger.info("Automated enhanced retraining completed successfully")
                self.log_automation_event("retraining_success", "Automated enhanced retraining completed", {
                    'job': job,
                    'result': result,
                    'enhanced_features_used': retrainer.use_enhanced_features
                })
                return True
            else:
                logger.error(f"Automated enhanced retraining failed: {result}")
                self.log_automation_event("retraining_failed", f"Automated enhanced retraining failed: {result}", {
                    'job': job,
                    'enhanced_features_used': retrainer.use_enhanced_features
                })
                return False
                
        except Exception as e:
            logger.error(f"Enhanced retraining execution failed: {e}")
            self.log_automation_event("retraining_error", f"Enhanced retraining execution error: {str(e)}", {'job': job})
            return False
    
    def is_in_cooldown_period(self) -> bool:
        """Check if we're in cooldown period after recent training"""
        try:
            if not self.automation_status['last_automated_training']:
                return False
            
            last_training = datetime.fromisoformat(self.automation_status['last_automated_training'])
            cooldown_hours = self.automation_config['retraining_conditions']['cooldown_hours_after_training']
            cooldown_period = timedelta(hours=cooldown_hours)
            
            return datetime.now() - last_training < cooldown_period
            
        except Exception as e:
            logger.warning(f"Failed to check cooldown period: {e}")
            return False
    
    def exceeded_daily_retraining_limit(self) -> bool:
        """Check if daily retraining limit has been exceeded"""
        try:
            if not self.automation_status['last_automated_training']:
                return False
            
            last_training = datetime.fromisoformat(self.automation_status['last_automated_training'])
            
            # Count trainings in last 24 hours
            training_logs = self.get_recent_automation_logs(hours=24)
            training_count = len([log for log in training_logs if log.get('event') == 'retraining_success'])
            
            max_daily = self.automation_config['monitoring_schedule']['max_daily_retrainings']
            
            return training_count >= max_daily
            
        except Exception as e:
            logger.warning(f"Failed to check daily limit: {e}")
            return False
    
    def load_retraining_queue(self) -> List[Dict]:
        """Load retraining queue from file"""
        try:
            if self.retraining_queue_path.exists():
                with open(self.retraining_queue_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Failed to load retraining queue: {e}")
            return []
    
    def save_retraining_queue(self, queue: List[Dict]):
        """Save retraining queue to file"""
        try:
            with open(self.retraining_queue_path, 'w') as f:
                json.dump(queue, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save retraining queue: {e}")
    
    def log_automation_event(self, event: str, message: str, details: Dict = None):
        """Log automation events with enhanced feature information"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event': event,
                'message': message,
                'details': details or {},
                'enhanced_features_available': self.enhanced_features_available
            }
            
            # Load existing logs
            logs = []
            if self.automation_log_path.exists():
                try:
                    with open(self.automation_log_path, 'r') as f:
                        logs = json.load(f)
                except:
                    logs = []
            
            logs.append(log_entry)
            
            # Keep only last 1000 entries
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Save logs
            with open(self.automation_log_path, 'w') as f:
                json.dump(logs, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to log automation event: {e}")
    
    def get_recent_automation_logs(self, hours: int = 24) -> List[Dict]:
        """Get recent automation logs"""
        try:
            if not self.automation_log_path.exists():
                return []
            
            with open(self.automation_log_path, 'r') as f:
                logs = json.load(f)
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_logs = [
                log for log in logs
                if datetime.fromisoformat(log['timestamp']) > cutoff_time
            ]
            
            return recent_logs
            
        except Exception as e:
            logger.error(f"Failed to get recent logs: {e}")
            return []
    
    def get_automation_status(self) -> Dict:
        """Get current automation status with enhanced feature information"""
        try:
            status = {
                **self.automation_status,
                'monitoring_active': self.retraining_active,
                'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
                'in_cooldown': self.is_in_cooldown_period(),
                'daily_limit_exceeded': self.exceeded_daily_retraining_limit(),
                'queued_jobs': len(self.load_retraining_queue()),
                'recent_logs': self.get_recent_automation_logs(hours=6),
                'enhanced_features_status': {
                    'available': self.enhanced_features_available,
                    'preference': self.automation_config['retraining_conditions'].get('prefer_enhanced_features', True)
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get automation status: {e}")
            return {'error': str(e)}
    
    def trigger_manual_retraining(self, reason: str = "manual_trigger", use_enhanced: bool = None) -> Dict:
        """Manually trigger retraining with enhanced feature options"""
        try:
            # Use enhanced features by default if available
            if use_enhanced is None:
                use_enhanced = self.enhanced_features_available
            
            # Create manual trigger results
            trigger_results = {
                'should_retrain': True,
                'urgency': 'high',
                'trigger_reason': reason,
                'triggers_detected': [{
                    'type': 'manual_trigger',
                    'severity': 'high',
                    'message': f'Manual enhanced retraining triggered: {reason}'
                }],
                'recommendations': ['Manual enhanced retraining requested'],
                'enhanced_features_requested': use_enhanced
            }
            
            # Queue the retraining
            self.queue_retraining(trigger_results, forced=True)
            
            feature_info = " with enhanced features" if use_enhanced else " with standard features"
            return {
                'success': True, 
                'message': f'Manual enhanced retraining queued{feature_info}',
                'enhanced_features': use_enhanced
            }
            
        except Exception as e:
            logger.error(f"Manual enhanced retraining trigger failed: {e}")
            return {'success': False, 'error': str(e)}


def start_automation_system():
    """Start the enhanced automated retraining system"""
    try:
        automation_manager = AutomatedRetrainingManager()
        automation_manager.start_automated_monitoring()
        return automation_manager
    except Exception as e:
        logger.error(f"Failed to start enhanced automation system: {e}")
        return None

def get_automation_manager() -> Optional[AutomatedRetrainingManager]:
    """Get or create enhanced automation manager instance"""
    global _automation_manager
    
    if '_automation_manager' not in globals():
        _automation_manager = AutomatedRetrainingManager()
    
    return _automation_manager

def main():
    """Main execution function with enhanced CV and feature engineering"""
    retrainer = EnhancedModelRetrainer()
    success, message = retrainer.retrain_model()
    
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
        exit(1)

if __name__ == "__main__":
    main()