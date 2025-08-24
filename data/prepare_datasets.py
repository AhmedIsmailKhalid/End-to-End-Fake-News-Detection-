import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
import hashlib
import json
from datetime import datetime
from data.data_validator import DataValidationPipeline
from data.validation_schemas import ValidationLevel, DataSource
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetPreparer:
    """Robust dataset preparation with comprehensive validation and error handling"""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self.setup_paths()
    
    def setup_paths(self):
        """Setup all necessary paths"""
        # Input paths
        self.kaggle_fake = self.base_dir / "kaggle" / "Fake.csv"
        self.kaggle_real = self.base_dir / "kaggle" / "True.csv"
        self.liar_paths = [
            self.base_dir / "liar" / "train.tsv",
            self.base_dir / "liar" / "test.tsv", 
            self.base_dir / "liar" / "valid.tsv"
        ]
        
        # Output paths
        self.output_dir = Path("/tmp/data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / "combined_dataset.csv"
        self.metadata_path = self.output_dir / "dataset_metadata.json"
    
    def validate_text_quality(self, text: str) -> bool:
        """Validate text quality with comprehensive checks"""
        if not isinstance(text, str):
            return False
        
        text = text.strip()
        
        # Basic length check
        if len(text) < 10:
            return False
        
        # Check for meaningful content
        if not any(c.isalpha() for c in text):
            return False
        
        # Check for sentence structure
        if not any(punct in text for punct in '.!?'):
            return False
        
        # Check for excessive repetition
        words = text.lower().split()
        if len(words) > 0:
            most_common_word_count = max(words.count(word) for word in set(words))
            if most_common_word_count > len(words) * 0.5:  # More than 50% repetition
                return False
        
        # Check for excessive special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.3:  # More than 30% special characters
            return False
        
        return True
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '...', text)
        
        # Remove non-printable characters
        text = ''.join(char for char in text if ord(char) >= 32)
        
        return text.strip()
    
    def load_kaggle_dataset(self) -> Optional[pd.DataFrame]:
        """Load and process Kaggle dataset with error handling"""
        try:
            logger.info("Loading Kaggle dataset...")
            
            # Check if files exist
            if not self.kaggle_fake.exists() or not self.kaggle_real.exists():
                logger.warning("Kaggle dataset files not found")
                return None
            
            # Load datasets
            df_fake = pd.read_csv(self.kaggle_fake)
            df_real = pd.read_csv(self.kaggle_real)
            
            logger.info(f"Loaded {len(df_fake)} fake and {len(df_real)} real articles from Kaggle")
            
            # Process fake news
            df_fake['label'] = 1
            df_fake['text'] = df_fake['title'].fillna('') + ". " + df_fake['text'].fillna('')
            df_fake['source'] = 'kaggle_fake'
            
            # Process real news
            df_real['label'] = 0
            df_real['text'] = df_real['title'].fillna('') + ". " + df_real['text'].fillna('')
            df_real['source'] = 'kaggle_real'
            
            # Combine datasets
            df_combined = pd.concat([
                df_fake[['text', 'label', 'source']],
                df_real[['text', 'label', 'source']]
            ], ignore_index=True)
            
            logger.info(f"Combined Kaggle dataset: {len(df_combined)} samples")
            return self.validate_dataset_with_schemas(df_combined, 'kaggle_combined')
            
        except Exception as e:
            logger.error(f"Error loading Kaggle dataset: {e}")
            return None
    
    def load_liar_dataset(self) -> Optional[pd.DataFrame]:
        """Load and process LIAR dataset with robust error handling"""
        try:
            logger.info("Loading LIAR dataset...")
            
            liar_dfs = []
            total_processed = 0
            
            for path in self.liar_paths:
                if not path.exists():
                    logger.warning(f"LIAR file not found: {path}")
                    continue
                
                try:
                    # Read TSV with flexible parameters
                    df = pd.read_csv(
                        path, 
                        sep='\t', 
                        header=None, 
                        quoting=3,
                        on_bad_lines='skip',
                        low_memory=False
                    )
                    
                    # Expected columns for LIAR dataset
                    expected_columns = [
                        'id', 'label_text', 'statement', 'subject', 'speaker', 'job',
                        'state', 'party', 'barely_true', 'false', 'half_true', 
                        'mostly_true', 'pants_on_fire', 'context'
                    ]
                    
                    # Handle different column counts
                    if len(df.columns) >= 3:
                        df.columns = expected_columns[:len(df.columns)]
                        
                        # Map labels to binary classification
                        if 'label_text' in df.columns:
                            df['label'] = df['label_text'].apply(
                                lambda x: 1 if str(x).lower() in ['false', 'pants-fire', 'barely-true'] else 0
                            )
                        else:
                            continue
                        
                        # Extract text
                        if 'statement' in df.columns:
                            df['text'] = df['statement'].astype(str)
                        else:
                            continue
                        
                        df['source'] = f'liar_{path.stem}'
                        
                        processed_df = df[['text', 'label', 'source']].copy()
                        liar_dfs.append(processed_df)
                        total_processed += len(processed_df)
                        
                        logger.info(f"Processed {len(processed_df)} samples from {path.name}")
                    
                except Exception as e:
                    logger.error(f"Error processing LIAR file {path}: {e}")
                    continue
            
            if liar_dfs:
                combined_liar = pd.concat(liar_dfs, ignore_index=True)
                logger.info(f"Combined LIAR dataset: {len(combined_liar)} samples")
                return self.validate_dataset_with_schemas(combined_liar, 'liar_combined')
            else:
                logger.warning("No LIAR data could be processed")
                return None
                
        except Exception as e:
            logger.error(f"Error loading LIAR dataset: {e}")
            return None
    
    def validate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive dataset validation and cleaning"""
        logger.info("Starting dataset validation...")
        
        initial_count = len(df)
        
        # Remove null texts
        df = df.dropna(subset=['text'])
        logger.info(f"Removed {initial_count - len(df)} null text entries")
        
        # Clean text
        df['text'] = df['text'].apply(self.clean_text)
        
        # Validate text quality
        valid_mask = df['text'].apply(self.validate_text_quality)
        df = df[valid_mask]
        # logger.info(f"Removed {initial_count - len(valid_mask.sum())} low-quality texts")
        logger.info(f"Removed {initial_count - valid_mask.sum()} low-quality texts")
        
        # Remove duplicates
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['text'])
        logger.info(f"Removed {before_dedup - len(df)} duplicate texts")
        
        # Validate label distribution
        label_counts = df['label'].value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")
        
        # Check for balance
        if len(label_counts) > 1:
            balance_ratio = label_counts.min() / label_counts.max()
            if balance_ratio < 0.3:
                logger.warning(f"Dataset is imbalanced (ratio: {balance_ratio:.2f})")
        
        # Add metadata
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        df['processed_timestamp'] = datetime.now().isoformat()
        
        return df
    
    def generate_dataset_metadata(self, df: pd.DataFrame) -> dict:
        """Generate comprehensive dataset metadata"""
        metadata = {
            'total_samples': len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'source_distribution': df['source'].value_counts().to_dict(),
            'text_length_stats': {
                'mean': float(df['text_length'].mean()),
                'std': float(df['text_length'].std()),
                'min': int(df['text_length'].min()),
                'max': int(df['text_length'].max()),
                'median': float(df['text_length'].median())
            },
            'word_count_stats': {
                'mean': float(df['word_count'].mean()),
                'std': float(df['word_count'].std()),
                'min': int(df['word_count'].min()),
                'max': int(df['word_count'].max()),
                'median': float(df['word_count'].median())
            },
            'data_hash': hashlib.md5(df['text'].str.cat().encode()).hexdigest(),
            'creation_timestamp': datetime.now().isoformat(),
            'quality_score': self.calculate_quality_score(df)
        }
        
        return metadata
    
    def calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall dataset quality score"""
        scores = []
        
        # Balance score
        label_counts = df['label'].value_counts()
        if len(label_counts) > 1:
            balance_score = label_counts.min() / label_counts.max()
            scores.append(balance_score)
        
        # Diversity score (based on unique text ratio)
        diversity_score = df['text'].nunique() / len(df)
        scores.append(diversity_score)
        
        # Length consistency score
        text_lengths = df['text_length']
        length_cv = text_lengths.std() / text_lengths.mean()  # Coefficient of variation
        length_score = max(0, 1 - length_cv / 2)  # Normalize to 0-1
        scores.append(length_score)
        
        return float(np.mean(scores))
    
    def prepare_datasets(self) -> Tuple[bool, str]:
        """Main method to prepare all datasets with validation"""
        logger.info("Starting dataset preparation with validation...")
        
        try:
            # Load and validate datasets
            kaggle_result = self.load_kaggle_dataset()
            liar_result = self.load_liar_dataset()
            
            # Handle None returns gracefully
            if kaggle_result is None:
                logger.warning("Kaggle dataset loading failed")
                kaggle_df, kaggle_validation = pd.DataFrame(), {
                    'source': 'kaggle_combined', 'original_count': 0, 'valid_count': 0, 
                    'success_rate': 0, 'overall_quality_score': 0, 'validation_timestamp': datetime.now().isoformat()
                }
            else:
                kaggle_df, kaggle_validation = kaggle_result
            
            if liar_result is None:
                logger.warning("LIAR dataset loading failed")
                liar_df, liar_validation = pd.DataFrame(), {
                    'source': 'liar_combined', 'original_count': 0, 'valid_count': 0, 
                    'success_rate': 0, 'overall_quality_score': 0, 'validation_timestamp': datetime.now().isoformat()
                }
            else:
                liar_df, liar_validation = liar_result
            
            # Combine datasets
            datasets_to_combine = [df for df in [kaggle_df, liar_df] if not df.empty]
            
            if not datasets_to_combine:
                return False, "No datasets could be loaded and validated"
            
            combined_df = pd.concat(datasets_to_combine, ignore_index=True)
            
            # Save combined dataset
            combined_df.to_csv(self.output_path, index=False)
            
            # Save validation reports
            total_original = kaggle_validation['original_count'] + liar_validation['original_count']
            validation_report = {
                'datasets': {
                    'kaggle': kaggle_validation,
                    'liar': liar_validation
                },
                'combined_stats': {
                    'total_articles': len(combined_df),
                    'total_original': total_original,
                    'overall_success_rate': len(combined_df) / max(1, total_original),
                    'validation_timestamp': datetime.now().isoformat()
                }
            }
            
            validation_report_path = self.output_dir / "dataset_validation_report.json"
            with open(validation_report_path, 'w') as f:
                json.dump(validation_report, f, indent=2)
            
            logger.info(f"Dataset preparation complete. Validation report saved to {validation_report_path}")
            return True, f"Successfully prepared {len(combined_df)} validated articles"
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            return False, f"Dataset preparation failed: {str(e)}"

    def validate_dataset_with_schemas(self, df: pd.DataFrame, source_name: str) -> Tuple[pd.DataFrame, Dict]:
        """Validate dataset using comprehensive schemas"""
        logger.info(f"Starting schema validation for {source_name}...")
        
        validator = DataValidationPipeline()
        
        # Convert DataFrame to validation format
        articles_data = []
        for _, row in df.iterrows():
            article_data = {
                'text': str(row.get('text', '')),
                'label': int(row.get('label', 0)),
                'source': source_name
            }
            
            if 'title' in row and pd.notna(row['title']):
                article_data['title'] = str(row['title'])
            if 'url' in row and pd.notna(row['url']):
                article_data['url'] = str(row['url'])
                
            articles_data.append(article_data)
        
        # Perform batch validation
        validation_result = validator.validate_batch(
            articles_data, 
            batch_id=f"{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            validation_level=ValidationLevel.MODERATE
        )
        
        # Filter valid articles and add quality scores
        valid_indices = [i for i, result in enumerate(validation_result.validation_results) if result.is_valid]
        
        if valid_indices:
            valid_df = df.iloc[valid_indices].copy()
            quality_scores = [validation_result.validation_results[i].quality_metrics.get('overall_quality_score', 0.0) 
                             for i in valid_indices]
            valid_df['validation_quality_score'] = quality_scores
            valid_df['validation_timestamp'] = datetime.now().isoformat()
        else:
            valid_df = pd.DataFrame(columns=df.columns)
        
        validation_summary = {
            'source': source_name,
            'original_count': len(df),
            'valid_count': len(valid_df),
            'success_rate': validation_result.success_rate,
            'overall_quality_score': validation_result.overall_quality_score,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        return valid_df, validation_summary
    
    
    
def main():
    """Main execution function"""
    preparer = DatasetPreparer()
    success, message = preparer.prepare_datasets()
    
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
        exit(1)

if __name__ == "__main__":
    main()
    