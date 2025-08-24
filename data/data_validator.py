# File: data/data_validator.py (NEW FILE)
# Comprehensive data validation pipeline with checkpoints and monitoring

import json
import time
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union
from pydantic import ValidationError
import hashlib
from collections import defaultdict, Counter

# Import validation schemas
from .validation_schemas import (
    NewsArticleSchema, TextContentSchema, LabelSchema, DataSourceSchema,
    BatchValidationSchema, ValidationResultSchema, BatchValidationResultSchema,
    ValidationLevel, TextQualityLevel, DataSource, NewsLabel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationCheckpoint:
    """Individual validation checkpoint for pipeline monitoring"""
    
    def __init__(self, name: str, description: str, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.name = name
        self.description = description
        self.validation_level = validation_level
        self.start_time = None
        self.end_time = None
        self.results = []
        self.errors = []
        self.warnings = []
        
    def start(self):
        """Start checkpoint timing"""
        self.start_time = time.time()
        logger.info(f"Starting validation checkpoint: {self.name}")
        
    def end(self):
        """End checkpoint timing"""
        self.end_time = time.time()
        duration = self.processing_time
        logger.info(f"Completed validation checkpoint: {self.name} ({duration:.2f}s)")
        
    def add_result(self, result: ValidationResultSchema):
        """Add validation result"""
        self.results.append(result)
        
    def add_error(self, error: str):
        """Add validation error"""
        self.errors.append(error)
        logger.error(f"Checkpoint {self.name}: {error}")
        
    def add_warning(self, warning: str):
        """Add validation warning"""
        self.warnings.append(warning)
        logger.warning(f"Checkpoint {self.name}: {warning}")
        
    @property
    def processing_time(self) -> float:
        """Calculate processing time"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
        
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if not self.results:
            return 0.0
        valid_count = sum(1 for result in self.results if result.is_valid)
        return valid_count / len(self.results)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'validation_level': self.validation_level.value,
            'processing_time': self.processing_time,
            'total_validations': len(self.results),
            'success_rate': self.success_rate,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings
        }


class DataValidationPipeline:
    """Comprehensive data validation pipeline with checkpoints and monitoring"""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path("/tmp")
        self.setup_paths()
        self.checkpoints = {}
        self.validation_history = []
        self.quality_stats = defaultdict(int)
        
    def setup_paths(self):
        """Setup validation paths"""
        self.logs_dir = self.base_path / "logs"
        self.validation_dir = self.base_path / "validation"
        self.cache_dir = self.base_path / "cache"
        
        # Create directories
        for path in [self.logs_dir, self.validation_dir, self.cache_dir]:
            path.mkdir(parents=True, exist_ok=True)
            
        # Setup file paths
        self.validation_log_path = self.logs_dir / "validation_log.json"
        self.validation_stats_path = self.validation_dir / "validation_statistics.json"
        self.failed_validations_path = self.validation_dir / "failed_validations.json"
        self.quality_report_path = self.validation_dir / "quality_report.json"
        
    def create_checkpoint(self, name: str, description: str, 
                         validation_level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationCheckpoint:
        """Create a new validation checkpoint"""
        checkpoint = ValidationCheckpoint(name, description, validation_level)
        self.checkpoints[name] = checkpoint
        return checkpoint
        
    def validate_single_article(self, text: str, label: int, source: str, 
                               validation_level: ValidationLevel = ValidationLevel.MODERATE,
                               **metadata) -> ValidationResultSchema:
        """Validate a single article with comprehensive checks"""
        
        start_time = time.time()
        errors = []
        warnings = []
        quality_metrics = {}
        
        try:
            # Create text content schema
            text_content = TextContentSchema(text=text)
            quality_metrics['word_count'] = text_content.word_count
            quality_metrics['character_count'] = text_content.character_count
            quality_metrics['sentence_count'] = text_content.sentence_count
            
        except ValidationError as e:
            for error in e.errors():
                errors.append(f"Text validation: {error['msg']}")
            
        try:
            # Create label schema
            label_info = LabelSchema(label=label)
            
        except ValidationError as e:
            for error in e.errors():
                errors.append(f"Label validation: {error['msg']}")
                
        try:
            # Create source schema
            source_info = DataSourceSchema(
                source=DataSource(source),
                timestamp=datetime.now(),
                **{k: v for k, v in metadata.items() if k in ['url', 'batch_id']}
            )
            
        except ValidationError as e:
            for error in e.errors():
                errors.append(f"Source validation: {error['msg']}")
                
        # Additional quality checks based on validation level
        if validation_level in [ValidationLevel.MODERATE, ValidationLevel.STRICT]:
            
            # Language detection (simplified)
            if text:
                english_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'as', 'was', 'on', 'are', 'you'}
                words = set(text.lower().split())
                english_ratio = len(words & english_words) / len(words) if words else 0
                
                if english_ratio < 0.1:
                    warnings.append("Text may not be in English")
                    
                quality_metrics['english_ratio'] = english_ratio
                
            # Content coherence check
            if text and len(text.split()) > 10:
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                if len(sentences) > 1:
                    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
                    quality_metrics['avg_sentence_length'] = avg_sentence_length
                    
                    if avg_sentence_length < 3:
                        warnings.append("Very short average sentence length")
                    elif avg_sentence_length > 50:
                        warnings.append("Very long average sentence length")
                        
        if validation_level == ValidationLevel.STRICT:
            
            # Advanced quality checks
            if text:
                # Check for AI-generated patterns (simplified)
                ai_indicators = ['as an ai', 'i am an artificial', 'generated by', 'chatgpt', 'gpt-3', 'gpt-4']
                if any(indicator in text.lower() for indicator in ai_indicators):
                    warnings.append("Text may be AI-generated")
                    
                # Check for template patterns
                template_patterns = [r'\{[^}]+\}', r'\[[^\]]+\]', r'<[^>]+>']
                import re
                for pattern in template_patterns:
                    if re.search(pattern, text):
                        warnings.append("Text contains template patterns")
                        break
                        
                # Check readability (simplified Flesch reading ease)
                words = text.split()
                sentences = len([s for s in text.split('.') if s.strip()])
                syllables = sum(max(1, len([c for c in word if c.lower() in 'aeiouy'])) for word in words)
                
                if sentences > 0 and words:
                    avg_sentence_length = len(words) / sentences
                    avg_syllables = syllables / len(words)
                    
                    # Simplified Flesch score
                    flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
                    quality_metrics['flesch_score'] = flesch_score
                    
                    if flesch_score < 30:
                        warnings.append("Text is very difficult to read")
                    elif flesch_score > 90:
                        warnings.append("Text is very easy to read (may be simplistic)")
                        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(quality_metrics, errors, warnings)
        quality_metrics['overall_quality_score'] = quality_score
        
        # Determine if validation passed
        is_valid = len(errors) == 0
        processing_time = time.time() - start_time
        
        return ValidationResultSchema(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            quality_metrics=quality_metrics,
            validation_level=validation_level,
            processing_time=processing_time
        )
        
    def validate_batch(self, articles_data: List[Dict[str, Any]], 
                      batch_id: Optional[str] = None,
                      validation_level: ValidationLevel = ValidationLevel.MODERATE) -> BatchValidationResultSchema:
        """Validate a batch of articles with comprehensive reporting"""
        
        if not batch_id:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(articles_data).encode()).hexdigest()[:8]}"
            
        logger.info(f"Starting batch validation: {batch_id} ({len(articles_data)} articles)")
        
        # Create validation checkpoint
        checkpoint = self.create_checkpoint(
            f"batch_validation_{batch_id}",
            f"Batch validation for {len(articles_data)} articles",
            validation_level
        )
        checkpoint.start()
        
        validation_results = []
        valid_count = 0
        invalid_count = 0
        quality_distribution = Counter()
        source_distribution = Counter()
        
        # Validate each article
        for i, article_data in enumerate(articles_data):
            try:
                text = article_data.get('text', '')
                label = article_data.get('label', 0)
                source = article_data.get('source', 'unknown')
                
                # Extract metadata
                metadata = {k: v for k, v in article_data.items() 
                          if k not in ['text', 'label', 'source']}
                
                # Validate article
                result = self.validate_single_article(
                    text, label, source, validation_level, **metadata
                )
                
                validation_results.append(result)
                checkpoint.add_result(result)
                
                if result.is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
                    
                # Update distributions
                quality_score = result.quality_metrics.get('overall_quality_score', 0)
                if quality_score >= 0.8:
                    quality_level = 'high'
                elif quality_score >= 0.6:
                    quality_level = 'medium'
                elif quality_score >= 0.4:
                    quality_level = 'low'
                else:
                    quality_level = 'invalid'
                    
                quality_distribution[quality_level] += 1
                source_distribution[source] += 1
                
            except Exception as e:
                error_msg = f"Failed to validate article {i}: {str(e)}"
                checkpoint.add_error(error_msg)
                invalid_count += 1
                
        checkpoint.end()
        
        # Calculate overall quality score
        if validation_results:
            quality_scores = [r.quality_metrics.get('overall_quality_score', 0) for r in validation_results]
            overall_quality_score = sum(quality_scores) / len(quality_scores)
        else:
            overall_quality_score = 0.0
            
        # Create validation summary
        validation_summary = {
            'batch_id': batch_id,
            'total_articles': len(articles_data),
            'validation_level': validation_level.value,
            'processing_time': checkpoint.processing_time,
            'success_rate': checkpoint.success_rate,
            'error_count': len(checkpoint.errors),
            'warning_count': len(checkpoint.warnings),
            'quality_metrics': {
                'average_quality_score': overall_quality_score,
                'quality_distribution': dict(quality_distribution),
                'source_distribution': dict(source_distribution)
            }
        }
        
        # Create batch validation result
        batch_result = BatchValidationResultSchema(
            batch_id=batch_id,
            total_articles=len(articles_data),
            valid_articles=valid_count,
            invalid_articles=invalid_count,
            validation_results=validation_results,
            overall_quality_score=overall_quality_score,
            quality_distribution=dict(quality_distribution),
            source_distribution=dict(source_distribution),
            validation_summary=validation_summary
        )
        
        # Log batch validation
        self._log_batch_validation(batch_result)
        
        # Update statistics
        self._update_validation_statistics(batch_result)
        
        logger.info(f"Batch validation completed: {batch_id} "
                   f"({valid_count}/{len(articles_data)} valid, "
                   f"quality: {overall_quality_score:.3f})")
        
        return batch_result
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          validation_level: ValidationLevel = ValidationLevel.MODERATE,
                          batch_id: Optional[str] = None) -> BatchValidationResultSchema:
        """Validate a pandas DataFrame"""
        
        # Convert DataFrame to list of dictionaries
        articles_data = df.to_dict('records')
        
        return self.validate_batch(articles_data, batch_id, validation_level)
    
    def validate_csv_file(self, file_path: Path, 
                         validation_level: ValidationLevel = ValidationLevel.MODERATE,
                         batch_id: Optional[str] = None) -> BatchValidationResultSchema:
        """Validate articles from a CSV file"""
        
        try:
            df = pd.read_csv(file_path)
            if batch_id is None:
                batch_id = f"csv_{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return self.validate_dataframe(df, validation_level, batch_id)
            
        except Exception as e:
            logger.error(f"Failed to validate CSV file {file_path}: {e}")
            raise
    
    def validate_scraped_data(self, scraped_data: List[Dict[str, Any]], 
                             source_name: str = "scraped_data") -> BatchValidationResultSchema:
        """Validate scraped data with specific checks for web content"""
        
        # Create checkpoint for scraped data validation
        checkpoint = self.create_checkpoint(
            f"scraped_validation_{source_name}",
            f"Validation for scraped data from {source_name}",
            ValidationLevel.MODERATE
        )
        checkpoint.start()
        
        # Add scraped-specific validation logic
        enhanced_data = []
        for item in scraped_data:
            # Ensure required fields
            if 'source' not in item:
                item['source'] = 'scraped_real'
            if 'label' not in item:
                item['label'] = 0  # Default to real for scraped news
                
            enhanced_data.append(item)
            
        result = self.validate_batch(
            enhanced_data, 
            f"scraped_{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            ValidationLevel.MODERATE
        )
        
        checkpoint.end()
        
        # Additional scraped data quality checks
        if result.overall_quality_score < 0.6:
            checkpoint.add_warning(f"Low quality scraped data: {result.overall_quality_score:.3f}")
        
        # Check for suspicious patterns in scraped data
        suspicious_count = 0
        for validation_result in result.validation_results:
            if any('suspicious' in warning.lower() for warning in validation_result.warnings):
                suspicious_count += 1
                
        if suspicious_count > len(scraped_data) * 0.1:  # More than 10% suspicious
            checkpoint.add_warning(f"High number of suspicious articles: {suspicious_count}/{len(scraped_data)}")
        
        return result
    
    def _calculate_quality_score(self, quality_metrics: Dict[str, Any], 
                                errors: List[str], warnings: List[str]) -> float:
        """Calculate overall quality score based on metrics and issues"""
        
        base_score = 1.0
        
        # Penalize for errors and warnings
        base_score -= len(errors) * 0.2
        base_score -= len(warnings) * 0.05
        
        # Adjust based on content metrics
        word_count = quality_metrics.get('word_count', 0)
        if word_count < 20:
            base_score -= 0.3
        elif word_count < 50:
            base_score -= 0.1
        elif word_count > 1000:
            base_score += 0.1
            
        # Adjust based on readability
        flesch_score = quality_metrics.get('flesch_score')
        if flesch_score:
            if 30 <= flesch_score <= 70:  # Good readability range
                base_score += 0.1
            elif flesch_score < 10 or flesch_score > 90:  # Poor readability
                base_score -= 0.15
                
        # Adjust based on English content ratio
        english_ratio = quality_metrics.get('english_ratio')
        if english_ratio:
            if english_ratio >= 0.3:
                base_score += 0.05
            else:
                base_score -= 0.1
                
        return max(0.0, min(1.0, base_score))
    
    def _log_batch_validation(self, batch_result: BatchValidationResultSchema):
        """Log batch validation results"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'batch_id': batch_result.batch_id,
                'total_articles': batch_result.total_articles,
                'valid_articles': batch_result.valid_articles,
                'success_rate': batch_result.success_rate,
                'overall_quality_score': batch_result.overall_quality_score,
                'validation_summary': batch_result.validation_summary
            }
            
            # Load existing logs
            logs = []
            if self.validation_log_path.exists():
                try:
                    with open(self.validation_log_path, 'r') as f:
                        logs = json.load(f)
                except:
                    logs = []
            
            logs.append(log_entry)
            
            # Keep only last 1000 entries
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Save logs
            with open(self.validation_log_path, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log batch validation: {e}")
    
    def _update_validation_statistics(self, batch_result: BatchValidationResultSchema):
        """Update validation statistics"""
        try:
            # Load existing stats
            stats = {}
            if self.validation_stats_path.exists():
                try:
                    with open(self.validation_stats_path, 'r') as f:
                        stats = json.load(f)
                except:
                    stats = {}
            
            # Initialize stats if empty
            if not stats:
                stats = {
                    'total_validations': 0,
                    'total_articles': 0,
                    'total_valid_articles': 0,
                    'average_quality_score': 0.0,
                    'validation_history': [],
                    'quality_trends': [],
                    'source_statistics': {},
                    'last_updated': None
                }
            
            # Update statistics
            stats['total_validations'] += 1
            stats['total_articles'] += batch_result.total_articles
            stats['total_valid_articles'] += batch_result.valid_articles
            
            # Update average quality score
            current_avg = stats['average_quality_score']
            total_validations = stats['total_validations']
            stats['average_quality_score'] = (
                (current_avg * (total_validations - 1) + batch_result.overall_quality_score) / 
                total_validations
            )
            
            # Add to history
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'batch_id': batch_result.batch_id,
                'quality_score': batch_result.overall_quality_score,
                'success_rate': batch_result.success_rate,
                'article_count': batch_result.total_articles
            }
            
            stats['validation_history'].append(history_entry)
            stats['quality_trends'].append({
                'timestamp': datetime.now().isoformat(),
                'quality_score': batch_result.overall_quality_score
            })
            
            # Keep only last 100 history entries
            if len(stats['validation_history']) > 100:
                stats['validation_history'] = stats['validation_history'][-100:]
            if len(stats['quality_trends']) > 100:
                stats['quality_trends'] = stats['quality_trends'][-100:]
            
            # Update source statistics
            for source, count in batch_result.source_distribution.items():
                if source not in stats['source_statistics']:
                    stats['source_statistics'][source] = {'total_articles': 0, 'total_validations': 0}
                
                stats['source_statistics'][source]['total_articles'] += count
                stats['source_statistics'][source]['total_validations'] += 1
            
            stats['last_updated'] = datetime.now().isoformat()
            
            # Save updated stats
            with open(self.validation_stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update validation statistics: {e}")
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get current validation statistics"""
        try:
            if self.validation_stats_path.exists():
                with open(self.validation_stats_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load validation statistics: {e}")
            return {}
    
    def get_validation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get validation history"""
        try:
            if self.validation_log_path.exists():
                with open(self.validation_log_path, 'r') as f:
                    logs = json.load(f)
                return logs[-limit:] if limit else logs
            return []
        except Exception as e:
            logger.error(f"Failed to load validation history: {e}")
            return []
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        try:
            stats = self.get_validation_statistics()
            
            if not stats:
                return {'error': 'No validation statistics available'}
            
            # Calculate trends
            quality_trends = stats.get('quality_trends', [])
            if len(quality_trends) >= 2:
                recent_scores = [t['quality_score'] for t in quality_trends[-10:]]
                older_scores = [t['quality_score'] for t in quality_trends[-20:-10]] if len(quality_trends) >= 20 else []
                
                recent_avg = sum(recent_scores) / len(recent_scores)
                older_avg = sum(older_scores) / len(older_scores) if older_scores else recent_avg
                
                quality_trend = recent_avg - older_avg
            else:
                quality_trend = 0.0
            
            # Generate report
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'overall_statistics': {
                    'total_validations': stats.get('total_validations', 0),
                    'total_articles': stats.get('total_articles', 0),
                    'overall_success_rate': (stats.get('total_valid_articles', 0) / 
                                           max(stats.get('total_articles', 1), 1)),
                    'average_quality_score': stats.get('average_quality_score', 0.0),
                    'quality_trend': quality_trend
                },
                'source_breakdown': stats.get('source_statistics', {}),
                'recent_performance': {
                    'last_10_validations': quality_trends[-10:] if quality_trends else [],
                    'recent_average_quality': (sum(t['quality_score'] for t in quality_trends[-10:]) / 
                                             len(quality_trends[-10:])) if quality_trends else 0.0
                },
                'quality_assessment': self._assess_overall_quality(stats),
                'recommendations': self._generate_recommendations(stats)
            }
            
            # Save report
            with open(self.quality_report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}")
            return {'error': str(e)}
    
    def _assess_overall_quality(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall data quality"""
        avg_quality = stats.get('average_quality_score', 0.0)
        success_rate = stats.get('total_valid_articles', 0) / max(stats.get('total_articles', 1), 1)
        
        if avg_quality >= 0.8 and success_rate >= 0.9:
            quality_level = 'excellent'
            quality_color = 'green'
        elif avg_quality >= 0.6 and success_rate >= 0.8:
            quality_level = 'good'
            quality_color = 'blue'
        elif avg_quality >= 0.4 and success_rate >= 0.6:
            quality_level = 'fair'
            quality_color = 'yellow'
        else:
            quality_level = 'poor'
            quality_color = 'red'
        
        return {
            'quality_level': quality_level,
            'quality_color': quality_color,
            'average_score': avg_quality,
            'success_rate': success_rate,
            'assessment': f"Data quality is {quality_level} with {success_rate:.1%} validation success rate"
        }
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        avg_quality = stats.get('average_quality_score', 0.0)
        success_rate = stats.get('total_valid_articles', 0) / max(stats.get('total_articles', 1), 1)
        
        if avg_quality < 0.6:
            recommendations.append("Improve data source quality - consider additional content filters")
        
        if success_rate < 0.8:
            recommendations.append("Review validation criteria - high failure rate detected")
        
        source_stats = stats.get('source_statistics', {})
        if source_stats:
            # Find problematic sources
            for source, source_info in source_stats.items():
                if source_info.get('total_articles', 0) > 10:  # Only check sources with enough data
                    # This is simplified - in practice you'd track success rates per source
                    pass
        
        if len(recommendations) == 0:
            recommendations.append("Data quality is satisfactory - continue current practices")
        
        return recommendations
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old validation logs"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean validation logs
            if self.validation_log_path.exists():
                with open(self.validation_log_path, 'r') as f:
                    logs = json.load(f)
                
                filtered_logs = []
                for log in logs:
                    try:
                        log_date = datetime.fromisoformat(log['timestamp'])
                        if log_date > cutoff_date:
                            filtered_logs.append(log)
                    except:
                        # Keep logs with invalid timestamps
                        filtered_logs.append(log)
                
                with open(self.validation_log_path, 'w') as f:
                    json.dump(filtered_logs, f, indent=2)
                
                logger.info(f"Cleaned up validation logs: kept {len(filtered_logs)}/{len(logs)} entries")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")


# Convenience functions for external use
def validate_text(text: str, label: int, source: str = "user_input", 
                 validation_level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationResultSchema:
    """Validate a single text input"""
    validator = DataValidationPipeline()
    return validator.validate_single_article(text, label, source, validation_level)


def validate_articles_list(articles: List[Dict[str, Any]], 
                          validation_level: ValidationLevel = ValidationLevel.MODERATE) -> BatchValidationResultSchema:
    """Validate a list of articles"""
    validator = DataValidationPipeline()
    return validator.validate_batch(articles, validation_level=validation_level)


def validate_csv(file_path: str, 
                validation_level: ValidationLevel = ValidationLevel.MODERATE) -> BatchValidationResultSchema:
    """Validate articles from a CSV file"""
    validator = DataValidationPipeline()
    return validator.validate_csv_file(Path(file_path), validation_level)


def get_validation_stats() -> Dict[str, Any]:
    """Get current validation statistics"""
    validator = DataValidationPipeline()
    return validator.get_validation_statistics()


def generate_quality_report() -> Dict[str, Any]:
    """Generate quality report"""
    validator = DataValidationPipeline()
    return validator.generate_quality_report()