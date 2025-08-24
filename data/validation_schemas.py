# File: data/validation_schemas.py
# Comprehensive Pydantic validation schemas for data quality assurance

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import re
import hashlib
from enum import Enum


class ValidationLevel(str, Enum):
    """Validation strictness levels"""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class TextQualityLevel(str, Enum):
    """Text quality assessment levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"


class DataSource(str, Enum):
    """Valid data sources"""
    KAGGLE_FAKE = "kaggle_fake"
    KAGGLE_REAL = "kaggle_real"
    LIAR_TRAIN = "liar_train"
    LIAR_TEST = "liar_test"
    LIAR_VALID = "liar_valid"
    SCRAPED_REAL = "scraped_real"
    GENERATED_FAKE = "generated_fake"
    CUSTOM_UPLOAD = "custom_upload"
    USER_INPUT = "user_input"


class NewsLabel(int, Enum):
    """Valid news labels"""
    REAL = 0
    FAKE = 1


class TextContentSchema(BaseModel):
    """Comprehensive text content validation schema"""
    
    text: str = Field(
        ...,
        min_length=10,
        max_length=50000,
        description="The news article text content"
    )
    
    @field_validator('text')
    @classmethod
    def validate_text_content(cls, v):
        """Comprehensive text content validation"""
        if not v or not isinstance(v, str):
            raise ValueError("Text must be a non-empty string")
        
        # Strip and normalize whitespace
        v = re.sub(r'\s+', ' ', v.strip())
        
        # Basic length check
        if len(v) < 10:
            raise ValueError("Text must be at least 10 characters long")
        
        if len(v) > 50000:
            raise ValueError("Text exceeds maximum length of 50,000 characters")
        
        # Check for meaningful content
        if not any(c.isalpha() for c in v):
            raise ValueError("Text must contain alphabetic characters")
        
        # Check for sentence structure
        if not any(punct in v for punct in '.!?'):
            raise ValueError("Text must contain sentence-ending punctuation")
        
        # Check for excessive repetition
        words = v.lower().split()
        if len(words) > 0:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            most_common_count = max(word_counts.values())
            if most_common_count > len(words) * 0.5:
                raise ValueError("Text contains excessive word repetition")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                # JavaScript URLs
            r'data:text/html',            # Data URLs
            r'<iframe[^>]*>.*?</iframe>', # Iframes
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE | re.DOTALL):
                raise ValueError("Text contains potentially malicious content")
        
        # Check for excessive special characters
        special_char_count = sum(1 for c in v if not c.isalnum() and not c.isspace() and c not in '.,!?;:-')
        if special_char_count > len(v) * 0.3:
            raise ValueError("Text contains excessive special characters")
        
        # Check for minimum word count
        if len(words) < 5:
            raise ValueError("Text must contain at least 5 words")
        
        return v
    
    @property
    def word_count(self) -> int:
        """Calculate word count"""
        return len(self.text.split())
    
    @property
    def character_count(self) -> int:
        """Calculate character count"""
        return len(self.text)
    
    @property
    def sentence_count(self) -> int:
        """Estimate sentence count"""
        sentences = re.split(r'[.!?]+', self.text)
        return len([s for s in sentences if s.strip()])
    
    @property
    def text_hash(self) -> str:
        """Generate text hash for deduplication"""
        return hashlib.md5(self.text.encode()).hexdigest()


class LabelSchema(BaseModel):
    """News label validation schema"""
    
    label: NewsLabel = Field(
        ...,
        description="News label: 0 for real, 1 for fake"
    )
    
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Label confidence score (0-1)"
    )
    
    source_reliability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Source reliability score (0-1)"
    )
    
    @field_validator('label')
    @classmethod
    def validate_label(cls, v):
        """Validate label value"""
        if v not in [0, 1]:
            raise ValueError("Label must be 0 (real) or 1 (fake)")
        return v


class DataSourceSchema(BaseModel):
    """Data source validation schema"""
    
    source: DataSource = Field(
        ...,
        description="Data source identifier"
    )
    
    url: Optional[str] = Field(
        None,
        max_length=2048,
        description="Source URL if applicable"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Data collection timestamp"
    )
    
    batch_id: Optional[str] = Field(
        None,
        max_length=64,
        description="Batch identifier for grouped data"
    )
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        """Validate URL format"""
        if v is not None:
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            
            if not url_pattern.match(v):
                raise ValueError("Invalid URL format")
        
        return v


class NewsArticleSchema(BaseModel):
    """Complete news article validation schema"""
    
    text_content: TextContentSchema
    label_info: LabelSchema
    source_info: DataSourceSchema
    
    # Additional metadata
    title: Optional[str] = Field(
        None,
        max_length=500,
        description="Article title"
    )
    
    author: Optional[str] = Field(
        None,
        max_length=200,
        description="Article author"
    )
    
    publication_date: Optional[datetime] = Field(
        None,
        description="Original publication date"
    )
    
    language: str = Field(
        default="en",
        max_length=5,
        description="Article language code"
    )
    
    quality_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Overall quality score (0-1)"
    )
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        """Validate article title"""
        if v is not None:
            v = v.strip()
            if len(v) < 3:
                raise ValueError("Title must be at least 3 characters long")
            
            # Check for excessive special characters
            special_char_count = sum(1 for c in v if not c.isalnum() and not c.isspace() and c not in '.,!?;:-')
            if special_char_count > len(v) * 0.4:
                raise ValueError("Title contains excessive special characters")
        
        return v
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v):
        """Validate language code"""
        valid_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'ru', 'zh', 'ja', 'ko']
        if v not in valid_languages:
            raise ValueError(f"Unsupported language code: {v}")
        return v
    
    @model_validator(mode='after')
    def validate_article_consistency(self):
        """Cross-field validation"""
        text_content = self.text_content
        title = self.title
        
        if text_content and title:
            # Check if title and content are suspiciously similar
            title_words = set(title.lower().split()) if title else set()
            text_words = set(text_content.text.lower().split()[:50])  # First 50 words
            
            if title_words and len(title_words & text_words) / len(title_words) > 0.9:
                # This is fine, just noting high similarity
                pass
        
        return self
    
    @property
    def text_quality_level(self) -> TextQualityLevel:
        """Assess text quality level"""
        score = 0
        
        # Length score
        word_count = self.text_content.word_count
        if word_count >= 100:
            score += 3
        elif word_count >= 50:
            score += 2
        elif word_count >= 20:
            score += 1
        
        # Structure score
        sentence_count = self.text_content.sentence_count
        if sentence_count >= 5:
            score += 2
        elif sentence_count >= 3:
            score += 1
        
        # Content diversity score
        words = self.text_content.text.lower().split()
        unique_words = len(set(words))
        diversity_ratio = unique_words / len(words) if words else 0
        
        if diversity_ratio >= 0.7:
            score += 2
        elif diversity_ratio >= 0.5:
            score += 1
        
        # Quality assessment
        if score >= 6:
            return TextQualityLevel.HIGH
        elif score >= 4:
            return TextQualityLevel.MEDIUM
        elif score >= 2:
            return TextQualityLevel.LOW
        else:
            return TextQualityLevel.INVALID


class BatchValidationSchema(BaseModel):
    """Schema for batch data validation"""
    
    batch_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique batch identifier"
    )
    
    articles: List[NewsArticleSchema] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of articles to validate"
    )
    
    validation_level: ValidationLevel = Field(
        default=ValidationLevel.MODERATE,
        description="Validation strictness level"
    )
    
    source_filter: Optional[List[DataSource]] = Field(
        None,
        description="Filter by specific data sources"
    )
    
    quality_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum quality score threshold"
    )
    
    @field_validator('articles')
    @classmethod
    def validate_article_list(cls, v):
        """Validate article list"""
        if not v:
            raise ValueError("Articles list cannot be empty")
        
        # Check for duplicates
        text_hashes = [article.text_content.text_hash for article in v]
        if len(text_hashes) != len(set(text_hashes)):
            raise ValueError("Duplicate articles detected in batch")
        
        return v
    
    @property
    def total_articles(self) -> int:
        """Total number of articles in batch"""
        return len(self.articles)
    
    @property
    def quality_distribution(self) -> Dict[TextQualityLevel, int]:
        """Distribution of quality levels"""
        distribution = {level: 0 for level in TextQualityLevel}
        for article in self.articles:
            distribution[article.text_quality_level] += 1
        return distribution
    
    @property
    def source_distribution(self) -> Dict[DataSource, int]:
        """Distribution of data sources"""
        distribution = {}
        for article in self.articles:
            source = article.source_info.source
            distribution[source] = distribution.get(source, 0) + 1
        return distribution


class ValidationResultSchema(BaseModel):
    """Validation result schema"""
    
    is_valid: bool = Field(
        ...,
        description="Overall validation result"
    )
    
    validation_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Validation timestamp"
    )
    
    errors: List[str] = Field(
        default_factory=list,
        description="List of validation errors"
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="List of validation warnings"
    )
    
    quality_metrics: Dict[str, Union[int, float]] = Field(
        default_factory=dict,
        description="Quality metrics and statistics"
    )
    
    validation_level: ValidationLevel = Field(
        default=ValidationLevel.MODERATE,
        description="Validation level used"
    )
    
    processing_time: Optional[float] = Field(
        None,
        ge=0.0,
        description="Validation processing time in seconds"
    )
    
    @property
    def has_errors(self) -> bool:
        """Check if validation has errors"""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings"""
        return len(self.warnings) > 0
    
    @property
    def error_count(self) -> int:
        """Number of validation errors"""
        return len(self.errors)
    
    @property
    def warning_count(self) -> int:
        """Number of validation warnings"""
        return len(self.warnings)


class BatchValidationResultSchema(BaseModel):
    """Batch validation result schema"""
    
    batch_id: str = Field(
        ...,
        description="Batch identifier"
    )
    
    total_articles: int = Field(
        ...,
        ge=0,
        description="Total articles processed"
    )
    
    valid_articles: int = Field(
        ...,
        ge=0,
        description="Number of valid articles"
    )
    
    invalid_articles: int = Field(
        ...,
        ge=0,
        description="Number of invalid articles"
    )
    
    validation_results: List[ValidationResultSchema] = Field(
        default_factory=list,
        description="Individual validation results"
    )
    
    overall_quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall batch quality score"
    )
    
    quality_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Quality level distribution"
    )
    
    source_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Source distribution"
    )
    
    validation_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validation summary statistics"
    )
    
    @field_validator('valid_articles', 'invalid_articles')
    @classmethod
    def validate_article_counts(cls, v, info):
        """Validate article count consistency"""
        if 'total_articles' in info.data:
            total = info.data['total_articles']
            if v > total:
                raise ValueError("Article count cannot exceed total")
        return v
    
    @model_validator(mode='after')
    def validate_counts_consistency(self):
        """Validate count consistency"""
        total = self.total_articles
        valid = self.valid_articles
        invalid = self.invalid_articles
        
        if valid + invalid != total:
            raise ValueError("Valid + invalid articles must equal total articles")
        
        return self
    
    @property
    def success_rate(self) -> float:
        """Calculate validation success rate"""
        if self.total_articles == 0:
            return 0.0
        return self.valid_articles / self.total_articles
    
    @property
    def failure_rate(self) -> float:
        """Calculate validation failure rate"""
        return 1.0 - self.success_rate