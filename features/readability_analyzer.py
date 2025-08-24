# features/readability_analyzer.py
# Readability and Linguistic Complexity Analysis Component

import numpy as np
import pandas as pd
import re
import logging
from typing import List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ReadabilityAnalyzer(BaseEstimator, TransformerMixin):
    """
    Advanced readability and linguistic complexity analyzer.
    Detects patterns in text complexity that may indicate misinformation tactics.
    """
    
    def __init__(self):
        self.is_fitted_ = False
        
    def fit(self, X, y=None):
        """Fit the readability analyzer (for API consistency)"""
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Extract readability and complexity features"""
        if not self.is_fitted_:
            raise ValueError("ReadabilityAnalyzer must be fitted before transform")
        
        # Convert input to array if needed
        if isinstance(X, pd.Series):
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)
        
        features = []
        
        for text in X:
            text_features = self._extract_readability_features(str(text))
            features.append(text_features)
        
        return np.array(features)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def _extract_readability_features(self, text):
        """Extract comprehensive readability features"""
        # Basic text statistics
        sentences = self._split_sentences(text)
        words = self._split_words(text)
        syllables = self._count_syllables_total(words)
        
        # Handle edge cases
        if len(sentences) == 0 or len(words) == 0:
            return [0.0] * 15
        
        features = []
        
        # Basic metrics
        avg_words_per_sentence = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        avg_chars_per_word = sum(len(word) for word in words) / len(words)
        
        features.extend([avg_words_per_sentence, avg_syllables_per_word, avg_chars_per_word])
        
        # Readability scores
        flesch_reading_ease = self._calculate_flesch_reading_ease(words, sentences, syllables)
        flesch_kincaid_grade = self._calculate_flesch_kincaid_grade(words, sentences, syllables)
        automated_readability_index = self._calculate_ari(words, sentences, text)
        
        features.extend([flesch_reading_ease, flesch_kincaid_grade, automated_readability_index])
        
        # Complexity indicators
        complex_words_ratio = self._calculate_complex_words_ratio(words)
        long_words_ratio = self._calculate_long_words_ratio(words)
        technical_terms_ratio = self._calculate_technical_terms_ratio(words)
        
        features.extend([complex_words_ratio, long_words_ratio, technical_terms_ratio])
        
        # Sentence structure complexity
        sentence_length_variance = self._calculate_sentence_length_variance(sentences)
        punctuation_density = self._calculate_punctuation_density(text)
        subordinate_clause_ratio = self._calculate_subordinate_clause_ratio(text)
        
        features.extend([sentence_length_variance, punctuation_density, subordinate_clause_ratio])
        
        # Vocabulary sophistication
        unique_word_ratio = self._calculate_unique_word_ratio(words)
        rare_word_ratio = self._calculate_rare_word_ratio(words)
        formal_language_ratio = self._calculate_formal_language_ratio(words)
        
        features.extend([unique_word_ratio, rare_word_ratio, formal_language_ratio])
        
        return features
    
    def _split_sentences(self, text):
        """Split text into sentences"""
        # Simple sentence splitting - could be enhanced with NLTK
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _split_words(self, text):
        """Split text into words"""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words
    
    def _count_syllables(self, word):
        """Count syllables in a word (approximation)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)  # Every word has at least 1 syllable
    
    def _count_syllables_total(self, words):
        """Count total syllables in word list"""
        return sum(self._count_syllables(word) for word in words)
    
    def _calculate_flesch_reading_ease(self, words, sentences, syllables):
        """Calculate Flesch Reading Ease score"""
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, score))  # Clamp between 0-100
    
    def _calculate_flesch_kincaid_grade(self, words, sentences, syllables):
        """Calculate Flesch-Kincaid Grade Level"""
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        return max(0, grade)
    
    def _calculate_ari(self, words, sentences, text):
        """Calculate Automated Readability Index"""
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        chars = len(re.sub(r'\s+', '', text))
        avg_chars_per_word = chars / len(words)
        avg_words_per_sentence = len(words) / len(sentences)
        
        ari = (4.71 * avg_chars_per_word) + (0.5 * avg_words_per_sentence) - 21.43
        return max(0, ari)
    
    def _calculate_complex_words_ratio(self, words):
        """Calculate ratio of complex words (3+ syllables)"""
        if not words:
            return 0
        
        complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
        return complex_words / len(words)
    
    def _calculate_long_words_ratio(self, words):
        """Calculate ratio of long words (7+ characters)"""
        if not words:
            return 0
        
        long_words = sum(1 for word in words if len(word) >= 7)
        return long_words / len(words)
    
    def _calculate_technical_terms_ratio(self, words):
        """Calculate ratio of potentially technical terms"""
        if not words:
            return 0
        
        # Heuristics for technical terms
        technical_indicators = {
            'tion', 'sion', 'ment', 'ness', 'ance', 'ence', 'ism', 'ist',
            'ogy', 'ics', 'phy', 'logical', 'ical', 'ative', 'itive'
        }
        
        technical_words = 0
        for word in words:
            if (len(word) > 6 and 
                any(word.endswith(suffix) for suffix in technical_indicators)):
                technical_words += 1
        
        return technical_words / len(words)
    
    def _calculate_sentence_length_variance(self, sentences):
        """Calculate variance in sentence lengths"""
        if len(sentences) <= 1:
            return 0
        
        lengths = [len(sentence.split()) for sentence in sentences]
        mean_length = sum(lengths) / len(lengths)
        variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
        
        return variance
    
    def _calculate_punctuation_density(self, text):
        """Calculate density of punctuation marks"""
        if not text:
            return 0
        
        punctuation_marks = re.findall(r'[.,;:!?()-"]', text)
        return len(punctuation_marks) / len(text)
    
    def _calculate_subordinate_clause_ratio(self, text):
        """Calculate ratio of subordinate clauses (approximation)"""
        if not text:
            return 0
        
        # Look for subordinating conjunctions and relative pronouns
        subordinate_indicators = [
            'although', 'because', 'since', 'while', 'whereas', 'if', 'unless',
            'when', 'whenever', 'where', 'wherever', 'that', 'which', 'who',
            'whom', 'whose', 'after', 'before', 'until', 'as'
        ]
        
        text_lower = text.lower()
        subordinate_count = sum(text_lower.count(f' {indicator} ') for indicator in subordinate_indicators)
        sentences = self._split_sentences(text)
        
        return subordinate_count / len(sentences) if sentences else 0
    
    def _calculate_unique_word_ratio(self, words):
        """Calculate ratio of unique words (lexical diversity)"""
        if not words:
            return 0
        
        unique_words = len(set(words))
        return unique_words / len(words)
    
    def _calculate_rare_word_ratio(self, words):
        """Calculate ratio of rare/uncommon words"""
        if not words:
            return 0
        
        # Common English words (top 1000 most frequent)
        common_words = {
            'the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it',
            'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they',
            'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had',
            'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when',
            'your', 'can', 'said', 'there', 'each', 'which', 'she', 'do',
            'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out',
            'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would',
            'make', 'like', 'into', 'him', 'has', 'two', 'more', 'very',
            'after', 'words', 'first', 'where', 'much', 'through', 'back',
            'years', 'work', 'came', 'right', 'used', 'take', 'three',
            'states', 'himself', 'few', 'house', 'use', 'during', 'without',
            'again', 'place', 'around', 'however', 'small', 'found', 'mrs',
            'thought', 'went', 'say', 'part', 'once', 'general', 'high',
            'upon', 'school', 'every', 'don', 'does', 'got', 'united',
            'left', 'number', 'course', 'war', 'until', 'always', 'away',
            'something', 'fact', 'though', 'water', 'less', 'public', 'put',
            'think', 'almost', 'hand', 'enough', 'far', 'took', 'head',
            'yet', 'government', 'system', 'better', 'set', 'told', 'nothing',
            'night', 'end', 'why', 'called', 'didn', 'eyes', 'find', 'going',
            'look', 'asked', 'later', 'knew', 'point', 'next', 'city', 'did',
            'want', 'way', 'could', 'people', 'may', 'says', 'each', 'those',
            'now', 'such', 'here', 'take', 'than', 'only', 'well', 'year'
        }
        
        rare_words = sum(1 for word in words if word not in common_words and len(word) > 4)
        return rare_words / len(words)
    
    def _calculate_formal_language_ratio(self, words):
        """Calculate ratio of formal/academic language"""
        if not words:
            return 0
        
        # Formal language indicators
        formal_indicators = {
            'therefore', 'however', 'furthermore', 'moreover', 'nevertheless',
            'consequently', 'subsequently', 'accordingly', 'thus', 'hence',
            'whereas', 'whereby', 'wherein', 'hereafter', 'heretofore',
            'notwithstanding', 'inasmuch', 'insofar', 'albeit', 'vis'
        }
        
        # Academic/formal suffixes
        formal_suffixes = {
            'tion', 'sion', 'ment', 'ance', 'ence', 'ity', 'ness', 'ism',
            'ize', 'ise', 'ate', 'fy', 'able', 'ible', 'ous', 'eous',
            'ious', 'ive', 'ary', 'ory', 'al', 'ic', 'ical'
        }
        
        formal_words = 0
        for word in words:
            if (word in formal_indicators or 
                (len(word) > 5 and any(word.endswith(suffix) for suffix in formal_suffixes))):
                formal_words += 1
        
        return formal_words / len(words)
    
    def get_feature_names(self):
        """Get names of extracted features"""
        feature_names = [
            'readability_avg_words_per_sentence',
            'readability_avg_syllables_per_word',
            'readability_avg_chars_per_word',
            'readability_flesch_reading_ease',
            'readability_flesch_kincaid_grade',
            'readability_automated_readability_index',
            'readability_complex_words_ratio',
            'readability_long_words_ratio',
            'readability_technical_terms_ratio',
            'readability_sentence_length_variance',
            'readability_punctuation_density',
            'readability_subordinate_clause_ratio',
            'readability_unique_word_ratio',
            'readability_rare_word_ratio',
            'readability_formal_language_ratio'
        ]
        
        return feature_names
    
    def analyze_text_readability(self, text):
        """Detailed readability analysis of a single text"""
        if not self.is_fitted_:
            raise ValueError("ReadabilityAnalyzer must be fitted before analysis")
        
        sentences = self._split_sentences(text)
        words = self._split_words(text)
        syllables = self._count_syllables_total(words)
        
        if len(sentences) == 0 or len(words) == 0:
            return {
                'error': 'Text too short for analysis',
                'text_length': len(text),
                'word_count': len(words),
                'sentence_count': len(sentences)
            }
        
        analysis = {
            'basic_stats': {
                'text_length': len(text),
                'word_count': len(words),
                'sentence_count': len(sentences),
                'syllable_count': syllables,
                'avg_words_per_sentence': len(words) / len(sentences),
                'avg_syllables_per_word': syllables / len(words),
                'avg_chars_per_word': sum(len(word) for word in words) / len(words)
            },
            'readability_scores': {
                'flesch_reading_ease': self._calculate_flesch_reading_ease(words, sentences, syllables),
                'flesch_kincaid_grade': self._calculate_flesch_kincaid_grade(words, sentences, syllables),
                'automated_readability_index': self._calculate_ari(words, sentences, text)
            },
            'complexity_metrics': {
                'complex_words_ratio': self._calculate_complex_words_ratio(words),
                'long_words_ratio': self._calculate_long_words_ratio(words),
                'technical_terms_ratio': self._calculate_technical_terms_ratio(words),
                'unique_word_ratio': self._calculate_unique_word_ratio(words),
                'rare_word_ratio': self._calculate_rare_word_ratio(words),
                'formal_language_ratio': self._calculate_formal_language_ratio(words)
            },
            'structure_analysis': {
                'sentence_length_variance': self._calculate_sentence_length_variance(sentences),
                'punctuation_density': self._calculate_punctuation_density(text),
                'subordinate_clause_ratio': self._calculate_subordinate_clause_ratio(text)
            }
        }
        
        # Interpret readability level
        flesch_score = analysis['readability_scores']['flesch_reading_ease']
        if flesch_score >= 90:
            readability_level = 'very_easy'
        elif flesch_score >= 80:
            readability_level = 'easy'
        elif flesch_score >= 70:
            readability_level = 'fairly_easy'
        elif flesch_score >= 60:
            readability_level = 'standard'
        elif flesch_score >= 50:
            readability_level = 'fairly_difficult'
        elif flesch_score >= 30:
            readability_level = 'difficult'
        else:
            readability_level = 'very_difficult'
        
        analysis['interpretation'] = {
            'readability_level': readability_level,
            'grade_level': analysis['readability_scores']['flesch_kincaid_grade'],
            'complexity_assessment': self._assess_complexity(analysis)
        }
        
        return analysis
    
    def _assess_complexity(self, analysis):
        """Assess overall complexity level"""
        complexity_indicators = [
            analysis['complexity_metrics']['complex_words_ratio'],
            analysis['complexity_metrics']['technical_terms_ratio'],
            analysis['complexity_metrics']['formal_language_ratio'],
            min(1.0, analysis['structure_analysis']['subordinate_clause_ratio'])  # Cap at 1.0
        ]
        
        avg_complexity = sum(complexity_indicators) / len(complexity_indicators)
        
        if avg_complexity > 0.3:
            return 'high'
        elif avg_complexity > 0.15:
            return 'medium'
        else:
            return 'low'