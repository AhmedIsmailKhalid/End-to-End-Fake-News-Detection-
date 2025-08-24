# features/sentiment_analyzer.py
# Sentiment Analysis Component for Enhanced Feature Engineering

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging
import re
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SentimentAnalyzer(BaseEstimator, TransformerMixin):
    """
    Advanced sentiment analysis for fake news detection.
    Focuses on emotional manipulation patterns common in misinformation.
    """
    
    def __init__(self):
        self.emotion_lexicons = self._load_emotion_lexicons()
        self.manipulation_patterns = self._load_manipulation_patterns()
        self.is_fitted_ = False
        
    def _load_emotion_lexicons(self):
        """Load emotion lexicons for sentiment analysis"""
        # Basic emotion lexicons (in production, these could be loaded from files)
        lexicons = {
            'positive': {
                'amazing', 'awesome', 'brilliant', 'excellent', 'fantastic', 'great', 
                'incredible', 'outstanding', 'perfect', 'wonderful', 'superb', 'magnificent',
                'love', 'adore', 'cherish', 'treasure', 'admire', 'appreciate',
                'happy', 'joyful', 'pleased', 'delighted', 'thrilled', 'ecstatic',
                'hope', 'optimistic', 'confident', 'assured', 'certain', 'positive'
            },
            'negative': {
                'awful', 'terrible', 'horrible', 'disgusting', 'appalling', 'shocking',
                'hate', 'despise', 'loathe', 'detest', 'abhor', 'resent',
                'angry', 'furious', 'outraged', 'livid', 'irate', 'enraged',
                'sad', 'depressed', 'miserable', 'devastated', 'heartbroken', 'grief',
                'fear', 'afraid', 'terrified', 'scared', 'anxious', 'worried',
                'disaster', 'catastrophe', 'crisis', 'emergency', 'danger', 'threat'
            },
            'anger': {
                'angry', 'furious', 'outraged', 'livid', 'irate', 'enraged', 'mad',
                'rage', 'fury', 'wrath', 'indignant', 'resentful', 'hostile',
                'attack', 'assault', 'violence', 'aggression', 'combat', 'fight'
            },
            'fear': {
                'fear', 'afraid', 'terrified', 'scared', 'anxious', 'worried', 'panic',
                'terror', 'horror', 'dread', 'nightmare', 'threat', 'danger',
                'risk', 'warning', 'alert', 'caution', 'alarm', 'emergency'
            },
            'trust': {
                'trust', 'believe', 'faith', 'confidence', 'reliable', 'honest',
                'truthful', 'sincere', 'genuine', 'authentic', 'credible', 'trustworthy'
            },
            'disgust': {
                'disgusting', 'revolting', 'repulsive', 'nauseating', 'sickening',
                'corrupt', 'contaminated', 'polluted', 'tainted', 'filthy', 'dirty'
            }
        }
        
        return lexicons
    
    def _load_manipulation_patterns(self):
        """Load patterns common in emotional manipulation"""
        patterns = {
            'urgency_words': {
                'urgent', 'immediate', 'emergency', 'crisis', 'breaking', 'alert',
                'now', 'quickly', 'hurry', 'rush', 'asap', 'immediately'
            },
            'authority_claims': {
                'experts', 'scientists', 'doctors', 'officials', 'authorities',
                'government', 'studies', 'research', 'proven', 'confirmed'
            },
            'conspiracy_words': {
                'conspiracy', 'cover-up', 'hidden', 'secret', 'expose', 'reveal',
                'truth', 'lies', 'deception', 'agenda', 'plot', 'scheme'
            },
            'absolute_terms': {
                'always', 'never', 'all', 'none', 'every', 'everyone', 'nobody',
                'everywhere', 'nowhere', 'completely', 'totally', 'absolutely'
            },
            'divisive_language': {
                'us', 'them', 'enemy', 'traitor', 'patriot', 'real', 'fake',
                'elite', 'establishment', 'mainstream', 'alternative'
            }
        }
        
        return patterns
    
    def fit(self, X, y=None):
        """Fit the sentiment analyzer (mainly for API consistency)"""
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Extract sentiment and emotional manipulation features"""
        if not self.is_fitted_:
            raise ValueError("SentimentAnalyzer must be fitted before transform")
        
        # Convert input to array if needed
        if isinstance(X, pd.Series):
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)
        
        features = []
        
        for text in X:
            text_features = self._extract_sentiment_features(str(text))
            features.append(text_features)
        
        return np.array(features)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def _extract_sentiment_features(self, text):
        """Extract comprehensive sentiment features from text"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        total_words = len(words)
        
        if total_words == 0:
            return [0.0] * 25  # Return zeros for empty text
        
        features = []
        
        # Basic sentiment scores
        for emotion, lexicon in self.emotion_lexicons.items():
            emotion_count = sum(1 for word in words if word in lexicon)
            emotion_ratio = emotion_count / total_words
            features.append(emotion_ratio)
        
        # Manipulation pattern features
        for pattern_type, pattern_words in self.manipulation_patterns.items():
            pattern_count = sum(1 for word in words if word in pattern_words)
            pattern_ratio = pattern_count / total_words
            features.append(pattern_ratio)
        
        # Advanced sentiment features
        features.extend(self._extract_advanced_sentiment_features(text, words))
        
        return features
    
    def _extract_advanced_sentiment_features(self, text, words):
        """Extract advanced sentiment and emotional manipulation features"""
        features = []
        
        # Exclamation and question mark patterns
        exclamation_count = text.count('!')
        question_count = text.count('?')
        features.append(exclamation_count / len(text) if len(text) > 0 else 0)
        features.append(question_count / len(text) if len(text) > 0 else 0)
        
        # Capitalization patterns (potential shouting)
        caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        features.append(caps_words / len(words) if len(words) > 0 else 0)
        
        # Emotional intensity (multiple exclamation/question marks)
        intense_exclamation = len(re.findall(r'!{2,}', text))
        intense_question = len(re.findall(r'\?{2,}', text))
        features.append(intense_exclamation / len(text) if len(text) > 0 else 0)
        features.append(intense_question / len(text) if len(text) > 0 else 0)
        
        # Emotional contrast (mixing positive and negative)
        positive_count = sum(1 for word in words if word in self.emotion_lexicons['positive'])
        negative_count = sum(1 for word in words if word in self.emotion_lexicons['negative'])
        
        if positive_count + negative_count > 0:
            emotional_contrast = min(positive_count, negative_count) / (positive_count + negative_count)
        else:
            emotional_contrast = 0
        features.append(emotional_contrast)
        
        # Authority vs conspiracy balance
        authority_count = sum(1 for word in words if word in self.manipulation_patterns['authority_claims'])
        conspiracy_count = sum(1 for word in words if word in self.manipulation_patterns['conspiracy_words'])
        
        total_claims = authority_count + conspiracy_count
        if total_claims > 0:
            authority_ratio = authority_count / total_claims
        else:
            authority_ratio = 0.5  # Neutral when no claims
        features.append(authority_ratio)
        
        # Urgency density
        urgency_count = sum(1 for word in words if word in self.manipulation_patterns['urgency_words'])
        features.append(urgency_count / len(words) if len(words) > 0 else 0)
        
        # Personal pronouns (engagement tactics)
        personal_pronouns = {'you', 'your', 'yours', 'we', 'us', 'our', 'ours'}
        pronoun_count = sum(1 for word in words if word in personal_pronouns)
        features.append(pronoun_count / len(words) if len(words) > 0 else 0)
        
        return features
    
    def get_feature_names(self):
        """Get names of extracted features"""
        feature_names = []
        
        # Basic emotion features
        for emotion in self.emotion_lexicons.keys():
            feature_names.append(f'sentiment_{emotion}_ratio')
        
        # Manipulation pattern features
        for pattern in self.manipulation_patterns.keys():
            feature_names.append(f'sentiment_{pattern}_ratio')
        
        # Advanced features
        advanced_features = [
            'sentiment_exclamation_density',
            'sentiment_question_density',
            'sentiment_caps_words_ratio',
            'sentiment_intense_exclamation_density',
            'sentiment_intense_question_density',
            'sentiment_emotional_contrast',
            'sentiment_authority_ratio',
            'sentiment_urgency_density',
            'sentiment_personal_pronouns_ratio'
        ]
        
        feature_names.extend(advanced_features)
        
        return feature_names
    
    def analyze_text_sentiment(self, text):
        """Detailed sentiment analysis of a single text"""
        if not self.is_fitted_:
            raise ValueError("SentimentAnalyzer must be fitted before analysis")
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        analysis = {
            'text_length': len(text),
            'word_count': len(words),
            'emotions': {},
            'manipulation_patterns': {},
            'overall_sentiment': 'neutral',
            'manipulation_score': 0.0,
            'emotional_intensity': 0.0
        }
        
        total_words = len(words)
        if total_words == 0:
            return analysis
        
        # Analyze emotions
        for emotion, lexicon in self.emotion_lexicons.items():
            emotion_count = sum(1 for word in words if word in lexicon)
            analysis['emotions'][emotion] = {
                'count': emotion_count,
                'ratio': emotion_count / total_words,
                'words_found': [word for word in words if word in lexicon][:5]  # Top 5 matches
            }
        
        # Analyze manipulation patterns
        for pattern_type, pattern_words in self.manipulation_patterns.items():
            pattern_count = sum(1 for word in words if word in pattern_words)
            analysis['manipulation_patterns'][pattern_type] = {
                'count': pattern_count,
                'ratio': pattern_count / total_words,
                'words_found': [word for word in words if word in pattern_words][:3]  # Top 3 matches
            }
        
        # Calculate overall sentiment
        positive_score = analysis['emotions']['positive']['ratio']
        negative_score = analysis['emotions']['negative']['ratio']
        
        if positive_score > negative_score + 0.02:
            analysis['overall_sentiment'] = 'positive'
        elif negative_score > positive_score + 0.02:
            analysis['overall_sentiment'] = 'negative'
        else:
            analysis['overall_sentiment'] = 'neutral'
        
        # Calculate manipulation score
        manipulation_indicators = [
            analysis['manipulation_patterns']['urgency_words']['ratio'],
            analysis['manipulation_patterns']['conspiracy_words']['ratio'],
            analysis['manipulation_patterns']['absolute_terms']['ratio'],
            analysis['manipulation_patterns']['divisive_language']['ratio']
        ]
        analysis['manipulation_score'] = sum(manipulation_indicators) / len(manipulation_indicators)
        
        # Calculate emotional intensity
        fear_anger_score = (analysis['emotions']['fear']['ratio'] + 
                           analysis['emotions']['anger']['ratio'])
        exclamation_density = text.count('!') / len(text) if len(text) > 0 else 0
        caps_density = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        
        analysis['emotional_intensity'] = (fear_anger_score + exclamation_density + caps_density) / 3
        
        return analysis
    
    def get_manipulation_indicators(self, text):
        """Get specific manipulation indicators for fact-checking"""
        analysis = self.analyze_text_sentiment(text)
        
        indicators = {
            'high_emotional_intensity': analysis['emotional_intensity'] > 0.1,
            'urgency_manipulation': analysis['manipulation_patterns']['urgency_words']['ratio'] > 0.02,
            'conspiracy_language': analysis['manipulation_patterns']['conspiracy_words']['ratio'] > 0.01,
            'absolute_claims': analysis['manipulation_patterns']['absolute_terms']['ratio'] > 0.03,
            'divisive_framing': analysis['manipulation_patterns']['divisive_language']['ratio'] > 0.02,
            'emotional_overload': (analysis['emotions']['fear']['ratio'] + 
                                 analysis['emotions']['anger']['ratio']) > 0.05
        }
        
        # Overall manipulation risk
        risk_score = sum(indicators.values()) / len(indicators)
        indicators['overall_manipulation_risk'] = 'high' if risk_score > 0.5 else 'medium' if risk_score > 0.3 else 'low'
        
        return indicators