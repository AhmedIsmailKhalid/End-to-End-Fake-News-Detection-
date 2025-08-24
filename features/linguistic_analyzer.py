# File: features/linguistic_analyzer.py
# Advanced Linguistic Analysis Component for Enhanced Feature Engineering

import numpy as np
import pandas as pd
import re
import logging
from typing import List, Dict, Any, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class LinguisticAnalyzer(BaseEstimator, TransformerMixin):
    """
    Advanced linguistic analysis for fake news detection.
    Analyzes syntactic patterns, discourse markers, and linguistic anomalies.
    """
    
    def __init__(self):
        self.discourse_markers = self._load_discourse_markers()
        self.linguistic_patterns = self._load_linguistic_patterns()
        self.pos_patterns = self._load_pos_patterns()
        self.is_fitted_ = False
        
    def _load_discourse_markers(self):
        """Load discourse markers for coherence analysis"""
        markers = {
            'addition': {'also', 'furthermore', 'moreover', 'additionally', 'besides', 'plus', 'and'},
            'contrast': {'however', 'but', 'nevertheless', 'nonetheless', 'yet', 'still', 'although', 'though'},
            'cause_effect': {'therefore', 'thus', 'consequently', 'as a result', 'because', 'since', 'so'},
            'temporal': {'then', 'next', 'afterwards', 'meanwhile', 'subsequently', 'finally', 'first', 'second'},
            'emphasis': {'indeed', 'certainly', 'obviously', 'clearly', 'definitely', 'absolutely', 'surely'},
            'concession': {'admittedly', 'granted', 'to be sure', 'of course', 'naturally', 'undoubtedly'},
            'exemplification': {'for example', 'for instance', 'such as', 'namely', 'specifically', 'particularly'},
            'summary': {'in conclusion', 'to summarize', 'in summary', 'overall', 'in general', 'basically'}
        }
        return markers
    
    def _load_linguistic_patterns(self):
        """Load patterns for linguistic analysis"""
        patterns = {
            'modal_verbs': {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'},
            'hedge_words': {'probably', 'possibly', 'perhaps', 'maybe', 'likely', 'apparently', 'seemingly', 'supposedly'},
            'boosters': {'very', 'extremely', 'highly', 'completely', 'totally', 'absolutely', 'definitely', 'certainly'},
            'negation': {'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor'},
            'intensifiers': {'so', 'such', 'quite', 'rather', 'pretty', 'fairly', 'really', 'truly', 'deeply'},
            'questioning': {'why', 'how', 'what', 'when', 'where', 'who', 'which', 'whose'},
            'personal_pronouns': {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'},
            'demonstratives': {'this', 'that', 'these', 'those', 'here', 'there'},
            'quantifiers': {'all', 'every', 'each', 'some', 'any', 'many', 'few', 'several', 'most', 'much'}
        }
        return patterns
    
    def _load_pos_patterns(self):
        """Load part-of-speech patterns (simplified without NLTK)"""
        # Simple heuristics for POS detection
        patterns = {
            'verb_endings': {'ed', 'ing', 'en', 's', 'es'},
            'noun_endings': {'tion', 'sion', 'ment', 'ness', 'ity', 'er', 'or', 'ist', 'ism'},
            'adjective_endings': {'able', 'ible', 'ful', 'less', 'ous', 'eous', 'ious', 'ive', 'ic', 'al'},
            'adverb_endings': {'ly', 'ward', 'wise'}
        }
        return patterns
    
    def fit(self, X, y=None):
        """Fit the linguistic analyzer"""
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Extract linguistic features"""
        if not self.is_fitted_:
            raise ValueError("LinguisticAnalyzer must be fitted before transform")
        
        # Convert input to array if needed
        if isinstance(X, pd.Series):
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)
        
        features = []
        
        for text in X:
            text_features = self._extract_linguistic_features(str(text))
            features.append(text_features)
        
        return np.array(features)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def _extract_linguistic_features(self, text):
        """Extract comprehensive linguistic features"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) == 0:
            return [0.0] * 25  # Return zeros for empty text
        
        features = []
        
        # Discourse marker features
        discourse_features = self._extract_discourse_features(text_lower, words)
        features.extend(discourse_features)
        
        # Linguistic pattern features
        pattern_features = self._extract_pattern_features(text_lower, words)
        features.extend(pattern_features)
        
        # Syntactic complexity features
        syntax_features = self._extract_syntax_features(text, sentences, words)
        features.extend(syntax_features)
        
        # Coherence and flow features
        coherence_features = self._extract_coherence_features(text, sentences)
        features.extend(coherence_features)
        
        return features
    
    def _extract_discourse_features(self, text_lower, words):
        """Extract discourse marker features"""
        features = []
        total_words = len(words)
        
        # Count discourse markers by category
        for marker_type, markers in self.discourse_markers.items():
            marker_count = 0
            
            # Single word markers
            marker_count += sum(1 for word in words if word in markers)
            
            # Multi-word markers
            for marker in markers:
                if ' ' in marker:
                    marker_count += text_lower.count(marker)
            
            marker_ratio = marker_count / total_words if total_words > 0 else 0
            features.append(marker_ratio)
        
        return features
    
    def _extract_pattern_features(self, text_lower, words):
        """Extract linguistic pattern features"""
        features = []
        total_words = len(words)
        
        # Count linguistic patterns
        for pattern_type, pattern_words in self.linguistic_patterns.items():
            pattern_count = sum(1 for word in words if word in pattern_words)
            pattern_ratio = pattern_count / total_words if total_words > 0 else 0
            features.append(pattern_ratio)
        
        return features
    
    def _extract_syntax_features(self, text, sentences, words):
        """Extract syntactic complexity features"""
        features = []
        
        # Average sentence length
        if sentences:
            avg_sentence_length = len(words) / len(sentences)
        else:
            avg_sentence_length = 0
        features.append(avg_sentence_length)
        
        # Sentence length variance
        if len(sentences) > 1:
            sentence_lengths = [len(sentence.split()) for sentence in sentences]
            mean_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((length - mean_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
        else:
            variance = 0
        features.append(variance)
        
        # Complex sentence indicators
        complex_indicators = self._count_complex_sentence_indicators(text)
        features.extend(complex_indicators)
        
        return features
    
    def _count_complex_sentence_indicators(self, text):
        """Count indicators of complex sentence structure"""
        indicators = []
        
        # Subordinating conjunctions
        subordinating = ['although', 'because', 'since', 'while', 'whereas', 'if', 'unless', 'when', 'where']
        sub_count = sum(text.lower().count(f' {conj} ') for conj in subordinating)
        indicators.append(sub_count / len(text) * 1000 if text else 0)
        
        # Relative pronouns
        relative_pronouns = ['that', 'which', 'who', 'whom', 'whose', 'where', 'when']
        rel_count = sum(text.lower().count(f' {pron} ') for pron in relative_pronouns)
        indicators.append(rel_count / len(text) * 1000 if text else 0)
        
        # Passive voice indicators (simplified)
        passive_indicators = ['was', 'were', 'been', 'being']
        passive_count = sum(text.lower().count(f' {ind} ') for ind in passive_indicators)
        indicators.append(passive_count / len(text) * 1000 if text else 0)
        
        return indicators
    
    def _extract_coherence_features(self, text, sentences):
        """Extract text coherence and flow features"""
        features = []
        
        # Paragraph structure (approximate)
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Average paragraph length
        if paragraphs:
            avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
        else:
            avg_paragraph_length = 0
        features.append(avg_paragraph_length)
        
        # Topic coherence (simplified using word repetition)
        coherence_score = self._calculate_lexical_coherence(sentences)
        features.append(coherence_score)
        
        # Transition density
        transition_density = self._calculate_transition_density(text)
        features.append(transition_density)
        
        return features
    
    def _calculate_lexical_coherence(self, sentences):
        """Calculate lexical coherence between sentences"""
        if len(sentences) < 2:
            return 0
        
        coherence_scores = []
        
        for i in range(len(sentences) - 1):
            words1 = set(re.findall(r'\b\w+\b', sentences[i].lower()))
            words2 = set(re.findall(r'\b\w+\b', sentences[i + 1].lower()))
            
            # Remove very common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words1 = words1 - common_words
            words2 = words2 - common_words
            
            if words1 and words2:
                overlap = len(words1.intersection(words2))
                union = len(words1.union(words2))
                coherence = overlap / union if union > 0 else 0
                coherence_scores.append(coherence)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
    
    def _calculate_transition_density(self, text):
        """Calculate density of transition words"""
        transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'nevertheless', 'nonetheless', 'meanwhile', 'additionally', 'similarly',
            'likewise', 'in contrast', 'on the other hand', 'for example', 'for instance'
        }
        
        text_lower = text.lower()
        transition_count = 0
        
        for transition in transition_words:
            if ' ' in transition:
                transition_count += text_lower.count(transition)
            else:
                transition_count += len(re.findall(rf'\b{transition}\b', text_lower))
        
        return transition_count / len(text) * 1000 if text else 0
    
    def get_feature_names(self):
        """Get names of extracted features"""
        feature_names = []
        
        # Discourse marker features
        for marker_type in self.discourse_markers.keys():
            feature_names.append(f'linguistic_{marker_type}_markers_ratio')
        
        # Linguistic pattern features
        for pattern_type in self.linguistic_patterns.keys():
            feature_names.append(f'linguistic_{pattern_type}_ratio')
        
        # Syntax features
        syntax_features = [
            'linguistic_avg_sentence_length',
            'linguistic_sentence_length_variance',
            'linguistic_subordinating_density',
            'linguistic_relative_pronouns_density',
            'linguistic_passive_voice_density'
        ]
        feature_names.extend(syntax_features)
        
        # Coherence features
        coherence_features = [
            'linguistic_avg_paragraph_length',
            'linguistic_lexical_coherence',
            'linguistic_transition_density'
        ]
        feature_names.extend(coherence_features)
        
        return feature_names
    
    def analyze_text_linguistics(self, text):
        """Detailed linguistic analysis of a single text"""
        if not self.is_fitted_:
            raise ValueError("LinguisticAnalyzer must be fitted before analysis")
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        analysis = {
            'basic_stats': {
                'text_length': len(text),
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0
            },
            'discourse_markers': {},
            'linguistic_patterns': {},
            'syntactic_complexity': {},
            'coherence_analysis': {}
        }
        
        # Analyze discourse markers
        for marker_type, markers in self.discourse_markers.items():
            found_markers = []
            for word in words:
                if word in markers:
                    found_markers.append(word)
            
            # Check multi-word markers
            for marker in markers:
                if ' ' in marker and marker in text_lower:
                    found_markers.extend([marker] * text_lower.count(marker))
            
            analysis['discourse_markers'][marker_type] = {
                'count': len(found_markers),
                'ratio': len(found_markers) / len(words) if words else 0,
                'markers_found': list(set(found_markers))[:5]  # Top 5 unique markers
            }
        
        # Analyze linguistic patterns
        for pattern_type, pattern_words in self.linguistic_patterns.items():
            found_patterns = [word for word in words if word in pattern_words]
            analysis['linguistic_patterns'][pattern_type] = {
                'count': len(found_patterns),
                'ratio': len(found_patterns) / len(words) if words else 0,
                'patterns_found': list(set(found_patterns))[:5]
            }
        
        # Analyze syntactic complexity
        complex_indicators = self._count_complex_sentence_indicators(text)
        analysis['syntactic_complexity'] = {
            'subordinating_conjunctions_density': complex_indicators[0],
            'relative_pronouns_density': complex_indicators[1],
            'passive_voice_density': complex_indicators[2],
            'sentence_length_variance': self._extract_syntax_features(text, sentences, words)[1],
            'complexity_score': sum(complex_indicators) / len(complex_indicators)
        }
        
        # Analyze coherence
        analysis['coherence_analysis'] = {
            'lexical_coherence': self._calculate_lexical_coherence(sentences),
            'transition_density': self._calculate_transition_density(text),
            'paragraph_structure': len(text.split('\n\n')),
            'overall_coherence_score': (self._calculate_lexical_coherence(sentences) + 
                                      min(1.0, self._calculate_transition_density(text) / 10)) / 2
        }
        
        # Overall assessment
        analysis['overall_assessment'] = {
            'linguistic_sophistication': self._assess_sophistication(analysis),
            'discourse_quality': self._assess_discourse_quality(analysis),
            'potential_anomalies': self._detect_linguistic_anomalies(analysis)
        }
        
        return analysis
    
    def _assess_sophistication(self, analysis):
        """Assess overall linguistic sophistication"""
        sophistication_score = 0
        
        # Discourse marker variety
        marker_variety = len([mt for mt, data in analysis['discourse_markers'].items() if data['count'] > 0])
        sophistication_score += marker_variety / len(self.discourse_markers) * 0.3
        
        # Complex syntax usage
        syntax_score = analysis['syntactic_complexity']['complexity_score']
        sophistication_score += min(syntax_score, 0.02) / 0.02 * 0.3  # Cap and normalize
        
        # Coherence quality
        coherence_score = analysis['coherence_analysis']['overall_coherence_score']
        sophistication_score += coherence_score * 0.4
        
        if sophistication_score > 0.7:
            return 'high'
        elif sophistication_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _assess_discourse_quality(self, analysis):
        """Assess discourse quality and organization"""
        quality_indicators = []
        
        # Balanced use of discourse markers
        marker_counts = [data['count'] for data in analysis['discourse_markers'].values()]
        if marker_counts:
            marker_balance = 1 - (max(marker_counts) - min(marker_counts)) / (sum(marker_counts) + 1)
            quality_indicators.append(marker_balance)
        
        # Coherence score
        quality_indicators.append(analysis['coherence_analysis']['overall_coherence_score'])
        
        # Transition usage
        transition_score = min(1.0, analysis['coherence_analysis']['transition_density'] / 5)
        quality_indicators.append(transition_score)
        
        avg_quality = sum(quality_indicators) / len(quality_indicators) if quality_indicators else 0
        
        if avg_quality > 0.7:
            return 'excellent'
        elif avg_quality > 0.5:
            return 'good'
        elif avg_quality > 0.3:
            return 'fair'
        else:
            return 'poor'
    
    def _detect_linguistic_anomalies(self, analysis):
        """Detect potential linguistic anomalies that might indicate manipulation"""
        anomalies = []
        
        # Excessive use of boosters/intensifiers
        booster_ratio = analysis['linguistic_patterns']['boosters']['ratio']
        if booster_ratio > 0.05:  # More than 5% boosters
            anomalies.append({
                'type': 'excessive_boosters',
                'severity': 'medium',
                'description': f'High use of intensifying language ({booster_ratio:.1%})',
                'examples': analysis['linguistic_patterns']['boosters']['patterns_found']
            })
        
        # Unusual negation patterns
        negation_ratio = analysis['linguistic_patterns']['negation']['ratio']
        if negation_ratio > 0.08:  # More than 8% negation
            anomalies.append({
                'type': 'excessive_negation',
                'severity': 'low',
                'description': f'High use of negative language ({negation_ratio:.1%})',
                'examples': analysis['linguistic_patterns']['negation']['patterns_found']
            })
        
        # Low coherence with high complexity (potential obfuscation)
        coherence = analysis['coherence_analysis']['overall_coherence_score']
        complexity = analysis['syntactic_complexity']['complexity_score']
        if complexity > 0.01 and coherence < 0.3:
            anomalies.append({
                'type': 'complexity_without_coherence',
                'severity': 'high',
                'description': 'Complex language structure with poor coherence (potential obfuscation)',
                'coherence_score': coherence,
                'complexity_score': complexity
            })
        
        # Unusual question density
        question_ratio = analysis['linguistic_patterns']['questioning']['ratio']
        if question_ratio > 0.06:  # More than 6% question words
            anomalies.append({
                'type': 'excessive_questioning',
                'severity': 'medium',
                'description': f'High density of questioning language ({question_ratio:.1%})',
                'examples': analysis['linguistic_patterns']['questioning']['patterns_found']
            })
        
        return anomalies
    
    def get_manipulation_indicators(self, text):
        """Get specific linguistic manipulation indicators"""
        analysis = self.analyze_text_linguistics(text)
        
        indicators = {
            'linguistic_manipulation_score': 0.0,
            'specific_indicators': [],
            'overall_risk': 'low'
        }
        
        # Check for manipulation patterns
        manipulation_score = 0
        
        # Excessive emphasis/boosters
        if analysis['linguistic_patterns']['boosters']['ratio'] > 0.05:
            manipulation_score += 0.3
            indicators['specific_indicators'].append('excessive_emphasis')
        
        # Lack of hedging (overconfidence)
        if analysis['linguistic_patterns']['hedge_words']['ratio'] < 0.01:
            manipulation_score += 0.2
            indicators['specific_indicators'].append('overconfident_language')
        
        # Poor coherence (confusion tactics)
        if analysis['coherence_analysis']['overall_coherence_score'] < 0.3:
            manipulation_score += 0.4
            indicators['specific_indicators'].append('poor_coherence')
        
        # Excessive questioning (doubt seeding)
        if analysis['linguistic_patterns']['questioning']['ratio'] > 0.06:
            manipulation_score += 0.3
            indicators['specific_indicators'].append('excessive_questioning')
        
        # High personal pronoun usage (false intimacy)
        if analysis['linguistic_patterns']['personal_pronouns']['ratio'] > 0.15:
            manipulation_score += 0.2
            indicators['specific_indicators'].append('false_intimacy')
        
        indicators['linguistic_manipulation_score'] = min(1.0, manipulation_score)
        
        # Overall risk assessment
        if manipulation_score > 0.7:
            indicators['overall_risk'] = 'high'
        elif manipulation_score > 0.4:
            indicators['overall_risk'] = 'medium'
        else:
            indicators['overall_risk'] = 'low'
        
        return indicators
