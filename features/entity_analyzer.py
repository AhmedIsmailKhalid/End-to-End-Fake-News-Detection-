# features/entity_analyzer.py
# Named Entity Recognition and Analysis Component

import numpy as np
import pandas as pd
import re
import logging
from typing import List, Dict, Any, Set, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EntityAnalyzer(BaseEstimator, TransformerMixin):
    """
    Named Entity Recognition and Analysis for fake news detection.
    Identifies entities and patterns that may indicate misinformation.
    """
    
    def __init__(self):
        self.known_entities = self._load_entity_knowledge()
        self.entity_patterns = self._load_entity_patterns()
        self.is_fitted_ = False
        
    def _load_entity_knowledge(self):
        """Load knowledge bases for entity recognition"""
        # In production, these would be loaded from comprehensive databases
        entities = {
            'countries': {
                'united states', 'usa', 'america', 'china', 'russia', 'germany', 
                'france', 'italy', 'spain', 'uk', 'united kingdom', 'britain',
                'canada', 'australia', 'japan', 'india', 'brazil', 'mexico',
                'south korea', 'north korea', 'iran', 'iraq', 'afghanistan',
                'ukraine', 'poland', 'netherlands', 'belgium', 'sweden'
            },
            'cities': {
                'new york', 'los angeles', 'chicago', 'houston', 'philadelphia',
                'phoenix', 'san antonio', 'san diego', 'dallas', 'san jose',
                'london', 'paris', 'berlin', 'rome', 'madrid', 'moscow',
                'beijing', 'tokyo', 'seoul', 'mumbai', 'delhi', 'bangkok'
            },
            'organizations': {
                'fbi', 'cia', 'nsa', 'pentagon', 'nato', 'un', 'who', 'cdc',
                'fda', 'nasa', 'google', 'facebook', 'twitter', 'amazon',
                'microsoft', 'apple', 'tesla', 'spacex', 'walmart', 'mcdonalds'
            },
            'government_roles': {
                'president', 'prime minister', 'senator', 'congressman', 'governor',
                'mayor', 'ambassador', 'secretary', 'minister', 'chancellor',
                'director', 'chief', 'general', 'admiral', 'colonel'
            },
            'media_outlets': {
                'cnn', 'fox news', 'bbc', 'reuters', 'associated press', 'ap',
                'new york times', 'washington post', 'wall street journal',
                'guardian', 'times', 'npr', 'pbs', 'msnbc', 'abc', 'cbs', 'nbc'
            },
            'scientific_terms': {
                'research', 'study', 'experiment', 'clinical trial', 'peer review',
                'scientist', 'professor', 'university', 'laboratory', 'data',
                'evidence', 'hypothesis', 'theory', 'analysis', 'publication'
            }
        }
        return entities
    
    def _load_entity_patterns(self):
        """Load patterns for entity recognition"""
        patterns = {
            'person_titles': r'\b(?:Dr|Professor|Mr|Mrs|Ms|Miss|Sir|Lord|Lady|Hon)\.\s+[A-Z][a-z]+',
            'dates': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'money': r'\$[\d,]+(?:\.\d{2})?|\b\d+\s*(?:dollars?|euros?|pounds?|billion|million|thousand)',
            'percentages': r'\b\d+(?:\.\d+)?%',
            'phone_numbers': r'\b\d{3}[.-]?\d{3}[.-]?\d{4}',
            'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
            'urls': r'https?://[^\s]+|www\.[^\s]+',
            'coordinates': r'\b\d+°\d+\'[NS]\s+\d+°\d+\'[EW]',
            'times': r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?',
            'zip_codes': r'\b\d{5}(?:-\d{4})?\b',
            'social_security': r'\b\d{3}-\d{2}-\d{4}\b',
            'ip_addresses': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        return patterns
    
    def fit(self, X, y=None):
        """Fit the entity analyzer"""
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Extract entity-based features"""
        if not self.is_fitted_:
            raise ValueError("EntityAnalyzer must be fitted before transform")
        
        # Convert input to array if needed
        if isinstance(X, pd.Series):
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)
        
        features = []
        
        for text in X:
            text_features = self._extract_entity_features(str(text))
            features.append(text_features)
        
        return np.array(features)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def _extract_entity_features(self, text):
        """Extract comprehensive entity-based features"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        total_words = len(words)
        
        if total_words == 0:
            return [0.0] * 20  # Return zeros for empty text
        
        features = []
        
        # Entity type frequencies
        for entity_type, entity_set in self.known_entities.items():
            entity_count = sum(1 for word in words if word in entity_set)
            # Also check for multi-word entities
            for entity in entity_set:
                if ' ' in entity and entity in text_lower:
                    entity_count += text_lower.count(entity)
            
            entity_ratio = entity_count / total_words
            features.append(entity_ratio)
        
        # Pattern-based entity features
        pattern_features = self._extract_pattern_features(text)
        features.extend(pattern_features)
        
        # Entity diversity and density
        entity_diversity = self._calculate_entity_diversity(text_lower, words)
        entity_density = self._calculate_entity_density(text_lower, words)
        features.extend([entity_diversity, entity_density])
        
        # Authority and credibility indicators
        authority_score = self._calculate_authority_score(text_lower, words)
        credibility_score = self._calculate_credibility_score(text_lower)
        features.extend([authority_score, credibility_score])
        
        # Fact-checking potential
        fact_check_indicators = self._calculate_fact_check_indicators(text_lower)
        features.extend(fact_check_indicators)
        
        return features
    
    def _extract_pattern_features(self, text):
        """Extract features based on regex patterns"""
        features = []
        
        # Count occurrences of each pattern type
        for pattern_name, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            match_count = len(matches)
            
            # Normalize by text length
            if len(text) > 0:
                match_density = match_count / len(text) * 1000  # Per 1000 characters
            else:
                match_density = 0
            
            features.append(match_density)
        
        return features
    
    def _calculate_entity_diversity(self, text_lower, words):
        """Calculate diversity of entity types mentioned"""
        entity_types_found = set()
        
        for entity_type, entity_set in self.known_entities.items():
            for entity in entity_set:
                if entity in text_lower or any(word in entity_set for word in words):
                    entity_types_found.add(entity_type)
                    break
        
        # Diversity score: number of different entity types / total possible types
        diversity_score = len(entity_types_found) / len(self.known_entities)
        return diversity_score
    
    def _calculate_entity_density(self, text_lower, words):
        """Calculate overall density of named entities"""
        total_entities = 0
        
        # Count individual word entities
        for entity_set in self.known_entities.values():
            total_entities += sum(1 for word in words if word in entity_set)
        
        # Count multi-word entities
        for entity_set in self.known_entities.values():
            for entity in entity_set:
                if ' ' in entity:
                    total_entities += text_lower.count(entity)
        
        # Density: entities per 100 words
        if len(words) > 0:
            density = (total_entities / len(words)) * 100
        else:
            density = 0
        
        return min(density, 50)  # Cap at 50 to avoid outliers
    
    def _calculate_authority_score(self, text_lower, words):
        """Calculate score based on authoritative sources and references"""
        authority_indicators = {
            'academic': {'university', 'research', 'study', 'professor', 'phd', 'journal', 'peer review'},
            'government': {'government', 'official', 'department', 'agency', 'bureau', 'federal', 'state'},
            'media': {'news', 'reporter', 'journalist', 'newspaper', 'broadcast', 'interview'},
            'medical': {'doctor', 'hospital', 'medical', 'clinical', 'patient', 'treatment', 'diagnosis'},
            'legal': {'court', 'judge', 'lawyer', 'attorney', 'legal', 'law', 'constitution'},
            'expert': {'expert', 'specialist', 'authority', 'professional', 'certified', 'licensed'}
        }
        
        authority_score = 0
        for category, indicators in authority_indicators.items():
            category_score = sum(1 for word in words if word in indicators)
            authority_score += category_score
        
        # Normalize by text length
        if len(words) > 0:
            authority_score = (authority_score / len(words)) * 100
        
        return min(authority_score, 20)  # Cap at 20
    
    def _calculate_credibility_score(self, text_lower):
        """Calculate credibility score based on verifiable information patterns"""
        credibility_indicators = {
            'specific_dates': len(re.findall(self.entity_patterns['dates'], text_lower)),
            'specific_numbers': len(re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text_lower)),
            'citations': text_lower.count('according to') + text_lower.count('reported by') + text_lower.count('source'),
            'quotes': text_lower.count('"') // 2,  # Paired quotes
            'named_sources': len(re.findall(r'[A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Z][a-z]+)*', text_lower))
        }
        
        # Weight different indicators
        weights = {
            'specific_dates': 2,
            'specific_numbers': 1,
            'citations': 3,
            'quotes': 2,
            'named_sources': 3
        }
        
        weighted_score = sum(credibility_indicators[indicator] * weights[indicator] 
                           for indicator in credibility_indicators)
        
        # Normalize by text length
        if len(text_lower) > 0:
            credibility_score = (weighted_score / len(text_lower)) * 1000
        else:
            credibility_score = 0
        
        return min(credibility_score, 10)  # Cap at 10
    
    def _calculate_fact_check_indicators(self, text_lower):
        """Calculate indicators that suggest fact-checkable claims"""
        indicators = []
        
        # Statistical claims
        stat_patterns = [
            r'\b\d+(?:\.\d+)?%',  # Percentages
            r'\b\d+(?:,\d{3})*\s+(?:people|americans|citizens|voters|patients)',  # Population claims
            r'\b\d+(?:\.\d+)?\s*(?:times|fold)\s+(?:more|less|higher|lower)',  # Comparative claims
            r'\b\d+(?:\.\d+)?\s*(?:billion|million|thousand)',  # Large numbers
        ]
        
        statistical_claims = sum(len(re.findall(pattern, text_lower)) for pattern in stat_patterns)
        indicators.append(min(statistical_claims / max(1, len(text_lower)) * 1000, 5))
        
        # Causal claims
        causal_words = ['causes', 'leads to', 'results in', 'due to', 'because of', 'linked to']
        causal_claims = sum(text_lower.count(word) for word in causal_words)
        indicators.append(min(causal_claims / max(1, len(text_lower.split())) * 100, 3))
        
        # Temporal claims
        temporal_words = ['since', 'after', 'before', 'during', 'within', 'by']
        temporal_claims = sum(text_lower.count(word) for word in temporal_words)
        indicators.append(min(temporal_claims / max(1, len(text_lower.split())) * 100, 3))
        
        return indicators
    
    def get_feature_names(self):
        """Get names of extracted features"""
        feature_names = []
        
        # Entity type features
        for entity_type in self.known_entities.keys():
            feature_names.append(f'entity_{entity_type}_ratio')
        
        # Pattern-based features
        for pattern_name in self.entity_patterns.keys():
            feature_names.append(f'entity_{pattern_name}_density')
        
        # Additional features
        additional_features = [
            'entity_diversity_score',
            'entity_density_score',
            'entity_authority_score',
            'entity_credibility_score',
            'entity_statistical_claims',
            'entity_causal_claims',
            'entity_temporal_claims'
        ]
        
        feature_names.extend(additional_features)
        
        return feature_names
    
    def analyze_text_entities(self, text):
        """Detailed entity analysis of a single text"""
        if not self.is_fitted_:
            raise ValueError("EntityAnalyzer must be fitted before analysis")
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        analysis = {
            'text_length': len(text),
            'word_count': len(words),
            'entities_found': defaultdict(list),
            'patterns_found': {},
            'authority_assessment': {},
            'credibility_assessment': {},
            'fact_check_potential': {}
        }
        
        # Find specific entities
        for entity_type, entity_set in self.known_entities.items():
            found_entities = []
            
            # Single word entities
            for word in words:
                if word in entity_set:
                    found_entities.append(word)
            
            # Multi-word entities
            for entity in entity_set:
                if ' ' in entity and entity in text_lower:
                    found_entities.extend([entity] * text_lower.count(entity))
            
            analysis['entities_found'][entity_type] = list(set(found_entities))
        
        # Find pattern matches
        for pattern_name, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            analysis['patterns_found'][pattern_name] = matches
        
        # Authority assessment
        analysis['authority_assessment'] = {
            'authority_score': self._calculate_authority_score(text_lower, words),
            'has_academic_references': any(word in text_lower for word in ['university', 'research', 'study']),
            'has_government_references': any(word in text_lower for word in ['government', 'official', 'federal']),
            'has_media_references': any(word in text_lower for word in ['news', 'reporter', 'journalist']),
            'has_expert_references': any(word in text_lower for word in ['expert', 'specialist', 'professional'])
        }
        
        # Credibility assessment
        analysis['credibility_assessment'] = {
            'credibility_score': self._calculate_credibility_score(text_lower),
            'has_specific_dates': bool(re.search(self.entity_patterns['dates'], text)),
            'has_statistics': bool(re.search(r'\b\d+(?:\.\d+)?%', text)),
            'has_citations': 'according to' in text_lower or 'reported by' in text_lower,
            'has_quotes': '"' in text,
            'has_contact_info': any(re.search(pattern, text) for pattern in 
                                  [self.entity_patterns['emails'], self.entity_patterns['phone_numbers']])
        }
        
        # Fact-checking potential
        fact_check_indicators = self._calculate_fact_check_indicators(text_lower)
        analysis['fact_check_potential'] = {
            'statistical_claims_density': fact_check_indicators[0],
            'causal_claims_density': fact_check_indicators[1],
            'temporal_claims_density': fact_check_indicators[2],
            'overall_fact_check_score': sum(fact_check_indicators) / len(fact_check_indicators),
            'high_priority_for_fact_check': sum(fact_check_indicators) > 5
        }
        
        # Summary
        analysis['summary'] = {
            'total_entities_found': sum(len(entities) for entities in analysis['entities_found'].values()),
            'entity_diversity': len([et for et, entities in analysis['entities_found'].items() if entities]),
            'authority_level': 'high' if analysis['authority_assessment']['authority_score'] > 5 else 'medium' if analysis['authority_assessment']['authority_score'] > 2 else 'low',
            'credibility_level': 'high' if analysis['credibility_assessment']['credibility_score'] > 3 else 'medium' if analysis['credibility_assessment']['credibility_score'] > 1 else 'low',
            'fact_check_priority': 'high' if analysis['fact_check_potential']['overall_fact_check_score'] > 3 else 'medium' if analysis['fact_check_potential']['overall_fact_check_score'] > 1 else 'low'
        }
        
        return analysis
    
    def get_verification_suggestions(self, text):
        """Get suggestions for fact-checking and verification"""
        analysis = self.analyze_text_entities(text)
        suggestions = []
        
        # Statistical claims verification
        if analysis['fact_check_potential']['statistical_claims_density'] > 2:
            suggestions.append({
                'type': 'statistical_verification',
                'priority': 'high',
                'suggestion': 'Verify statistical claims and percentages against official sources',
                'specific_claims': re.findall(r'\b\d+(?:\.\d+)?%', text)
            })
        
        # Authority claims verification  
        if analysis['authority_assessment']['authority_score'] > 3:
            suggestions.append({
                'type': 'authority_verification',
                'priority': 'medium',
                'suggestion': 'Verify credentials and affiliations of cited authorities',
                'authorities_mentioned': analysis['entities_found']['government_roles'] + analysis['entities_found']['scientific_terms']
            })
        
        # Source verification
        if analysis['entities_found']['media_outlets']:
            suggestions.append({
                'type': 'source_verification',
                'priority': 'medium',
                'suggestion': 'Cross-reference with original reporting from mentioned media outlets',
                'sources_mentioned': analysis['entities_found']['media_outlets']
            })
        
        # Date and timeline verification
        if analysis['patterns_found']['dates']:
            suggestions.append({
                'type': 'timeline_verification',
                'priority': 'medium',
                'suggestion': 'Verify dates and timeline of events',
                'dates_mentioned': analysis['patterns_found']['dates']
            })
        
        return suggestions