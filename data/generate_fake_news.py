import pandas as pd
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json
import hashlib
import re
from collections import defaultdict
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/fake_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SophisticatedFakeNewsGenerator:
    """Advanced fake news generator with sophisticated templates and quality control"""
    
    def __init__(self):
        self.setup_paths()
        self.setup_templates()
        self.setup_generation_config()
        self.generated_cache = self.load_generated_cache()
    
    def setup_paths(self):
        """Setup all necessary paths"""
        self.base_dir = Path("/tmp")
        self.data_dir = self.base_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_path = self.data_dir / "generated_fake.csv"
        self.metadata_path = self.data_dir / "fake_generation_metadata.json"
        self.cache_path = self.data_dir / "generated_cache.json"
    
    def setup_generation_config(self):
        """Setup generation configuration"""
        self.default_generation_count = 25
        self.min_text_length = 50
        self.max_text_length = 500
        self.max_duplicate_ratio = 0.1
        self.quality_threshold = 0.7
    
    def setup_templates(self):
        """Setup sophisticated fake news templates"""
        
        # Breaking news templates
        self.breaking_templates = [
            "BREAKING: {entity} {action} {location} {timeframe}",
            "URGENT: {authority} confirms {event} in {location}",
            "ALERT: {number} {group} {action} after {event}",
            "EXCLUSIVE: {celebrity} caught {action} with {entity}",
            "DEVELOPING: {event} causes {consequence} across {location}"
        ]
        
        # Conspiracy templates
        self.conspiracy_templates = [
            "EXPOSED: {authority} hiding truth about {topic}",
            "LEAKED: Secret {document} reveals {conspiracy}",
            "WHISTLEBLOWER: {entity} admits {confession}",
            "COVER-UP: {event} was actually {alternative_explanation}",
            "INVESTIGATION: {topic} linked to {conspiracy_group}"
        ]
        
        # Health/science misinformation templates
        self.health_templates = [
            "STUDY: {product} causes {health_effect} in {percentage}% of users",
            "DOCTORS: {treatment} more effective than {alternative}",
            "RESEARCH: {food} linked to {health_condition}",
            "BREAKTHROUGH: {substance} cures {disease} in {timeframe}",
            "WARNING: {activity} increases {health_risk} by {percentage}%"
        ]
        
        # Political misinformation templates
        self.political_templates = [
            "POLL: {percentage}% of {group} support {policy}",
            "INSIDER: {politician} plans to {action} {target}",
            "LEAKED: {document} shows {politician} received {amount} from {entity}",
            "SOURCES: {event} was planned by {political_group}",
            "REVEALED: {policy} will {consequence} {affected_group}"
        ]
        
        # Economic misinformation templates
        self.economic_templates = [
            "CRISIS: {economic_indicator} drops {percentage}% after {event}",
            "PREDICTION: {commodity} prices to {direction} {percentage}% by {timeframe}",
            "ANALYSIS: {economic_policy} will {effect} {economic_sector}",
            "REPORT: {company} to {action} {number} {asset_type}",
            "FORECAST: {economic_event} expected to {consequence}"
        ]
        
        # Template categories
        self.template_categories = {
            'breaking': self.breaking_templates,
            'conspiracy': self.conspiracy_templates,
            'health': self.health_templates,
            'political': self.political_templates,
            'economic': self.economic_templates
        }
        
        # Content variables
        self.content_variables = {
            'entity': [
                'Government officials', 'Tech giants', 'Pharmaceutical companies',
                'Media corporations', 'Intelligence agencies', 'Global elites',
                'Big pharma', 'Wall Street', 'Corporate executives', 'Billionaires'
            ],
            'celebrity': [
                'Hollywood star', 'Tech CEO', 'Pop icon', 'Sports legend',
                'Reality TV star', 'Social media influencer', 'Business mogul'
            ],
            'action': [
                'secretly meeting', 'planning to control', 'manipulating',
                'conspiring against', 'covering up', 'profiting from',
                'exploiting', 'deceiving', 'bribing', 'blackmailing'
            ],
            'location': [
                'major cities', 'rural areas', 'swing states', 'coastal regions',
                'the heartland', 'urban centers', 'suburban communities',
                'border towns', 'industrial areas', 'agricultural regions'
            ],
            'timeframe': [
                'within days', 'by next month', 'before elections',
                'this quarter', 'by year end', 'in the coming weeks',
                'over the holidays', 'during the summit', 'before the deadline'
            ],
            'authority': [
                'Federal agencies', 'State officials', 'Local authorities',
                'International bodies', 'Scientific community', 'Medical experts',
                'Intelligence sources', 'Industry insiders', 'Government whistleblowers'
            ],
            'event': [
                'massive data breach', 'coordinated attack', 'secret experiment',
                'covert operation', 'underground meeting', 'classified project',
                'hidden agenda', 'false flag operation', 'staged incident'
            ],
            'consequence': [
                'economic collapse', 'social unrest', 'mass surveillance',
                'population control', 'mind manipulation', 'health crisis',
                'political upheaval', 'civil liberties erosion', 'market manipulation'
            ],
            'topic': [
                'climate change', 'vaccination programs', 'election integrity',
                'economic policies', 'immigration reform', 'healthcare system',
                'education standards', 'energy independence', 'national security'
            ],
            'conspiracy_group': [
                'shadow government', 'global elite', 'secret society',
                'foreign agents', 'corporate cabal', 'deep state',
                'international conspiracy', 'hidden powers', 'puppet masters'
            ],
            'politician': [
                'Senior officials', 'Cabinet members', 'Congressional leaders',
                'Supreme Court justices', 'Federal judges', 'State governors',
                'Local politicians', 'Party leaders', 'Former presidents'
            ],
            'percentage': [str(x) for x in range(15, 95, 5)],
            'number': [str(x) for x in [100, 500, 1000, 5000, 10000, 50000, 100000]]
        }
    
    def load_generated_cache(self) -> set:
        """Load previously generated content to avoid duplicates"""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    cache_data = json.load(f)
                    # Only keep cache from last 7 days
                    cutoff_date = datetime.now() - timedelta(days=7)
                    recent_content = {
                        content for content, timestamp in cache_data.items()
                        if datetime.fromisoformat(timestamp) > cutoff_date
                    }
                    logger.info(f"Loaded {len(recent_content)} recent generated content from cache")
                    return recent_content
            except Exception as e:
                logger.warning(f"Failed to load generation cache: {e}")
        return set()
    
    def save_generated_cache(self, new_content: Dict[str, str]):
        """Save generated content with timestamps"""
        try:
            # Load existing cache
            cache_data = {}
            if self.cache_path.exists():
                with open(self.cache_path, 'r') as f:
                    cache_data = json.load(f)
            
            # Add new content
            cache_data.update(new_content)
            
            # Save updated cache
            with open(self.cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Saved {len(new_content)} new generated content to cache")
            
        except Exception as e:
            logger.error(f"Failed to save generation cache: {e}")
    
    def generate_realistic_variables(self, category: str) -> Dict[str, str]:
        """Generate realistic variables for templates"""
        variables = {}
        
        # Add specific variables based on category
        if category == 'health':
            variables.update({
                'product': random.choice(['dietary supplement', 'medication', 'device', 'treatment']),
                'health_effect': random.choice(['memory loss', 'organ damage', 'immune suppression', 'cancer']),
                'health_condition': random.choice(['diabetes', 'heart disease', 'arthritis', 'depression']),
                'disease': random.choice(['cancer', 'Alzheimer\'s', 'heart disease', 'diabetes']),
                'substance': random.choice(['natural compound', 'herb', 'vitamin', 'mineral']),
                'treatment': random.choice(['alternative therapy', 'natural remedy', 'new protocol', 'holistic approach']),
                'alternative': random.choice(['traditional medicine', 'pharmaceuticals', 'surgery', 'chemotherapy']),
                'food': random.choice(['processed foods', 'organic vegetables', 'dairy products', 'gluten']),
                'activity': random.choice(['using smartphones', 'eating sugar', 'lack of exercise', 'stress']),
                'health_risk': random.choice(['cancer risk', 'heart disease', 'cognitive decline', 'immune dysfunction'])
            })
        
        elif category == 'political':
            variables.update({
                'policy': random.choice(['immigration reform', 'healthcare policy', 'tax legislation', 'trade deal']),
                'political_group': random.choice(['opposition party', 'special interests', 'foreign powers', 'lobbyists']),
                'document': random.choice(['internal memo', 'classified report', 'email chain', 'phone transcript']),
                'amount': random.choice(['$1 million', '$10 million', '$100 million', '$1 billion']),
                'affected_group': random.choice(['middle class', 'seniors', 'small businesses', 'workers']),
                'target': random.choice(['social programs', 'military spending', 'tax rates', 'regulations'])
            })
        
        elif category == 'economic':
            variables.update({
                'economic_indicator': random.choice(['GDP', 'unemployment rate', 'inflation', 'stock market']),
                'commodity': random.choice(['oil', 'gold', 'wheat', 'lumber']),
                'direction': random.choice(['rise', 'fall', 'surge', 'plummet']),
                'economic_policy': random.choice(['tax cuts', 'stimulus package', 'trade tariffs', 'interest rates']),
                'economic_sector': random.choice(['manufacturing', 'technology', 'healthcare', 'agriculture']),
                'company': random.choice(['Tech giants', 'Major banks', 'Energy companies', 'Retail chains']),
                'asset_type': random.choice(['factories', 'stores', 'offices', 'facilities']),
                'economic_event': random.choice(['recession', 'market crash', 'inflation surge', 'currency devaluation']),
                'effect': random.choice(['boost', 'harm', 'transform', 'destroy'])
            })
        
        # Add common variables
        for var_type, options in self.content_variables.items():
            if var_type not in variables:
                variables[var_type] = random.choice(options)
        
        return variables
    
    def create_supporting_content(self, headline: str, category: str) -> str:
        """Create supporting content to make the fake news more believable"""
        supporting_sentences = []
        
        if category == 'breaking':
            supporting_sentences = [
                "Sources close to the situation report that this development was unexpected.",
                "Officials have not yet released an official statement regarding these events.",
                "The situation is rapidly evolving, with more details expected soon.",
                "Multiple witnesses have come forward with similar accounts.",
                "This story is developing, and updates will be provided as they become available."
            ]
        
        elif category == 'conspiracy':
            supporting_sentences = [
                "This information comes from anonymous sources within the organization.",
                "The evidence has been circulating in underground networks for months.",
                "Mainstream media has been reluctant to cover this story.",
                "Independent researchers have been investigating this for years.",
                "The full extent of the cover-up is only now coming to light."
            ]
        
        elif category == 'health':
            supporting_sentences = [
                "The findings were published in a peer-reviewed journal.",
                "Medical experts are calling for immediate action.",
                "The study followed participants for an extended period.",
                "Previous research has suggested similar connections.",
                "Health authorities are reviewing the new evidence."
            ]
        
        elif category == 'political':
            supporting_sentences = [
                "The revelations have sparked calls for investigation.",
                "Political opponents are demanding transparency.",
                "The timing of this disclosure raises serious questions.",
                "Legal experts suggest this could have major implications.",
                "The public deserves to know the truth about these matters."
            ]
        
        elif category == 'economic':
            supporting_sentences = [
                "Market analysts are closely monitoring the situation.",
                "The economic implications could be far-reaching.",
                "Investors are already reacting to the preliminary reports.",
                "Similar patterns have been observed in other markets.",
                "The full impact may not be known for several quarters."
            ]
        
        # Select 2-3 supporting sentences
        selected_sentences = random.sample(supporting_sentences, min(3, len(supporting_sentences)))
        supporting_content = " ".join(selected_sentences)
        
        return f"{headline} {supporting_content}"
    
    def validate_generated_content(self, content: str) -> Tuple[bool, str]:
        """Validate generated content quality"""
        # Check minimum length
        if len(content) < self.min_text_length:
            return False, "Content too short"
        
        if len(content) > self.max_text_length:
            return False, "Content too long"
        
        # Check for placeholder variables
        if '{' in content or '}' in content:
            return False, "Unfilled template variables"
        
        # Check for meaningful content
        if not any(c.isalpha() for c in content):
            return False, "No alphabetic content"
        
        # Check for sentence structure
        if not any(punct in content for punct in '.!?'):
            return False, "No sentence structure"
        
        # Check for duplicate content
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in self.generated_cache:
            return False, "Duplicate content"
        
        # Check for excessive repetition
        words = content.lower().split()
        if len(words) > 0:
            word_counts = defaultdict(int)
            for word in words:
                word_counts[word] += 1
            
            max_repetition = max(word_counts.values())
            if max_repetition > len(words) * 0.3:  # More than 30% repetition
                return False, "Excessive word repetition"
        
        return True, "Content passed validation"
    
    def generate_single_fake_news(self, category: str = None) -> Optional[Dict]:
        """Generate a single fake news article"""
        try:
            # Select category
            if category is None:
                category = random.choice(list(self.template_categories.keys()))
            
            # Select template
            template = random.choice(self.template_categories[category])
            
            # Generate variables
            variables = self.generate_realistic_variables(category)
            
            # Fill template
            headline = template.format(**variables)
            
            # Create supporting content
            full_content = self.create_supporting_content(headline, category)
            
            # Validate content
            is_valid, reason = self.validate_generated_content(full_content)
            if not is_valid:
                logger.debug(f"Generated content validation failed ({reason}): {headline[:50]}...")
                return None
            
            # Create article data
            article_data = {
                'text': full_content,
                'label': 1,  # Fake news
                'source': 'synthetic_generation',
                'category': category,
                'template': template,
                'headline': headline,
                'timestamp': datetime.now().isoformat(),
                'word_count': len(full_content.split()),
                'char_count': len(full_content),
                'generation_method': 'template_based'
            }
            
            logger.debug(f"Generated fake news: {headline}")
            return article_data
            
        except Exception as e:
            logger.warning(f"Failed to generate fake news: {str(e)}")
            return None
    
    def generate_fake_news_batch(self, count: int = None) -> List[Dict]:
        """Generate a batch of fake news articles"""
        if count is None:
            count = self.default_generation_count
        
        logger.info(f"Starting generation of {count} fake news articles...")
        
        articles = []
        generated_content = {}
        max_attempts = count * 3  # Allow some failed attempts
        attempt = 0
        
        # Ensure category distribution
        categories = list(self.template_categories.keys())
        articles_per_category = count // len(categories)
        remaining_articles = count % len(categories)
        
        category_targets = {cat: articles_per_category for cat in categories}
        
        # Distribute remaining articles
        for i in range(remaining_articles):
            category_targets[categories[i]] += 1
        
        category_counts = {cat: 0 for cat in categories}
        
        while len(articles) < count and attempt < max_attempts:
            attempt += 1
            
            # Select category based on targets
            available_categories = [
                cat for cat, target in category_targets.items()
                if category_counts[cat] < target
            ]
            
            if not available_categories:
                break
            
            category = random.choice(available_categories)
            
            article_data = self.generate_single_fake_news(category)
            
            if article_data:
                articles.append(article_data)
                category_counts[category] += 1
                
                # Add to generated content cache
                content_hash = hashlib.md5(article_data['text'].encode()).hexdigest()
                generated_content[content_hash] = datetime.now().isoformat()
        
        # Save generated content to cache
        if generated_content:
            self.save_generated_cache(generated_content)
        
        logger.info(f"Generated {len(articles)} fake news articles")
        return articles
    
    def save_generated_articles(self, articles: List[Dict]) -> bool:
        """Save generated fake news articles to CSV"""
        try:
            if not articles:
                logger.info("No articles to save")
                return True
            
            # Create DataFrame
            df_new = pd.DataFrame(articles)
            
            # Load existing data if present
            if self.output_path.exists():
                try:
                    df_existing = pd.read_csv(self.output_path)
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    
                    # Remove duplicates based on text hash
                    df_combined['text_hash'] = df_combined['text'].apply(
                        lambda x: hashlib.md5(x.encode()).hexdigest()
                    )
                    df_combined = df_combined.drop_duplicates(subset=['text_hash'], keep='last')
                    df_combined = df_combined.drop('text_hash', axis=1)
                    
                    logger.info(f"Combined with existing data. Total: {len(df_combined)} articles")
                    
                except Exception as e:
                    logger.warning(f"Failed to load existing data: {e}")
                    df_combined = df_new
            else:
                df_combined = df_new
            
            # Save to CSV
            df_combined.to_csv(self.output_path, index=False)
            
            logger.info(f"Successfully saved {len(articles)} new fake articles to {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save articles: {str(e)}")
            return False
    
    def generate_metadata(self, articles: List[Dict]) -> Dict:
        """Generate metadata about the generation session"""
        if not articles:
            return {}
        
        df = pd.DataFrame(articles)
        
        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'articles_generated': len(articles),
            'category_distribution': df['category'].value_counts().to_dict(),
            'average_word_count': float(df['word_count'].mean()),
            'total_characters': int(df['char_count'].sum()),
            'unique_templates': df['template'].nunique(),
            'quality_score': self.calculate_generation_quality(df)
        }
        
        return metadata
    
    def calculate_generation_quality(self, df: pd.DataFrame) -> float:
        """Calculate quality score for generated articles"""
        scores = []
        
        # Diversity score (different categories)
        category_diversity = df['category'].nunique() / len(self.template_categories)
        scores.append(category_diversity)
        
        # Template diversity score
        template_diversity = df['template'].nunique() / len(df)
        scores.append(template_diversity)
        
        # Length consistency score
        word_counts = df['word_count']
        if word_counts.std() > 0:
            length_score = 1.0 - (word_counts.std() / word_counts.mean())
            scores.append(max(0, min(1, length_score)))
        else:
            scores.append(1.0)
        
        return float(sum(scores) / len(scores))
    
    def generate_fake_news(self, count: int = None) -> Tuple[bool, str]:
        """Main function to generate fake news articles"""
        try:
            logger.info("Starting fake news generation process...")
            
            # Generate articles
            articles = self.generate_fake_news_batch(count)
            
            if not articles:
                logger.warning("No articles were generated successfully")
                return False, "No articles generated"
            
            # Save articles
            if not self.save_generated_articles(articles):
                return False, "Failed to save generated articles"
            
            # Generate and save metadata
            metadata = self.generate_metadata(articles)
            
            try:
                with open(self.metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save metadata: {e}")
            
            success_msg = f"Successfully generated {len(articles)} fake news articles"
            logger.info(success_msg)
            
            return True, success_msg
            
        except Exception as e:
            error_msg = f"Generation process failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

def generate_fake_news(count: int = 25):
    """Main function for external calls"""
    generator = SophisticatedFakeNewsGenerator()
    success, message = generator.generate_fake_news(count)
    
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
    
    return success

def main():
    """Main execution function"""
    generator = SophisticatedFakeNewsGenerator()
    success, message = generator.generate_fake_news()
    
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
        exit(1)

if __name__ == "__main__":
    main()