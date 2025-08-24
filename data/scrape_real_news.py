import re
import time
import json
import random
import hashlib
import logging
import requests
import pandas as pd
from pathlib import Path
from newspaper import Article, build
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional, Tuple
from data.validation_schemas import ValidationLevel
from data.data_validator import DataValidationPipeline
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/scraping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RobustNewsScraper:
    """Production-ready news scraper with comprehensive error handling and rate limiting"""
    
    def __init__(self):
        self.setup_paths()
        self.setup_scraping_config()
        self.session = self.create_session()
        self.scraped_urls = self.load_scraped_urls()
    
    def setup_paths(self):
        """Setup all necessary paths"""
        self.base_dir = Path("/tmp")
        self.data_dir = self.base_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_path = self.data_dir / "scraped_real.csv"
        self.metadata_path = self.data_dir / "scraping_metadata.json"
        self.urls_cache_path = self.data_dir / "scraped_urls.json"
    
    def setup_scraping_config(self):
        """Setup scraping configuration"""
        self.news_sites = [
            {
                "name": "Reuters",
                "url": "https://www.reuters.com/",
                "max_articles": 8,
                "delay": 2.0
            },
            {
                "name": "BBC",
                "url": "https://www.bbc.com/news",
                "max_articles": 7,
                "delay": 2.5
            },
            {
                "name": "NPR",
                "url": "https://www.npr.org/",
                "max_articles": 5,
                "delay": 3.0
            },
            {
                "name": "Associated Press",
                "url": "https://apnews.com/",
                "max_articles": 5,
                "delay": 2.0
            }
        ]
        
        self.max_articles_total = 20
        self.min_article_length = 100
        self.max_article_length = 10000
        self.scraping_timeout = 30
        self.max_retries = 3
    
    def create_session(self) -> requests.Session:
        """Create configured requests session"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        return session
    
    def load_scraped_urls(self) -> set:
        """Load previously scraped URLs to avoid duplicates"""
        if self.urls_cache_path.exists():
            try:
                with open(self.urls_cache_path, 'r') as f:
                    urls_data = json.load(f)
                    # Only keep URLs from last 30 days
                    cutoff_date = datetime.now() - timedelta(days=30)
                    recent_urls = {
                        url for url, timestamp in urls_data.items()
                        if datetime.fromisoformat(timestamp) > cutoff_date
                    }
                    logger.info(f"Loaded {len(recent_urls)} recent URLs from cache")
                    return recent_urls
            except Exception as e:
                logger.warning(f"Failed to load URL cache: {e}")
        return set()
    
    def save_scraped_urls(self, new_urls: Dict[str, str]):
        """Save scraped URLs with timestamps"""
        try:
            # Load existing URLs
            urls_data = {}
            if self.urls_cache_path.exists():
                with open(self.urls_cache_path, 'r') as f:
                    urls_data = json.load(f)
            
            # Add new URLs
            urls_data.update(new_urls)
            
            # Save updated cache
            with open(self.urls_cache_path, 'w') as f:
                json.dump(urls_data, f, indent=2)
            
            logger.info(f"Saved {len(new_urls)} new URLs to cache")
            
        except Exception as e:
            logger.error(f"Failed to save URL cache: {e}")
    
    def validate_article_quality(self, article: Article) -> Tuple[bool, str]:
        """Validate article quality with comprehensive checks"""
        # Check if article has minimum content
        if not article.text or len(article.text.strip()) < self.min_article_length:
            return False, "Article too short"
        
        if len(article.text) > self.max_article_length:
            return False, "Article too long"
        
        # Check if article has title
        if not article.title or len(article.title.strip()) < 10:
            return False, "Missing or inadequate title"
        
        # Check for meaningful content
        if not any(c.isalpha() for c in article.text):
            return False, "No alphabetic content"
        
        # Check for sentence structure
        if not any(punct in article.text for punct in '.!?'):
            return False, "No sentence structure"
        
        # Check for excessive HTML artifacts
        html_patterns = [
            r'<[^>]+>',
            r'&[a-zA-Z]+;',
            r'javascript:',
            r'document\.',
            r'window\.'
        ]
        
        for pattern in html_patterns:
            if len(re.findall(pattern, article.text)) > 5:
                return False, "Excessive HTML artifacts"
        
        # Check for advertising content
        ad_keywords = [
            'advertisement', 'sponsored', 'click here', 'buy now',
            'subscribe', 'newsletter', 'cookies', 'privacy policy'
        ]
        
        text_lower = article.text.lower()
        ad_count = sum(1 for keyword in ad_keywords if keyword in text_lower)
        if ad_count > 3:
            return False, "Excessive advertising content"
        
        return True, "Article passed validation"
    
    def clean_article_text(self, text: str) -> str:
        """Clean and normalize article text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Remove non-printable characters
        text = ''.join(char for char in text if ord(char) >= 32)
        
        return text.strip()
    
    def scrape_single_article(self, url: str) -> Optional[Dict]:
        """Scrape a single article with comprehensive error handling"""
        try:
            # Check if URL already scraped
            if url in self.scraped_urls:
                return None
            
            # Create article object
            article = Article(url)
            
            # Download with timeout
            article.download()
            
            # Parse article
            article.parse()
            
            # Validate article quality
            is_valid, reason = self.validate_article_quality(article)
            if not is_valid:
                logger.debug(f"Article validation failed ({reason}): {url}")
                return None
            
            # Clean article text
            clean_title = self.clean_article_text(article.title)
            clean_text = self.clean_article_text(article.text)
            
            # Combine title and text
            full_text = f"{clean_title}. {clean_text}"
            
            # Create article data
            article_data = {
                'text': full_text,
                'label': 0,  # Real news
                'source': urlparse(url).netloc,
                'url': url,
                'title': clean_title,
                'timestamp': datetime.now().isoformat(),
                'word_count': len(full_text.split()),
                'char_count': len(full_text)
            }
            
            logger.info(f"Successfully scraped article: {clean_title[:50]}...")
            return article_data
            
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {str(e)}")
            return None
    
    def scrape_site_articles(self, site_config: Dict) -> List[Dict]:
        """Scrape articles from a single news site"""
        logger.info(f"Starting scraping from {site_config['name']}...")
        
        articles = []
        scraped_urls = {}
        
        try:
            # Build newspaper object
            paper = build(site_config['url'], memoize_articles=False)
            
            # Get article URLs
            article_urls = [article.url for article in paper.articles]
            
            # Filter out already scraped URLs
            new_urls = [url for url in article_urls if url not in self.scraped_urls]
            
            # Shuffle URLs for randomness
            random.shuffle(new_urls)
            
            # Limit number of articles
            urls_to_scrape = new_urls[:site_config['max_articles']]
            
            logger.info(f"Found {len(urls_to_scrape)} new articles to scrape from {site_config['name']}")
            
            # Scrape articles with rate limiting
            for i, url in enumerate(urls_to_scrape):
                if len(articles) >= site_config['max_articles']:
                    break
                
                article_data = self.scrape_single_article(url)
                
                if article_data:
                    articles.append(article_data)
                    scraped_urls[url] = datetime.now().isoformat()
                
                # Rate limiting
                if i < len(urls_to_scrape) - 1:
                    time.sleep(site_config['delay'])
            
            # Save scraped URLs
            if scraped_urls:
                self.save_scraped_urls(scraped_urls)
            
            logger.info(f"Successfully scraped {len(articles)} articles from {site_config['name']}")
            
        except Exception as e:
            logger.error(f"Error scraping {site_config['name']}: {str(e)}")
        
        return articles
    
    def scrape_all_sources(self) -> List[Dict]:
        """Scrape articles from all configured sources"""
        logger.info("Starting comprehensive news scraping...")
        
        all_articles = []
        
        # Scrape from each source
        for site_config in self.news_sites:
            if len(all_articles) >= self.max_articles_total:
                break
            
            try:
                site_articles = self.scrape_site_articles(site_config)
                all_articles.extend(site_articles)
                
                # Delay between sites
                if site_config != self.news_sites[-1]:
                    time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error scraping {site_config['name']}: {str(e)}")
                continue
        
        # Limit total articles
        all_articles = all_articles[:self.max_articles_total]
        
        logger.info(f"Scraping complete. Total articles: {len(all_articles)}")
        return all_articles
    
    def save_scraped_articles(self, articles: List[Dict]) -> bool:
        """Save scraped articles with validation"""
        try:
            if not articles:
                return True
            
            # Validate articles first
            valid_articles, validation_summary = self.validate_scraped_articles(articles)
            
            logger.info(f"Validation: {len(valid_articles)}/{len(articles)} articles passed validation")
            
            if not valid_articles:
                logger.warning("No valid articles to save after validation")
                return True
            
            # Create DataFrame and save
            df_new = pd.DataFrame(valid_articles)
            
            # Existing file handling logic...
            if self.output_path.exists():
                df_existing = pd.read_csv(self.output_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=['text'], keep='first')
            else:
                df_combined = df_new
            
            df_combined.to_csv(self.output_path, index=False)
            
            # Save validation report
            validation_report_path = self.data_dir / "scraping_validation_report.json"
            with open(validation_report_path, 'w') as f:
                json.dump(validation_summary, f, indent=2)
            
            logger.info(f"Saved {len(valid_articles)} validated articles to {self.output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save validated articles: {e}")
            return False
    
    def generate_scraping_metadata(self, articles: List[Dict]) -> Dict:
        """Generate metadata about the scraping session"""
        if not articles:
            return {}
        
        df = pd.DataFrame(articles)
        
        metadata = {
            'scraping_timestamp': datetime.now().isoformat(),
            'articles_scraped': len(articles),
            'sources': df['source'].value_counts().to_dict(),
            'average_word_count': float(df['word_count'].mean()),
            'total_characters': int(df['char_count'].sum()),
            'scraping_duration': None,  # Will be set by caller
            'quality_score': self.calculate_scraping_quality(df)
        }
        
        return metadata
    
    def calculate_scraping_quality(self, df: pd.DataFrame) -> float:
        """Calculate quality score for scraped articles"""
        scores = []
        
        # Diversity score (different sources)
        source_diversity = df['source'].nunique() / len(self.news_sites)
        scores.append(source_diversity)
        
        # Length consistency score
        word_counts = df['word_count']
        length_score = 1.0 - (word_counts.std() / word_counts.mean())
        scores.append(max(0, min(1, length_score)))
        
        # Freshness score (all articles should be recent)
        freshness_score = 1.0  # All articles are fresh by definition
        scores.append(freshness_score)
        
        return float(sum(scores) / len(scores))
    
    def scrape_articles(self) -> Tuple[bool, str]:
        """Main scraping function with comprehensive error handling"""
        start_time = time.time()
        
        try:
            logger.info("Starting news scraping process...")
            
            # Scrape articles from all sources
            articles = self.scrape_all_sources()
            
            if not articles:
                logger.warning("No articles were scraped successfully")
                return False, "No articles scraped"
            
            # Save articles
            if not self.save_scraped_articles(articles):
                return False, "Failed to save articles"
            
            # Generate and save metadata
            metadata = self.generate_scraping_metadata(articles)
            metadata['scraping_duration'] = time.time() - start_time
            
            try:
                with open(self.metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save metadata: {e}")
            
            success_msg = f"Successfully scraped {len(articles)} articles"
            logger.info(success_msg)
            
            return True, success_msg
            
        except Exception as e:
            error_msg = f"Scraping process failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def validate_scraped_articles(self, articles: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Validate scraped articles using validation schemas"""
        if not articles:
            return articles, {}
        
        validator = DataValidationPipeline()
        
        # Ensure required fields for validation
        enhanced_articles = []
        for article in articles:
            enhanced_article = article.copy()
            if 'source' not in enhanced_article:
                enhanced_article['source'] = 'scraped_real'
            if 'label' not in enhanced_article:
                enhanced_article['label'] = 0  # Real news
            enhanced_articles.append(enhanced_article)
        
        # Validate batch
        validation_result = validator.validate_scraped_data(enhanced_articles, "web_scraping")
        
        # Filter valid articles
        valid_articles = []
        for i, result in enumerate(validation_result.validation_results):
            if result.is_valid:
                article = enhanced_articles[i].copy()
                article['validation_quality_score'] = result.quality_metrics.get('overall_quality_score', 0.0)
                valid_articles.append(article)
        
        validation_summary = {
            'original_count': len(articles),
            'valid_count': len(valid_articles),
            'success_rate': validation_result.success_rate,
            'overall_quality_score': validation_result.overall_quality_score
        }
        
        return valid_articles, validation_summary
    
    
def scrape_articles():
    """Main function for external calls"""
    scraper = RobustNewsScraper()
    success, message = scraper.scrape_articles()
    
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
    
    return success

def main():
    """Main execution function"""
    scraper = RobustNewsScraper()
    success, message = scraper.scrape_articles()
    
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
        exit(1)

if __name__ == "__main__":
    main()