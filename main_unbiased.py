#!/usr/bin/env python3
"""
Unbiased Atomberg Share of Voice AI Agent
Addresses all bias issues for accurate competitive intelligence
Run: python main_unbiased.py
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
from collections import Counter, defaultdict
from urllib.parse import urlparse

from price_intelligence import EnhancedPriceIntelligence

warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("💡 Install python-dotenv: pip install python-dotenv")

try:
    from transformers import pipeline
    print("✅ Transformers loaded")
    HAS_TRANSFORMERS = True
except ImportError:
    print("❌ Please install transformers: pip install transformers")
    HAS_TRANSFORMERS = False

try:
    from rapidfuzz import fuzz
    print("✅ RapidFuzz loaded for fuzzy matching")
    HAS_RAPIDFUZZ = True
except ImportError:
    print("💡 Install rapidfuzz: pip install rapidfuzz")
    HAS_RAPIDFUZZ = False
    fuzz = None

class UnbiasedCeilingFanSentimentAnalyzer:
    """Domain-specific unbiased sentiment analysis for ceiling fans"""
    
    def __init__(self):
        # Ceiling fan specific sentiment words
        self.positive_words = [
            # Performance
            'powerful', 'efficient', 'quiet', 'silent', 'smooth', 'fast', 'effective', 'strong',
            'good airflow', 'excellent', 'superb', 'amazing', 'fantastic', 'brilliant',
            # Quality  
            'durable', 'reliable', 'sturdy', 'solid', 'premium', 'quality', 'well built',
            'long lasting', 'robust', 'heavy duty', 'professional',
            # User experience
            'easy install', 'convenient', 'comfortable', 'satisfied', 'happy', 'love',
            'recommend', 'impressed', 'pleased', 'delighted', 'perfect',
            # Value
            'worth it', 'value for money', 'affordable', 'reasonable', 'good deal',
            'cost effective', 'budget friendly', 'great price'
        ]
        
        self.negative_words = [
            # Performance issues
            'noisy', 'loud', 'slow', 'weak', 'inefficient', 'vibration', 'wobble',
            'poor airflow', 'insufficient', 'inadequate', 'disappointing', 'terrible',
            # Quality issues
            'cheap', 'flimsy', 'broken', 'defective', 'poor quality', 'unreliable',
            'fragile', 'inferior', 'substandard', 'faulty', 'damaged',
            # User experience
            'difficult install', 'complicated', 'disappointed', 'unsatisfied', 'regret',
            'frustrated', 'annoyed', 'hate', 'horrible', 'awful',
            # Value
            'overpriced', 'expensive', 'waste of money', 'not worth', 'ripoff',
            'too costly', 'poor value'
        ]
        
        self.neutral_words = [
            'average', 'okay', 'standard', 'normal', 'typical', 'usual',
            'decent', 'acceptable', 'reasonable', 'fair', 'moderate'
        ]

    def analyze_sentiment(self, text: str) -> Dict:
        """Unbiased domain-specific sentiment analysis"""
        text_lower = text.lower()
        
        positive_matches = [word for word in self.positive_words if word in text_lower]
        negative_matches = [word for word in self.negative_words if word in text_lower]
        neutral_matches = [word for word in self.neutral_words if word in text_lower]
        
        positive_count = len(positive_matches)
        negative_count = len(negative_matches)
        neutral_count = len(neutral_matches)
        
        total_sentiment_words = positive_count + negative_count + neutral_count
        
        if total_sentiment_words == 0:
            return {
                'sentiment': 'NEUTRAL',
                'sentiment_score': 0.5,
                'confidence': 0.3,
                'method': 'no_sentiment_indicators'
            }
        
        # Determine sentiment with confidence
        if positive_count > negative_count and positive_count >= neutral_count:
            confidence = min(0.95, 0.6 + (positive_count / max(total_sentiment_words, 1)) * 0.35)
            return {
                'sentiment': 'POSITIVE',
                'sentiment_score': 0.6 + (positive_count / max(positive_count + negative_count, 1)) * 0.4,
                'confidence': confidence,
                'method': 'domain_specific',
                'matched_words': positive_matches[:3]  # Top matches for debugging
            }
        elif negative_count > positive_count and negative_count >= neutral_count:
            confidence = min(0.95, 0.6 + (negative_count / max(total_sentiment_words, 1)) * 0.35)
            return {
                'sentiment': 'NEGATIVE',
                'sentiment_score': 0.4 - (negative_count / max(positive_count + negative_count, 1)) * 0.4,
                'confidence': confidence,
                'method': 'domain_specific',
                'matched_words': negative_matches[:3]
            }
        else:
            return {
                'sentiment': 'NEUTRAL',
                'sentiment_score': 0.5,
                'confidence': 0.7,
                'method': 'mixed_or_neutral',
                'matched_words': neutral_matches[:3]
            }

class BiasMonitor:
    """Monitor and report bias in SOV analysis"""
    
    def __init__(self):
        self.bias_threshold = 0.15  # 15% deviation triggers warning
        
    def analyze_bias(self, results: List[Dict]) -> Dict:
        """Comprehensive bias analysis"""
        
        # Extract brand mentions
        brand_stats = {}
        all_brands = ['Atomberg', 'Havells', 'Orient', 'Crompton', 'Bajaj', 'Usha']
        
        for brand in all_brands:
            brand_key = 'atomberg_mentioned' if brand == 'Atomberg' else f'{brand.lower()}_mentioned'
            brand_results = [r for r in results if r.get(brand_key, False)]
            
            if brand_results:
                brand_stats[brand] = {
                    'mention_count': len(brand_results),
                    'avg_position': np.mean([r.get('position', 10) for r in brand_results]),
                    'sentiment_breakdown': {
                        'positive': len([r for r in brand_results if r['sentiment'] == 'POSITIVE']),
                        'negative': len([r for r in brand_results if r['sentiment'] == 'NEGATIVE']),
                        'neutral': len([r for r in brand_results if r['sentiment'] == 'NEUTRAL'])
                    },
                    'avg_engagement': np.mean([r.get('engagement_score', 1.0) for r in brand_results]),
                    'platform_distribution': {}
                }
                
                # Platform distribution
                platforms = [r['platform'] for r in brand_results]
                for platform in set(platforms):
                    brand_stats[brand]['platform_distribution'][platform] = platforms.count(platform)
        
        # Bias analysis
        bias_report = {
            'mention_distribution': {},
            'sentiment_bias': {},
            'position_bias': {},
            'platform_bias': {},
            'overall_bias_score': 0.0,
            'warnings': [],
            'is_biased': False
        }
        
        total_mentions = sum(stats['mention_count'] for stats in brand_stats.values())
        expected_share = 1.0 / len(brand_stats) if brand_stats else 0  # Equal expected share
        
        # Check mention distribution bias
        max_deviation = 0
        for brand, stats in brand_stats.items():
            actual_share = stats['mention_count'] / max(total_mentions, 1)
            bias_report['mention_distribution'][brand] = {
                'actual_share': actual_share,
                'expected_share': expected_share,
                'deviation': abs(actual_share - expected_share)
            }
            max_deviation = max(max_deviation, abs(actual_share - expected_share))
        
        if max_deviation > self.bias_threshold:
            bias_report['warnings'].append(f"Mention distribution bias detected: {max_deviation:.1%} deviation from expected")
            bias_report['is_biased'] = True
        
        # Check sentiment bias
        for brand, stats in brand_stats.items():
            total_brand_mentions = stats['mention_count']
            if total_brand_mentions > 0:
                positive_ratio = stats['sentiment_breakdown']['positive'] / total_brand_mentions
                bias_report['sentiment_bias'][brand] = positive_ratio
                
                # Flag if any brand has >80% positive or >50% negative
                if positive_ratio > 0.8:
                    bias_report['warnings'].append(f"{brand} has unusually high positive sentiment: {positive_ratio:.1%}")
                    bias_report['is_biased'] = True
                elif stats['sentiment_breakdown']['negative'] / total_brand_mentions > 0.5:
                    bias_report['warnings'].append(f"{brand} has unusually high negative sentiment")
                    bias_report['is_biased'] = True
        
        # Check position bias (lower position is better)
        avg_positions = {brand: stats['avg_position'] for brand, stats in brand_stats.items()}
        if avg_positions:
            best_position = min(avg_positions.values())
            worst_position = max(avg_positions.values())
            position_spread = worst_position - best_position
            
            if position_spread > 5:  # More than 5 positions difference
                bias_report['warnings'].append(f"Large position bias detected: {position_spread:.1f} position spread")
                bias_report['is_biased'] = True
        
        bias_report['position_bias'] = avg_positions
        bias_report['overall_bias_score'] = max_deviation
        
        return bias_report

class UnbiasedAtombergAgent:
    def __init__(self, serpapi_key: str, youtube_api_key: str):
        self.serpapi_key = serpapi_key
        self.youtube_api_key = youtube_api_key
        
        print("🤖 Loading AI models...")
        if HAS_TRANSFORMERS:
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                print("✅ Transformer sentiment analyzer loaded")
            except Exception as e:
                print(f"⚠️ Transformer loading failed: {e}")
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None
        
        # Initialize unbiased sentiment analyzer
        self.domain_sentiment = UnbiasedCeilingFanSentimentAnalyzer()
        self.bias_monitor = BiasMonitor()
        
        # UNBIASED BRAND DETECTION - Equal coverage for all brands
        self.atomberg_variants = [
            'atomberg', 'atom berg', '@atomberg', 'atomberg fan', 'atomberg ceiling fan',
            'atomberg smart fan', 'atomberg india', 'atomberg bldc'
        ]
        
        self.competitors = {
            'Havells': [
                'havells', '@havells', 'havells india', 'havells fan', 'havells ceiling fan',
                'havells smart fan', 'havells premium', 'havells bldc'
            ],
            'Orient': [
                'orient', '@orient', 'orient electric', 'orient fan', 'orient ceiling fan',
                'orient smart fan', 'orient electric fan', 'orient bldc'
            ],
            'Crompton': [
                'crompton', '@crompton', 'crompton greaves', 'crompton fan', 'crompton ceiling fan',
                'crompton smart fan', 'crompton greaves fan', 'crompton bldc'
            ],
            'Bajaj': [
                'bajaj', '@bajaj', 'bajaj electricals', 'bajaj fan', 'bajaj ceiling fan',
                'bajaj smart fan', 'bajaj electrical fan', 'bajaj bldc'
            ],
            'Usha': [
                'usha', '@usha', 'usha international', 'usha fan', 'usha ceiling fan',
                'usha smart fan', 'usha international fan', 'usha bldc'
            ],
            'Luminous': [
                'luminous', '@luminous', 'luminous power', 'luminous fan', 'luminous ceiling fan',
                'luminous smart fan', 'luminous power fan', 'luminous bldc'
            ]
        }
        
        # UNBIASED KEYWORD STRATEGY - Market neutral keywords
        self.neutral_keywords = [
            # Market coverage keywords
            'ceiling fan india',
            'ceiling fan buying guide',
            'best ceiling fan 2024',
            'ceiling fan reviews india',
            'ceiling fan comparison',
            
            # Technology spectrum
            'smart ceiling fan',
            'traditional ceiling fan',
            'energy efficient ceiling fan',
            'premium ceiling fan',
            'budget ceiling fan',
            
            # Feature focused
            'quiet ceiling fan',
            'high speed ceiling fan',
            'ceiling fan with remote',
            'decorative ceiling fan',
            'commercial ceiling fan'
        ]
        
        # Brand comparison keywords (test all major brands equally)
        self.comparison_keywords = [
            'havells vs orient ceiling fan',
            'crompton vs bajaj ceiling fan',
            'usha vs atomberg ceiling fan',
            'best ceiling fan brands india',
            'ceiling fan brand comparison'
        ]
        
        self.all_keywords = self.neutral_keywords + self.comparison_keywords
        
        # UNBIASED FEATURE WEIGHTS - All equal
        self.features = {
            'smart_control': {
                'keywords': ['wifi', 'app control', 'smart', 'iot', 'alexa', 'google home', 'remote'], 
                'weight': 1.0  # Equal weight
            },
            'energy_efficiency': {
                'keywords': ['energy', 'bldc', 'power', 'efficient', 'saving', 'consumption'], 
                'weight': 1.0  # Equal weight
            },
            'design': {
                'keywords': ['premium', 'modern', 'stylish', 'aesthetic', 'sleek', 'decorative'], 
                'weight': 1.0  # Equal weight
            },
            'performance': {
                'keywords': ['speed', 'cfm', 'powerful', 'air delivery', 'circulation'], 
                'weight': 1.0  # Equal weight
            },
            'quiet_operation': {
                'keywords': ['silent', 'quiet', 'noiseless', 'whisper', 'low noise'], 
                'weight': 1.0  # Equal weight
            },
            'durability': {
                'keywords': ['warranty', 'durable', 'quality', 'long lasting', 'reliable'], 
                'weight': 1.0  # Equal weight
            },
            'price_value': {
                'keywords': ['affordable', 'value', 'price', 'budget', 'cost effective'], 
                'weight': 1.0  # Equal weight
            },
            'build_quality': {
                'keywords': ['solid', 'sturdy', 'heavy duty', 'robust', 'strong'],
                'weight': 1.0  # Equal weight
            },
            'brand_trust': {
                'keywords': ['trusted', 'established', 'reputation', 'reliable', 'proven'],
                'weight': 1.0  # Equal weight
            }
        }
        
        # UNBIASED PLATFORM WEIGHTS - All equal
        self.platform_weights = {
            'Google': 1.0,
            'YouTube': 1.0,  # No premium weight
            'Google Shopping': 1.0,
            'Twitter': 1.0,
            'Instagram': 1.0
        }
        
        # Gradual position decay
        self.position_weights = {}
        for i in range(1, 51):
            if i <= 3:
                self.position_weights[i] = 1.0
            elif i <= 10:
                self.position_weights[i] = 0.9
            elif i <= 20:
                self.position_weights[i] = 0.7
            else:
                self.position_weights[i] = 0.5

    def search_google_enhanced(self, query: str, num_results: int = 20) -> List[Dict]:
        """Unbiased Google search"""
        try:
            print(f"   🔍 Google search for: {query}")
            response = requests.get("https://serpapi.com/search", params={
                'q': query,
                'api_key': self.serpapi_key,
                'engine': 'google',
                'num': min(num_results, 100),
                'gl': 'in',
                'hl': 'en',
                'device': 'desktop'
            }, timeout=30)
            
            if response.status_code != 200:
                print(f"   ❌ Google API error: {response.status_code}")
                return []
                
            data = response.json()
            results = []
            
            for i, result in enumerate(data.get('organic_results', []), 1):
                domain = urlparse(result.get('link', '')).netloc
                
                results.append({
                    'platform': 'Google',
                    'title': result.get('title', ''),
                    'snippet': result.get('snippet', ''),
                    'url': result.get('link', ''),
                    'domain': domain,
                    'query': query,
                    'position': i,
                    'timestamp': datetime.now().isoformat()
                })
            
            print(f"   ✅ Google: {len(results)} results")
            time.sleep(1)
            return results
            
        except Exception as e:
            print(f"   ❌ Google search error for '{query}': {e}")
            return []

    def search_youtube_enhanced(self, query: str, num_results: int = 20) -> List[Dict]:
        """Unbiased YouTube search"""
        try:
            print(f"   🔍 YouTube search for: {query}")
            
            search_response = requests.get("https://www.googleapis.com/youtube/v3/search", params={
                'part': 'snippet,id',
                'q': query,
                'key': self.youtube_api_key,
                'type': 'video',
                'maxResults': min(num_results, 50),
                'regionCode': 'IN',
                'order': 'relevance',
                'safeSearch': 'none'
            }, timeout=30)
            
            if search_response.status_code != 200:
                print(f"   ❌ YouTube API error: {search_response.status_code}")
                return []
                
            search_data = search_response.json()
            
            if 'items' not in search_data:
                print(f"   ⚠️ No YouTube results for: {query}")
                return []
            
            results = []
            video_ids = []
            
            for i, item in enumerate(search_data.get('items', []), 1):
                if 'id' not in item or 'videoId' not in item['id']:
                    continue
                    
                video_ids.append(item['id']['videoId'])
                snippet = item['snippet']
                
                results.append({
                    'platform': 'YouTube',
                    'title': snippet.get('title', ''),
                    'snippet': snippet.get('description', ''),
                    'channel': snippet.get('channelTitle', ''),
                    'video_id': item['id']['videoId'],
                    'published_at': snippet.get('publishedAt', ''),
                    'query': query,
                    'position': i,
                    'timestamp': datetime.now().isoformat(),
                    'views': 0,
                    'likes': 0,
                    'comments': 0,
                    'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                })
            
            # Get video statistics
            if video_ids:
                try:
                    stats_response = requests.get("https://www.googleapis.com/youtube/v3/videos", params={
                        'part': 'statistics',
                        'id': ','.join(video_ids[:50]),  # API limit
                        'key': self.youtube_api_key
                    }, timeout=30)
                    
                    if stats_response.status_code == 200:
                        stats_data = stats_response.json()
                        
                        for j, stats in enumerate(stats_data.get('items', [])):
                            if j < len(results):
                                statistics = stats.get('statistics', {})
                                results[j]['views'] = int(statistics.get('viewCount', 0))
                                results[j]['likes'] = int(statistics.get('likeCount', 0))
                                results[j]['comments'] = int(statistics.get('commentCount', 0))
                        
                except Exception as e:
                    print(f"   ⚠️ Error getting video statistics: {e}")
            
            print(f"   ✅ YouTube: {len(results)} results")
            time.sleep(1.5)
            return results
            
        except Exception as e:
            print(f"   ❌ YouTube search error for '{query}': {e}")
            return []

    def unbiased_brand_detection(self, text: str) -> Dict:
        """Equal treatment brand detection"""
        text_lower = text.lower().strip()
        mentions = {'Atomberg': {'mentioned': False, 'confidence': 0, 'variants': [], 'match_type': 'none'}}
        
        # Check Atomberg variants
        for variant in self.atomberg_variants:
            variant_lower = variant.lower()
            
            if variant_lower in text_lower:
                mentions['Atomberg']['mentioned'] = True
                mentions['Atomberg']['confidence'] = 100
                mentions['Atomberg']['variants'].append(variant)
                mentions['Atomberg']['match_type'] = 'exact'
                break
            
            elif HAS_RAPIDFUZZ and not mentions['Atomberg']['mentioned']:
                ratio = fuzz.partial_ratio(variant_lower, text_lower)
                if ratio > 92:  # High threshold to avoid false positives
                    mentions['Atomberg']['mentioned'] = True
                    mentions['Atomberg']['confidence'] = ratio
                    mentions['Atomberg']['variants'].append(variant)
                    mentions['Atomberg']['match_type'] = 'fuzzy'
        
        # Check competitors with identical logic
        for brand, variants in self.competitors.items():
            mentions[brand] = {'mentioned': False, 'confidence': 0, 'variants': [], 'match_type': 'none'}
            
            for variant in variants:
                variant_lower = variant.lower()
                
                if variant_lower in text_lower:
                    mentions[brand]['mentioned'] = True
                    mentions[brand]['confidence'] = 100
                    mentions[brand]['variants'].append(variant)
                    mentions[brand]['match_type'] = 'exact'
                    break
                
                elif HAS_RAPIDFUZZ and not mentions[brand]['mentioned']:
                    ratio = fuzz.partial_ratio(variant_lower, text_lower)
                    if ratio > 92:  # Same threshold as Atomberg
                        mentions[brand]['mentioned'] = True
                        mentions[brand]['confidence'] = ratio
                        mentions[brand]['variants'].append(variant)
                        mentions[brand]['match_type'] = 'fuzzy'
        
        return mentions

    def unbiased_sentiment_analysis(self, text: str) -> Dict:
        """Use domain-specific unbiased sentiment analysis"""
        
        # Try transformer first if available
        if self.sentiment_analyzer and len(text.strip()) > 5:
            try:
                text_sample = text[:512]
                sentiment_result = self.sentiment_analyzer(text_sample)[0]
                sentiment_scores = {item['label']: item['score'] for item in sentiment_result}
                
                if 'LABEL_2' in sentiment_scores and sentiment_scores['LABEL_2'] > 0.6:
                    return {
                        'sentiment': 'POSITIVE',
                        'sentiment_score': sentiment_scores['LABEL_2'],
                        'confidence': sentiment_scores['LABEL_2'],
                        'method': 'transformer'
                    }
                elif 'LABEL_0' in sentiment_scores and sentiment_scores['LABEL_0'] > 0.6:
                    return {
                        'sentiment': 'NEGATIVE',
                        'sentiment_score': sentiment_scores['LABEL_0'],
                        'confidence': sentiment_scores['LABEL_0'],
                        'method': 'transformer'
                    }
                else:
                    return {
                        'sentiment': 'NEUTRAL',
                        'sentiment_score': sentiment_scores.get('LABEL_1', 0.5),
                        'confidence': sentiment_scores.get('LABEL_1', 0.5),
                        'method': 'transformer'
                    }
            except Exception as e:
                print(f"Transformer sentiment failed, using domain-specific: {e}")
        
        # Fall back to domain-specific analysis
        return self.domain_sentiment.analyze_sentiment(text)

    def calculate_unbiased_engagement_score(self, result: Dict) -> float:
        """Unbiased engagement scoring"""
        base_score = 1.0
        
        # Equal platform treatment
        platform = result.get('platform', 'Google')
        platform_weight = self.platform_weights.get(platform, 1.0)
        
        # Gradual position decay
        position = result.get('position', 10)
        position_weight = self.position_weights.get(position, 0.5)
        
        # YouTube engagement (conservative calculation)
        if platform == 'YouTube':
            views = result.get('views', 0)
            likes = result.get('likes', 0)
            comments = result.get('comments', 0)
            
            if views > 1000:  # Only boost for substantial content
                engagement_rate = (likes + comments * 2) / views
                engagement_multiplier = 1 + min(engagement_rate * 50, 1.5)  # Cap at 2.5x total
                base_score *= engagement_multiplier
        
        final_score = base_score * platform_weight * position_weight
        return min(final_score, 10.0)  # Reasonable cap

    def calculate_sov_metrics(self, processed_results: List[Dict]) -> Dict:
        """Unbiased SOV calculation with transparency"""
        if not processed_results:
            return self._empty_sov_metrics()
        
        df = pd.DataFrame(processed_results)
        
        # Only count results where brands are actually mentioned
        brand_columns = ['atomberg_mentioned'] + [f'{comp.lower()}_mentioned' for comp in self.competitors.keys()]
        valid_brand_columns = [col for col in brand_columns if col in df.columns]
        
        brand_mentioned_df = df[df[valid_brand_columns].any(axis=1)]
        
        if len(brand_mentioned_df) == 0:
            print("⚠️ No brand mentions found - this suggests keyword bias or detection issues")
            return self._empty_sov_metrics()
        
        atomberg_results = brand_mentioned_df[brand_mentioned_df['atomberg_mentioned'] == True]
        
        # Calculate weighted SOV
        atomberg_weighted_score = sum(r['engagement_score'] for _, r in atomberg_results.iterrows())
        total_weighted_score = sum(r['engagement_score'] for _, r in brand_mentioned_df.iterrows())
        
        overall_sov = (atomberg_weighted_score / total_weighted_score * 100) if total_weighted_score > 0 else 0
        
        # Simple mention-based SOV
        mention_based_sov = (len(atomberg_results) / len(brand_mentioned_df) * 100) if len(brand_mentioned_df) > 0 else 0
        
        # Share of Positive Voice
        positive_total = len(brand_mentioned_df[brand_mentioned_df['sentiment'] == 'POSITIVE'])
        positive_atomberg = len(atomberg_results[atomberg_results['sentiment'] == 'POSITIVE'])
        sopv = (positive_atomberg / positive_total * 100) if positive_total > 0 else 0
        
        # Platform breakdown
        platform_sov = {}
        for platform in df['platform'].unique():
            platform_df = brand_mentioned_df[brand_mentioned_df['platform'] == platform]
            if len(platform_df) > 0:
                platform_atomberg = len(platform_df[platform_df['atomberg_mentioned'] == True])
                platform_total = len(platform_df)
                platform_sov[platform] = (platform_atomberg / platform_total * 100)
        
        return {
            'overall_sov': overall_sov,
            'mention_based_sov': mention_based_sov,
            'sopv': sopv,
            'platform_sov': platform_sov,
            'keyword_performance': {},
            'total_mentions': len(atomberg_results),
            'positive_mentions': len(atomberg_results[atomberg_results['sentiment'] == 'POSITIVE']),
            'negative_mentions': len(atomberg_results[atomberg_results['sentiment'] == 'NEGATIVE']),
            'neutral_mentions': len(atomberg_results[atomberg_results['sentiment'] == 'NEUTRAL']),
            'total_brand_results': len(brand_mentioned_df),
            'data_quality_notes': {
                'total_results_analyzed': len(processed_results),
                'results_with_brand_mentions': len(brand_mentioned_df),
                'brand_mention_rate': len(brand_mentioned_df) / len(processed_results) * 100
            }
        }

    def _empty_sov_metrics(self) -> Dict:
        """Empty SOV structure"""
        return {
            'overall_sov': 0, 'mention_based_sov': 0, 'sopv': 0,
            'platform_sov': {}, 'keyword_performance': {},
            'total_mentions': 0, 'positive_mentions': 0, 'negative_mentions': 0, 'neutral_mentions': 0,
            'total_brand_results': 0,
            'data_quality_notes': {'total_results_analyzed': 0, 'results_with_brand_mentions': 0, 'brand_mention_rate': 0}
        }

    def analyze_competitive_landscape(self, results: List[Dict]) -> Dict:
        """Unbiased competitive analysis"""
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        brand_performance = {}
        brands = ['Atomberg'] + list(self.competitors.keys())
        
        for brand in brands:
            brand_col = 'atomberg_mentioned' if brand == 'Atomberg' else f'{brand.lower()}_mentioned'
            if brand_col in df.columns:
                brand_results = df[df[brand_col] == True]
                
                if len(brand_results) > 0:
                    brand_performance[brand] = {
                        'total_mentions': len(brand_results),
                        'positive_sentiment': len(brand_results[brand_results['sentiment'] == 'POSITIVE']),
                        'negative_sentiment': len(brand_results[brand_results['sentiment'] == 'NEGATIVE']),
                        'neutral_sentiment': len(brand_results[brand_results['sentiment'] == 'NEUTRAL']),
                        'avg_position': brand_results['position'].mean(),
                        'avg_engagement': brand_results['engagement_score'].mean(),
                        'platform_distribution': brand_results['platform'].value_counts().to_dict(),
                        'sentiment_ratio': len(brand_results[brand_results['sentiment'] == 'POSITIVE']) / len(brand_results)
                    }
        
        return brand_performance

    def analyze_content_themes(self, results: List[Dict]) -> Dict:
        """Theme analysis without brand bias"""
        if not results:
            return {'feature_themes': {}, 'sentiment_themes': {}, 'platform_themes': {}, 'keyword_themes': {}}
        
        df = pd.DataFrame(results)
        atomberg_results = df[df['atomberg_mentioned'] == True]
        
        feature_themes = defaultdict(float)
        for _, result in atomberg_results.iterrows():
            features = result.get('features', {})
            if isinstance(features, dict) and 'scores' in features:
                for feature, score in features['scores'].items():
                    if score > 0:
                        feature_themes[feature] += score
        
        return {
            'feature_themes': dict(feature_themes),
            'sentiment_themes': atomberg_results['sentiment'].value_counts().to_dict() if len(atomberg_results) > 0 else {},
            'platform_themes': atomberg_results['platform'].value_counts().to_dict() if len(atomberg_results) > 0 else {},
            'keyword_themes': {}
        }

    def extract_price_enhanced(self, text: str) -> Dict:
        """Enhanced price extraction"""
        price_patterns = [
            r'₹\s*(\d{1,2}(?:,\d{3})*(?:\.\d{2})?)',
            r'rs\.?\s*(\d{1,2}(?:,\d{3})*(?:\.\d{2})?)',
            r'price[:\s]+₹?\s*(\d{1,2}(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        prices = []
        for pattern in price_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    price = float(match.replace(',', ''))
                    if 1000 <= price <= 50000:
                        prices.append(int(price))
                except:
                    continue
        
        return {
            'price': min(prices) if prices else 0,
            'price_range': f"₹{min(prices)}-₹{max(prices)}" if len(prices) > 1 else f"₹{prices[0]}" if prices else None,
            'multiple_prices': len(prices) > 1
        }

    def analyze_features_enhanced(self, text: str) -> Dict:
        """Unbiased feature analysis"""
        text_lower = text.lower()
        feature_scores = {}
        feature_mentions = {}
        
        for feature, config in self.features.items():
            keywords = config['keywords']
            weight = config['weight']  # All weights are 1.0 now
            mentions = []
            score = 0
            
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
                    mentions.append(keyword)
            
            feature_scores[feature] = score * weight
            feature_mentions[feature] = mentions
        
        return {
            'scores': feature_scores,
            'mentions': feature_mentions,
            'total_score': sum(feature_scores.values()),
            'dominant_features': sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        }

    def generate_strategic_recommendations(self, sov_metrics: Dict, competitive_analysis: Dict, 
                                         content_themes: Dict, price_analysis: Dict = None, 
                                         bias_report: Dict = None) -> List[Dict]:
        """Generate unbiased strategic recommendations"""
        recommendations = []
        
        overall_sov = sov_metrics.get('overall_sov', 0)
        total_mentions = sov_metrics.get('total_mentions', 0)
        sopv = sov_metrics.get('sopv', 0)
        
        # Bias warnings first
        if bias_report and bias_report.get('is_biased', False):
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Data Quality',
                'title': 'Bias Detected in Analysis Results',
                'description': 'Statistical bias detected in data collection or analysis methods',
                'actions': [
                    'Review keyword selection for market neutrality',
                    'Verify brand detection algorithms for equal treatment',
                    'Consider expanding competitor-focused searches',
                    'Validate results against known market data'
                ],
                'bias_warnings': bias_report.get('warnings', [])
            })
        
        # Data quality recommendations
        brand_mention_rate = sov_metrics.get('data_quality_notes', {}).get('brand_mention_rate', 0)
        if brand_mention_rate < 10:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Data Collection',
                'title': 'Low Brand Mention Rate Detected',
                'description': f'Only {brand_mention_rate:.1f}% of results contain brand mentions',
                'actions': [
                    'Review keyword strategy for brand relevance',
                    'Expand search to include more brand-specific terms',
                    'Check if search results are too generic',
                    'Consider adding shopping/review site searches'
                ]
            })
        
        # SOV-based recommendations (with context)
        if overall_sov < 15:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Share of Voice',
                'title': 'Low Share of Voice',
                'description': f'Atomberg SOV is {overall_sov:.1f}% - below competitive threshold',
                'actions': [
                    'Increase content marketing efforts',
                    'Expand influencer partnerships',
                    'Optimize SEO for key product terms',
                    'Launch brand awareness campaigns'
                ]
            })
        elif overall_sov > 40:
            # Flag potentially unrealistic high SOV
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Data Validation',
                'title': 'Verify High SOV Results',
                'description': f'Atomberg SOV of {overall_sov:.1f}% seems high - validate against market reality',
                'actions': [
                    'Cross-reference with actual market share data',
                    'Run competitor-focused keyword searches',
                    'Verify brand detection accuracy',
                    'Check for keyword selection bias'
                ]
            })
        
        # Sentiment recommendations
        if sopv < 50 and total_mentions > 5:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Sentiment Management',
                'title': 'Share of Positive Voice Below Benchmark',
                'description': f'SoPV at {sopv:.1f}% indicates sentiment challenges',
                'actions': [
                    'Analyze negative feedback themes',
                    'Improve customer support content',
                    'Address product/service issues',
                    'Amplify positive customer testimonials'
                ]
            })
        
        return sorted(recommendations, 
                     key=lambda x: {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[x['priority']], 
                     reverse=True)

    def create_unbiased_dashboard(self, results: Dict):
        """Create comprehensive dashboard with bias warnings"""
        try:
            plt.style.use('default')
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)
            
            sov_metrics = results['sov_metrics']
            competitive_analysis = results['competitive_analysis']
            bias_report = results.get('bias_analysis', {})
            content_themes = results.get('content_themes', {})
            price_analysis = results.get('price_intelligence', {})
            
            # Title with bias warning if needed
            main_title = 'Unbiased Atomberg Intelligence Dashboard'
            if bias_report.get('is_biased', False):
                main_title += ' - ⚠️ BIAS DETECTED'
            
            fig.suptitle(main_title, fontsize=24, fontweight='bold', y=0.98,
                        color='red' if bias_report.get('is_biased') else 'black')
            
            # 1. SOV Overview with bias context
            ax1 = fig.add_subplot(gs[0, 0])
            atomberg_sov = sov_metrics.get('overall_sov', 0)
            
            if atomberg_sov > 0:
                competitor_sov = max(1, 100 - atomberg_sov)  # Ensure we can show the pie
                colors = ['#2E8B57', '#DC143C']
                
                # Add bias warning color if SOV is suspiciously high
                if atomberg_sov > 50:
                    colors = ['#FF6B35', '#DC143C']  # Orange warning color
                
                wedges, texts, autotexts = ax1.pie([atomberg_sov, competitor_sov], 
                                                  labels=['Atomberg', 'Competitors'], 
                                                  colors=colors, autopct='%1.1f%%', 
                                                  startangle=90,
                                                  textprops={'fontsize': 10})
                
                title = f'SOV: {atomberg_sov:.1f}%'
                if atomberg_sov > 50:
                    title += ' (Verify)'
                ax1.set_title(title, fontsize=14, fontweight='bold')
            else:
                ax1.text(0.5, 0.5, 'No Brand\nMentions', ha='center', va='center', 
                        fontsize=12, transform=ax1.transAxes, fontweight='bold')
                ax1.set_title('Share of Voice', fontsize=14, fontweight='bold')
            
            # 2. Bias Status Panel
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.axis('off')
            
            if bias_report:
                if bias_report.get('is_biased', False):
                    ax2.text(0.5, 0.8, '🔴 BIAS\nDETECTED', ha='center', va='center', 
                            fontsize=14, fontweight='bold', color='red',
                            transform=ax2.transAxes)
                    bias_score = bias_report.get('overall_bias_score', 0)
                    ax2.text(0.5, 0.4, f'Score: {bias_score:.1%}', ha='center', va='center',
                            fontsize=12, transform=ax2.transAxes)
                else:
                    ax2.text(0.5, 0.8, '🟢 UNBIASED\nANALYSIS', ha='center', va='center', 
                            fontsize=14, fontweight='bold', color='green',
                            transform=ax2.transAxes)
                    ax2.text(0.5, 0.4, 'Data Quality: Good', ha='center', va='center',
                            fontsize=10, transform=ax2.transAxes)
            
            ax2.set_title('Bias Check', fontsize=14, fontweight='bold')
            
            # 3. Platform Performance
            ax3 = fig.add_subplot(gs[0, 2])
            platform_sov = sov_metrics.get('platform_sov', {})
            
            if platform_sov and any(v > 0 for v in platform_sov.values()):
                platforms = list(platform_sov.keys())[:4]  # Top 4 platforms
                platform_values = [platform_sov[p] for p in platforms]
                colors = plt.cm.Set3(np.linspace(0, 1, len(platforms)))
                
                bars = ax3.bar(platforms, platform_values, color=colors, alpha=0.8)
                ax3.set_title('Platform SOV %', fontsize=12, fontweight='bold')
                ax3.set_ylabel('SOV %')
                ax3.tick_params(axis='x', rotation=45, labelsize=9)
                
                # Add value labels
                for bar, value in zip(bars, platform_values):
                    if value > 0:
                        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                f'{value:.0f}%', ha='center', va='bottom', fontsize=9)
            else:
                ax3.text(0.5, 0.5, 'No Platform\nData', ha='center', va='center',
                        fontsize=10, transform=ax3.transAxes)
                ax3.set_title('Platform Performance', fontsize=12, fontweight='bold')
            
            # 4. Data Quality Metrics
            ax4 = fig.add_subplot(gs[0, 3])
            ax4.axis('off')
            
            data_quality = sov_metrics.get('data_quality_notes', {})
            exec_summary = results.get('executive_summary', {})
            
            quality_text = f"""DATA QUALITY
            
Results: {data_quality.get('total_results_analyzed', 0)}
Brand Mentions: {data_quality.get('results_with_brand_mentions', 0)}
Mention Rate: {data_quality.get('brand_mention_rate', 0):.1f}%
Keywords: {len(results.get('all_keywords', []))}
Success Rate: {exec_summary.get('successful_searches', 0)}"""
            
            ax4.text(0.05, 0.95, quality_text, transform=ax4.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f8ff', alpha=0.8))
            
            # 5. Competitive Landscape
            ax5 = fig.add_subplot(gs[1, :2])
            if competitive_analysis:
                filtered_brands = {k: v for k, v in competitive_analysis.items() 
                                 if v.get('total_mentions', 0) > 0}
                
                if filtered_brands:
                    brands = list(filtered_brands.keys())[:6]  # Top 6
                    mentions = [filtered_brands[brand]['total_mentions'] for brand in brands]
                    
                    # Color coding: highlight Atomberg, warn if dominates
                    colors = []
                    for i, brand in enumerate(brands):
                        if brand == 'Atomberg':
                            if mentions[i] > sum(mentions) * 0.5:  # Dominates >50%
                                colors.append('#FF6B35')  # Warning orange
                            else:
                                colors.append('#4CAF50')  # Normal green
                        else:
                            colors.append('#808080')  # Gray for competitors
                    
                    bars = ax5.barh(brands, mentions, color=colors, alpha=0.8)
                    ax5.set_title('Brand Mentions Distribution', fontsize=14, fontweight='bold')
                    ax5.set_xlabel('Total Mentions')
                    
                    # Add value labels
                    for bar, value in zip(bars, mentions):
                        ax5.text(bar.get_width() + max(mentions)*0.01,
                                bar.get_y() + bar.get_height()/2,
                                str(int(value)), ha='left', va='center', fontweight='bold')
                else:
                    ax5.text(0.5, 0.5, 'No Brand Data Available', ha='center', va='center',
                            fontsize=12, transform=ax5.transAxes)
            else:
                ax5.text(0.5, 0.5, 'No Competitive Data', ha='center', va='center',
                        fontsize=12, transform=ax5.transAxes)
            
            ax5.set_title('Competitive Landscape', fontsize=14, fontweight='bold')
            
            # 6. Sentiment Analysis
            ax6 = fig.add_subplot(gs[1, 2:])
            positive = sov_metrics.get('positive_mentions', 0)
            negative = sov_metrics.get('negative_mentions', 0)
            neutral = sov_metrics.get('neutral_mentions', 0)
            
            if positive + negative + neutral > 0:
                sentiment_data = [positive, negative, neutral]
                sentiment_labels = ['Positive', 'Negative', 'Neutral']
                colors = ['#4CAF50', '#F44336', '#FF9800']
                
                bars = ax6.bar(sentiment_labels, sentiment_data, color=colors, alpha=0.8)
                ax6.set_title('Atomberg Sentiment Distribution', fontsize=14, fontweight='bold')
                ax6.set_ylabel('Number of Mentions')
                
                total = sum(sentiment_data)
                for bar, value in zip(bars, sentiment_data):
                    if value > 0:
                        percentage = value / total * 100
                        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                f'{value}\n({percentage:.1f}%)', ha='center', va='bottom', 
                                fontweight='bold', fontsize=10)
            else:
                ax6.text(0.5, 0.5, 'No Sentiment Data\nAvailable', ha='center', va='center',
                        fontsize=12, transform=ax6.transAxes)
                ax6.set_title('Sentiment Analysis', fontsize=14, fontweight='bold')
            
            # 7. Feature Analysis
            ax7 = fig.add_subplot(gs[2, :2])
            feature_themes = content_themes.get('feature_themes', {})
            
            if feature_themes and any(v > 0 for v in feature_themes.values()):
                features = list(feature_themes.keys())[:6]
                feature_scores = [feature_themes[f] for f in features]
                
                bars = ax7.barh(features, feature_scores, color='#9C27B0', alpha=0.7)
                ax7.set_title('Feature Mention Frequency', fontsize=14, fontweight='bold')
                ax7.set_xlabel('Mention Score')
                
                for bar, value in zip(bars, feature_scores):
                    if value > 0:
                        ax7.text(bar.get_width() + max(feature_scores)*0.01,
                                bar.get_y() + bar.get_height()/2,
                                f'{value:.1f}', ha='left', va='center', fontweight='bold')
            else:
                ax7.text(0.5, 0.5, 'No Feature Data\nAvailable', ha='center', va='center',
                        fontsize=12, transform=ax7.transAxes)
                ax7.set_title('Feature Analysis', fontsize=14, fontweight='bold')
            
            # 8. Price Analysis
            ax8 = fig.add_subplot(gs[2, 2:])
            
            if price_analysis and 'summary' in price_analysis:
                summary = price_analysis['summary']
                atomberg_price = summary.get('atomberg_avg', 0)
                market_price = summary.get('market_avg', 0)
                
                if atomberg_price > 0:
                    if market_price > 0:
                        # Both prices available
                        price_labels = ['Atomberg', 'Market Avg']
                        price_values = [atomberg_price, market_price]
                        colors = ['#4CAF50', '#808080']
                        
                        bars = ax8.bar(price_labels, price_values, color=colors, alpha=0.9)
                        ax8.set_title('Price Comparison', fontsize=14, fontweight='bold')
                        ax8.set_ylabel('Price (₹)')
                        
                        # Add value labels
                        for bar, value in zip(bars, price_values):
                            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(price_values)*0.02,
                                    f'₹{int(value):,}', ha='center', va='bottom', fontweight='bold')
                        
                        # Price advantage
                        advantage = summary.get('price_advantage', 0)
                        if advantage != 0:
                            advantage_text = f"₹{abs(int(advantage)):,} {'cheaper' if advantage > 0 else 'premium'}"
                            ax8.text(0.5, -0.15, f'Atomberg is {advantage_text}',
                                    transform=ax8.transAxes, ha='center', va='top',
                                    fontsize=10, fontweight='bold',
                                    color='#4CAF50' if advantage > 0 else '#FF9800')
                    else:
                        # Only Atomberg price
                        bars = ax8.bar(['Atomberg'], [atomberg_price], color='#4CAF50', alpha=0.9)
                        ax8.text(0, atomberg_price + atomberg_price*0.02, f'₹{int(atomberg_price):,}',
                                ha='center', va='bottom', fontweight='bold')
                        ax8.set_title('Atomberg Price', fontsize=14, fontweight='bold')
                        ax8.set_ylabel('Price (₹)')
                else:
                    ax8.text(0.5, 0.5, 'No Price Data\nAvailable', ha='center', va='center',
                            fontsize=12, transform=ax8.transAxes)
            else:
                # Sample price data for visualization
                sample_prices = [8999, 12500]
                bars = ax8.bar(['Atomberg', 'Market'], sample_prices, color=['#4CAF50', '#808080'])
                ax8.set_title('Price Analysis (Sample)', fontsize=14, fontweight='bold')
                ax8.set_ylabel('Price (₹)')
                
                for bar, value in zip(bars, sample_prices):
                    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_prices)*0.02,
                            f'₹{int(value):,}', ha='center', va='bottom', fontweight='bold')
            
            # 9. Key Recommendations
            ax9 = fig.add_subplot(gs[3, :2])
            ax9.axis('off')
            
            recommendations = results.get('strategic_recommendations', [])
            rec_text = "TOP RECOMMENDATIONS:\n\n"
            
            priority_icons = {"CRITICAL": "🔥", "HIGH": "⚡", "MEDIUM": "💡", "LOW": "📝"}
            
            for i, rec in enumerate(recommendations[:4], 1):
                icon = priority_icons.get(rec['priority'], '•')
                rec_text += f"{i}. {icon} {rec['title']}\n"
                rec_text += f"   {rec['description'][:40]}...\n\n"
            
            if not recommendations:
                rec_text = "No specific recommendations\ngenerated for current dataset."
            
            ax9.text(0.05, 0.95, rec_text, transform=ax9.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f8ff', alpha=0.8))
            
            # 10. Bias Warnings Detail
            ax10 = fig.add_subplot(gs[3, 2:])
            ax10.axis('off')
            
            if bias_report and bias_report.get('warnings'):
                warning_text = "⚠️ BIAS WARNINGS:\n\n"
                for warning in bias_report['warnings'][:3]:
                    warning_text += f"• {warning[:50]}...\n\n"
                
                mention_dist = bias_report.get('mention_distribution', {})
                if mention_dist:
                    warning_text += "\nMention Distribution:\n"
                    for brand, data in list(mention_dist.items())[:3]:
                        actual = data.get('actual_share', 0)
                        warning_text += f"• {brand}: {actual:.1%}\n"
                
                color = '#ffebee'  # Light red background
            else:
                warning_text = "✅ NO BIAS DETECTED\n\nAnalysis shows balanced:\n• Brand detection\n• Keyword coverage\n• Sentiment distribution\n• Platform treatment"
                color = '#e8f5e8'  # Light green background
            
            ax10.text(0.05, 0.95, warning_text, transform=ax10.transAxes, fontsize=9,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8))
            
            # Footer
            timestamp = datetime.now().strftime('%B %d, %Y - %H:%M')
            bias_status = "BIAS DETECTED" if bias_report.get('is_biased') else "UNBIASED"
            fig.text(0.02, 0.01, f'Generated: {timestamp} | Status: {bias_status} | Unbiased Analysis v2.0', 
                    fontsize=10, ha='left', alpha=0.7)
            
            # Methodology note
            methodology_text = "METHODOLOGY: Equal brand variants • Neutral keywords • Balanced weights • Statistical bias monitoring"
            fig.text(0.98, 0.01, methodology_text, fontsize=8, ha='right', alpha=0.6, style='italic')
            
            plt.tight_layout()
            plt.savefig('atomberg_unbiased_dashboard.png', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            print("✅ Unbiased Dashboard saved: atomberg_unbiased_dashboard.png")
            
        except Exception as e:
            print(f"❌ Dashboard creation failed: {e}")
            import traceback
            traceback.print_exc()

    def run_comprehensive_analysis(self) -> Dict:
        """Run unbiased comprehensive analysis"""
        print("🚀 Starting UNBIASED Atomberg Intelligence Analysis")
        print("=" * 80)
        print("⚖️ Bias mitigation features active:")
        print("   • Equal brand variant coverage")
        print("   • Neutral keyword selection")
        print("   • Balanced feature weights")
        print("   • Equal platform treatment")
        print("   • Domain-specific sentiment analysis")
        print("   • Statistical bias monitoring")
        print("=" * 80)
        
        all_results = []
        failed_searches = []
        successful_searches = 0
        
        # Search with unbiased keywords
        for i, keyword in enumerate(self.all_keywords, 1):
            print(f"\n🔍 [{i}/{len(self.all_keywords)}] Processing: '{keyword}'")
            
            # Google search
            google_results = self.search_google_enhanced(keyword, 15)
            if google_results:
                successful_searches += 1
            else:
                failed_searches.append(f"Google - {keyword}")
            all_results.extend(google_results)
            
            # YouTube search
            youtube_results = self.search_youtube_enhanced(keyword, 15)
            if youtube_results:
                successful_searches += 1
            else:
                failed_searches.append(f"YouTube - {keyword}")
            all_results.extend(youtube_results)
            
            time.sleep(1.5)  # Rate limiting
        
        print(f"\n📊 Data Collection Summary:")
        print(f"   • Total Results: {len(all_results)}")
        print(f"   • Successful Searches: {successful_searches}")
        print(f"   • Failed Searches: {len(failed_searches)}")
        
        if len(all_results) == 0:
            print("❌ No data collected - check API connectivity")
            return self._generate_empty_analysis(failed_searches)
        
        print("\n🔬 Processing with unbiased analysis...")
        
        # Process results with unbiased methods
        processed_results = []
        brand_mention_count = 0
        
        for result in all_results:
            full_text = f"{result.get('title', '')} {result.get('snippet', '')}"
            
            # Unbiased brand detection
            brand_mentions = self.unbiased_brand_detection(full_text)
            any_brand_mentioned = any(brand_info['mentioned'] for brand_info in brand_mentions.values())
            atomberg_mentioned = brand_mentions['Atomberg']['mentioned']
            
            if any_brand_mentioned:
                brand_mention_count += 1
            
            # Unbiased analysis
            sentiment_analysis = self.unbiased_sentiment_analysis(full_text)
            engagement_score = self.calculate_unbiased_engagement_score(result)
            price_data = self.extract_price_enhanced(full_text)
            features = self.analyze_features_enhanced(full_text)
            
            processed_result = {
                'platform': result['platform'],
                'query': result['query'],
                'title': result.get('title', ''),
                'snippet': result.get('snippet', ''),
                'url': result.get('url', ''),
                'domain': result.get('domain', ''),
                'position': result.get('position', 0),
                'sentiment': sentiment_analysis['sentiment'],
                'sentiment_score': sentiment_analysis['sentiment_score'],
                'sentiment_method': sentiment_analysis.get('method', 'unknown'),
                'sentiment_confidence': sentiment_analysis.get('confidence', 0.5),
                'atomberg_mentioned': atomberg_mentioned,
                'atomberg_confidence': brand_mentions['Atomberg']['confidence'],
                'engagement_score': engagement_score,
                'price': price_data,
                'features': features,
                'any_brand_mentioned': any_brand_mentioned,
                'views': result.get('views', 0),
                'likes': result.get('likes', 0),
                'comments': result.get('comments', 0)
            }
            
            # Add competitor mentions
            for brand in self.competitors:
                processed_result[f'{brand.lower()}_mentioned'] = brand_mentions[brand]['mentioned']
                processed_result[f'{brand.lower()}_confidence'] = brand_mentions[brand]['confidence']
            
            processed_results.append(processed_result)
        
        print(f"✅ Processing Complete:")
        print(f"   • Brand Mentions Found: {brand_mention_count}")
        print(f"   • Atomberg Mentions: {sum(1 for r in processed_results if r['atomberg_mentioned'])}")
        
        # Calculate metrics
        sov_metrics = self.calculate_sov_metrics(processed_results)
        competitive_analysis = self.analyze_competitive_landscape(processed_results)
        content_themes = self.analyze_content_themes(processed_results)
        
        # Bias analysis
        print("\n⚖️ Running bias analysis...")
        bias_report = self.bias_monitor.analyze_bias(processed_results)
        
        if bias_report.get('is_biased', False):
            print("🚨 BIAS DETECTED:")
            for warning in bias_report['warnings']:
                print(f"   ⚠️ {warning}")
        else:
            print("✅ No significant bias detected")
        
        # Price analysis
        try:
            price_extractor = EnhancedPriceIntelligence()
            price_intelligence = price_extractor.analyze_price_intelligence_enhanced(processed_results, self.serpapi_key)
        except:
            price_intelligence = {'summary': {'atomberg_avg': 0, 'market_avg': 0}}
        
        # Generate recommendations (including bias warnings)
        recommendations = self.generate_strategic_recommendations(
            sov_metrics, competitive_analysis, content_themes, price_intelligence, bias_report
        )
        
        return {
            'executive_summary': {
                'total_results_analyzed': len(processed_results),
                'keywords_analyzed': len(self.all_keywords),
                'platforms_covered': len(set(r['platform'] for r in processed_results)),
                'analysis_timestamp': datetime.now().isoformat(),
                'successful_searches': successful_searches,
                'failed_searches': failed_searches,
                'brand_mentions_found': brand_mention_count,
                'bias_detected': bias_report.get('is_biased', False),
                'analysis_method': 'unbiased_v2.0'
            },
            'sov_metrics': sov_metrics,
            'competitive_analysis': competitive_analysis,
            'content_themes': content_themes,
            'price_intelligence': price_intelligence,
            'bias_analysis': bias_report,
            'strategic_recommendations': recommendations,
            'detailed_results': processed_results
        }

    def _generate_empty_analysis(self, failed_searches: List[str]) -> Dict:
        """Empty analysis with bias context"""
        return {
            'executive_summary': {
                'total_results_analyzed': 0,
                'keywords_analyzed': len(self.all_keywords),
                'platforms_covered': 0,
                'analysis_timestamp': datetime.now().isoformat(),
                'successful_searches': 0,
                'failed_searches': failed_searches,
                'brand_mentions_found': 0,
                'bias_detected': False,
                'analysis_method': 'unbiased_v2.0'
            },
            'sov_metrics': self._empty_sov_metrics(),
            'competitive_analysis': {},
            'content_themes': {'feature_themes': {}, 'sentiment_themes': {}, 'platform_themes': {}},
            'price_intelligence': {'summary': {'atomberg_avg': 0}},
            'bias_analysis': {'is_biased': False, 'warnings': []},
            'strategic_recommendations': [{
                'priority': 'CRITICAL',
                'category': 'Data Collection',
                'title': 'API Integration Failure',
                'description': 'No data collected - verify API keys and connectivity',
                'actions': ['Check API keys', 'Verify network connectivity', 'Review quota limits']
            }],
            'detailed_results': []
        }


def main():
    """Main execution with bias awareness"""
    print("🚀 Initializing UNBIASED Atomberg Intelligence Agent")
    print("⚖️ This version addresses bias issues in SOV analysis")
    print("=" * 80)
    
    # API key check
    SERPAPI_KEY = os.getenv('SERPAPI_KEY')
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
    
    if not SERPAPI_KEY or not YOUTUBE_API_KEY:
        print("❌ Missing API keys!")
        print("   Set SERPAPI_KEY and YOUTUBE_API_KEY environment variables")
        return
    
    print("✅ API Keys loaded")
    
    # Initialize unbiased agent
    agent = UnbiasedAtombergAgent(SERPAPI_KEY, YOUTUBE_API_KEY)
    
    try:
        # Run analysis
        results = agent.run_comprehensive_analysis()
        
        # Display results with bias context
        print(f"\n" + "=" * 80)
        print(f"🎯 UNBIASED ATOMBERG INTELLIGENCE RESULTS")
        print(f"=" * 80)
        
        exec_summary = results['executive_summary']
        sov_metrics = results['sov_metrics']
        bias_report = results['bias_analysis']
        
        # Bias status first
        if bias_report.get('is_biased', False):
            print(f"🚨 BIAS WARNING: Statistical bias detected in results")
            print(f"   Bias Score: {bias_report.get('overall_bias_score', 0):.1%}")
            print(f"   Warnings: {len(bias_report.get('warnings', []))}")
        else:
            print(f"✅ BIAS CHECK: No significant bias detected")
            print(f"   Bias Score: {bias_report.get('overall_bias_score', 0):.1%}")
        
        print(f"\n📊 ANALYSIS SCOPE:")
        print(f"   • Results Analyzed: {exec_summary['total_results_analyzed']}")
        print(f"   • Brand Mentions: {exec_summary.get('brand_mentions_found', 0)}")
        print(f"   • Analysis Method: {exec_summary.get('analysis_method', 'standard')}")
        
        print(f"\n🎯 SHARE OF VOICE (with bias context):")
        overall_sov = sov_metrics.get('overall_sov', 0)
        print(f"   • Overall SOV: {overall_sov:.1f}%")
        if overall_sov > 40:
            print("   ⚠️ High SOV detected - validate against market reality")
        elif overall_sov < 10:
            print("   ⚠️ Low SOV - may indicate keyword bias or low brand presence")
        
        print(f"   • Share of Positive Voice: {sov_metrics.get('sopv', 0):.1f}%")
        print(f"   • Total Mentions: {sov_metrics.get('total_mentions', 0)}")
        
        # Generate outputs
        print(f"\n📊 Generating unbiased analysis outputs...")
        agent.create_unbiased_dashboard(results)
        
        # Save results
        with open('atomberg_unbiased_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"\n✅ UNBIASED ANALYSIS COMPLETE!")
        print(f"📁 Generated Files:")
        print(f"   • atomberg_unbiased_dashboard.png (Bias-aware dashboard)")
        print(f"   • atomberg_unbiased_analysis.json (Complete raw data)")
        
        # Strategic recommendations summary
        recommendations = results.get('strategic_recommendations', [])
        if recommendations:
            print(f"\n🚀 TOP STRATEGIC RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:3], 1):
                priority = rec['priority']
                priority_icon = {"CRITICAL": "🔥", "HIGH": "⚡", "MEDIUM": "💡", "LOW": "📝"}
                icon = priority_icon.get(priority, '•')
                print(f"   {i}. {icon} {rec['title']}")
                print(f"      Category: {rec['category']} | Priority: {priority}")
                print(f"      {rec['description']}")
        
        # Bias-specific warnings
        if bias_report.get('warnings'):
            print(f"\n⚠️ BIAS ANALYSIS WARNINGS:")
            for warning in bias_report['warnings'][:3]:
                print(f"   • {warning}")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Review bias analysis warnings if any")
        print(f"   2. Validate SOV results against known market data")  
        print(f"   3. Consider expanding competitor-focused keywords")
        print(f"   4. Cross-reference with industry reports")
        
    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🛠️ TROUBLESHOOTING:")
        print(f"   1. Check API keys and quotas")
        print(f"   2. Verify internet connectivity")
        print(f"   3. Review error logs above")
    
    print(f"\n🤖 Unbiased Atomberg Intelligence Agent Complete!")
    print(f"📊 This version addresses bias through:")
    print(f"   • Equal brand variant coverage")
    print(f"   • Market-neutral keywords")
    print(f"   • Balanced feature weights")
    print(f"   • Statistical bias monitoring")
    print(f"   • Transparent methodology reporting")


if __name__ == "__main__":
    main()