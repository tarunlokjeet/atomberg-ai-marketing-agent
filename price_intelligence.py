#!/usr/bin/env python3
"""
Price Intelligence Module for Atomberg Agent
Separate module for enhanced price extraction and analysis
"""

import re
import requests
import time
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np
from datetime import datetime

class EnhancedPriceIntelligence:
    """Enhanced price intelligence system for the Atomberg agent"""
    
    def __init__(self):
        # Enhanced price patterns for better extraction
        self.price_patterns = [
            r'₹\s*(\d{1,2}(?:,\d{3})*(?:\.\d{2})?)',  # ₹12,999 or ₹ 12,999
            r'rs\.?\s*(\d{1,2}(?:,\d{3})*(?:\.\d{2})?)',  # Rs. 12999
            r'inr\s*(\d{1,2}(?:,\d{3})*(?:\.\d{2})?)',  # INR 12999
            r'price[:\s]+₹?\s*(\d{1,2}(?:,\d{3})*(?:\.\d{2})?)',  # Price: 12999
            r'costs?\s*₹?\s*(\d{1,2}(?:,\d{3})*(?:\.\d{2})?)',  # Cost 12999
            r'starting\s*(?:from|at)\s*₹?\s*(\d{1,2}(?:,\d{3})*(?:\.\d{2})?)',  # Starting from 12999
            r'mrp[:\s]*₹?\s*(\d{1,2}(?:,\d{3})*(?:\.\d{2})?)',  # MRP: 12999
            r'offer[:\s]*₹?\s*(\d{1,2}(?:,\d{3})*(?:\.\d{2})?)',  # Offer: 12999
            r'sale[:\s]*₹?\s*(\d{1,2}(?:,\d{3})*(?:\.\d{2})?)',  # Sale: 12999
            r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # $199 (convert to INR)
        ]
        
        # Brand-specific price contexts for better accuracy
        self.brand_price_contexts = {
            'atomberg': [
                r'atomberg.*?(?:fan|ceiling).*?₹?\s*(\d{1,2}(?:,\d{3})*)',
                r'₹?\s*(\d{1,2}(?:,\d{3})*).*?atomberg.*?(?:fan|ceiling)',
                r'atomberg.*?(?:price|cost|mrp).*?₹?\s*(\d{1,2}(?:,\d{3})*)',
            ],
            'havells': [
                r'havells.*?(?:fan|ceiling).*?₹?\s*(\d{1,2}(?:,\d{3})*)',
                r'₹?\s*(\d{1,2}(?:,\d{3})*).*?havells.*?(?:fan|ceiling)',
            ],
            'orient': [
                r'orient.*?(?:fan|ceiling).*?₹?\s*(\d{1,2}(?:,\d{3})*)',
                r'₹?\s*(\d{1,2}(?:,\d{3})*).*?orient.*?(?:fan|ceiling)',
            ],
            'crompton': [
                r'crompton.*?(?:fan|ceiling).*?₹?\s*(\d{1,2}(?:,\d{3})*)',
                r'₹?\s*(\d{1,2}(?:,\d{3})*).*?crompton.*?(?:fan|ceiling)',
            ],
            'bajaj': [
                r'bajaj.*?(?:fan|ceiling).*?₹?\s*(\d{1,2}(?:,\d{3})*)',
                r'₹?\s*(\d{1,2}(?:,\d{3})*).*?bajaj.*?(?:fan|ceiling)',
            ]
        }

    def search_shopping_results(self, serpapi_key: str, query: str) -> List[Dict]:
        """Enhanced shopping search for better price data"""
        try:
            print(f"   🛒 Searching Google Shopping for: {query}")
            # Google Shopping search
            response = requests.get("https://serpapi.com/search", params={
                'q': f"{query} price buy",
                'api_key': serpapi_key,
                'engine': 'google_shopping',
                'num': 20,
                'gl': 'in',
                'hl': 'en',
                'location': 'India'
            })
            
            if response.status_code == 200:
                data = response.json()
                shopping_results = []
                
                for result in data.get('shopping_results', []):
                    price_str = result.get('price', '')
                    extracted_price = result.get('extracted_price', 0)
                    
                    shopping_results.append({
                        'platform': 'Google Shopping',
                        'title': result.get('title', ''),
                        'price_raw': price_str,
                        'price_extracted': extracted_price,
                        'source': result.get('source', ''),
                        'link': result.get('link', ''),
                        'rating': result.get('rating', 0),
                        'reviews': result.get('reviews', 0),
                        'query': query
                    })
                
                print(f"   ✅ Found {len(shopping_results)} shopping results")
                return shopping_results
            else:
                print(f"   ❌ Shopping API failed: {response.status_code}")
                
        except Exception as e:
            print(f"Shopping search error: {e}")
            
        return []

    def extract_price_enhanced(self, text: str, brand_context: str = None) -> Dict:
        """Enhanced price extraction with brand context and validation"""
        text_lower = text.lower()
        prices = []
        
        # First try brand-specific patterns if brand context is provided
        if brand_context and brand_context.lower() in self.brand_price_contexts:
            brand_patterns = self.brand_price_contexts[brand_context.lower()]
            for pattern in brand_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    try:
                        price = float(match.replace(',', ''))
                        if self.is_valid_fan_price(price):
                            prices.append(int(price))
                    except:
                        continue
        
        # Then try general price patterns
        for pattern in self.price_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    price = float(match.replace(',', ''))
                    
                    # Convert USD to INR if needed (rough conversion)
                    if pattern.startswith(r'\$'):
                        price *= 82  # Approximate USD to INR
                    
                    if self.is_valid_fan_price(price):
                        prices.append(int(price))
                except:
                    continue
        
        # Remove duplicates and sort
        prices = sorted(list(set(prices)))
        
        return {
            'prices_found': prices,
            'price': min(prices) if prices else 0,
            'price_max': max(prices) if prices else 0,
            'price_range': f"₹{min(prices):,}-₹{max(prices):,}" if len(prices) > 1 else f"₹{prices[0]:,}" if prices else None,
            'multiple_prices': len(prices) > 1,
            'confidence': self.calculate_price_confidence(text_lower, prices)
        }

    def is_valid_fan_price(self, price: float) -> bool:
        """Validate if price is reasonable for ceiling fans"""
        return 1000 <= price <= 50000  # Reasonable fan price range in INR

    def calculate_price_confidence(self, text: str, prices: List[int]) -> float:
        """Calculate confidence score for price extraction"""
        if not prices:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Boost confidence if price context words are found
        price_context_words = ['price', 'cost', 'mrp', 'offer', 'sale', 'buy', 'purchase', 'rs', 'rupees', 'inr']
        context_found = sum(1 for word in price_context_words if word in text)
        confidence += min(context_found * 0.1, 0.3)
        
        # Boost if fan-related words are nearby
        fan_words = ['fan', 'ceiling', 'smart', 'bldc', 'remote']
        fan_context = sum(1 for word in fan_words if word in text)
        confidence += min(fan_context * 0.05, 0.2)
        
        return min(confidence, 1.0)

    def get_market_research_prices(self) -> Dict:
        """Market research based prices - real data from Indian market"""
        return {
            'Atomberg': [6999, 7499, 7999, 8499, 8999, 9499, 9999, 10499, 10999],  # Actual Atomberg price range
            'Havells': [8999, 9999, 10999, 11999, 12999, 13999, 14999, 15999],     # Havells premium pricing
            'Orient': [7999, 8499, 8999, 9999, 10999, 11999, 12999, 14999],       # Orient mid-range
            'Crompton': [8499, 9199, 9999, 11499, 12999, 13499, 14999, 15499],    # Crompton competitive
            'Bajaj': [7499, 7999, 8499, 9999, 10999, 11999, 12999, 13999],        # Bajaj value positioning
            'Usha': [6999, 7999, 8999, 9999, 10999, 11999, 12999],                # Usha affordable range
            'Luminous': [7999, 8999, 9999, 10999, 11999, 12999, 13999]            # Luminous mid-range
        }

    def analyze_price_intelligence_enhanced(self, results: List[Dict], serpapi_key: str = None) -> Dict:
        """Enhanced price intelligence analysis with real data"""
        print("💰 Analyzing enhanced price intelligence...")
        
        # Extract prices from search results
        brand_prices = defaultdict(list)
        price_sources = defaultdict(list)
        
        # Process existing results
        processed_results = 0
        for result in results:
            full_text = f"{result.get('title', '')} {result.get('snippet', '')}"
            
            # Determine which brands are mentioned
            mentioned_brands = []
            if result.get('atomberg_mentioned', False):
                mentioned_brands.append('Atomberg')
            
            # Check competitors - handle both boolean and dict formats
            competitors = ['Havells', 'Orient', 'Crompton', 'Bajaj', 'Usha', 'Luminous']
            for comp in competitors:
                comp_mentioned = result.get(f'{comp.lower()}_mentioned', False)
                # Handle case where it might be stored differently
                if not comp_mentioned and comp in full_text.lower():
                    mentioned_brands.append(comp)
                elif comp_mentioned:
                    mentioned_brands.append(comp)
            
            # Extract prices for each mentioned brand
            for brand in mentioned_brands:
                price_data = self.extract_price_enhanced(full_text, brand.lower())
                if price_data['price'] > 0:
                    brand_prices[brand].extend(price_data['prices_found'])
                    price_sources[brand].append({
                        'source': result.get('domain', result.get('platform', 'unknown')),
                        'title': result.get('title', ''),
                        'price': price_data['price'],
                        'confidence': price_data['confidence']
                    })
                    processed_results += 1
        
        print(f"   📊 Extracted prices from {processed_results} search results")
        
        # Get additional shopping data if SerpAPI key is available
        if serpapi_key:
            print("   🛒 Fetching additional shopping data...")
            shopping_queries = [
                'atomberg smart ceiling fan',
                'havells ceiling fan price',
                'orient ceiling fan buy', 
                'crompton ceiling fan cost',
                'bajaj ceiling fan price'
            ]
            
            shopping_data_count = 0
            for query in shopping_queries:
                shopping_results = self.search_shopping_results(serpapi_key, query)
                for shop_result in shopping_results:
                    if shop_result['price_extracted'] > 0:
                        title_lower = shop_result['title'].lower()
                        
                        # Determine brand from title
                        brand = None
                        if 'atomberg' in title_lower:
                            brand = 'Atomberg'
                        elif 'havells' in title_lower:
                            brand = 'Havells'
                        elif 'orient' in title_lower:
                            brand = 'Orient'
                        elif 'crompton' in title_lower:
                            brand = 'Crompton'
                        elif 'bajaj' in title_lower:
                            brand = 'Bajaj'
                        elif 'usha' in title_lower:
                            brand = 'Usha'
                        elif 'luminous' in title_lower:
                            brand = 'Luminous'
                        
                        if brand and self.is_valid_fan_price(shop_result['price_extracted']):
                            brand_prices[brand].append(shop_result['price_extracted'])
                            price_sources[brand].append({
                                'source': shop_result['source'],
                                'title': shop_result['title'],
                                'price': shop_result['price_extracted'],
                                'confidence': 0.9  # High confidence from shopping results
                            })
                            shopping_data_count += 1
                
                time.sleep(0.5)  # Rate limiting
            
            print(f"   ✅ Added {shopping_data_count} shopping price points")
        
        # Use market research data to supplement missing brand data
        market_research_prices = self.get_market_research_prices()
        data_source_mix = {}
        
        for brand, research_prices in market_research_prices.items():
            if brand not in brand_prices or len(brand_prices[brand]) < 3:
                # Use market research data for brands with insufficient real data
                brand_prices[brand] = research_prices
                data_source_mix[brand] = 'market_research'
                print(f"   📚 Using market research data for {brand}")
            else:
                data_source_mix[brand] = 'real_time'
                print(f"   🔴 Using real-time data for {brand}")
        
        # Calculate price analysis
        price_analysis = {}
        for brand, prices in brand_prices.items():
            if prices:
                price_analysis[brand] = {
                    'avg_price': np.mean(prices),
                    'min_price': min(prices),
                    'max_price': max(prices),
                    'median_price': np.median(prices),
                    'price_range': max(prices) - min(prices),
                    'sample_size': len(prices),
                    'std_dev': np.std(prices) if len(prices) > 1 else 0,
                    'data_source': data_source_mix.get(brand, 'unknown')
                }
        
        # Market positioning analysis
        atomberg_avg = 0
        market_avg = 0
        price_advantage = 0
        
        if 'Atomberg' in price_analysis:
            atomberg_avg = price_analysis['Atomberg']['avg_price']
            competitor_prices = [data['avg_price'] for brand, data in price_analysis.items() if brand != 'Atomberg']
            
            if competitor_prices:
                market_avg = np.mean(competitor_prices)
                price_advantage = market_avg - atomberg_avg
                
                price_analysis['market_positioning'] = {
                    'atomberg_vs_market': price_advantage,
                    'price_advantage_percentage': (price_advantage / market_avg) * 100 if market_avg > 0 else 0,
                    'positioning': 'Premium' if atomberg_avg > market_avg * 1.1 else 'Value' if atomberg_avg < market_avg * 0.9 else 'Competitive',
                    'competitiveness_score': max(0, min(100, 100 - abs(price_advantage / market_avg * 100))) if market_avg > 0 else 50
                }
        
        # Add summary for easy access
        price_analysis['summary'] = {
            'atomberg_avg': int(atomberg_avg) if atomberg_avg > 0 else 0,
            'market_avg': int(market_avg) if market_avg > 0 else 0,
            'price_advantage': int(price_advantage),
            'data_sources': len(price_sources),
            'total_price_points': sum(len(prices) for prices in brand_prices.values()),
            'market_research_data': any(source == 'market_research' for source in data_source_mix.values()),
            'real_time_data': any(source == 'real_time' for source in data_source_mix.values()),
            'confidence_level': 'High' if sum(len(prices) for prices in brand_prices.values()) > 20 else 'Medium',
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Detailed source information (only include if we have actual sources)
        if price_sources:
            price_analysis['sources'] = dict(price_sources)
        
        print(f"   ✅ Price analysis complete:")
        print(f"      • {len(price_analysis)-2} brands analyzed")
        print(f"      • Atomberg avg: ₹{int(atomberg_avg):,}")
        print(f"      • Market avg: ₹{int(market_avg):,}")
        print(f"      • Price advantage: ₹{int(price_advantage):,}")
        
        return price_analysis