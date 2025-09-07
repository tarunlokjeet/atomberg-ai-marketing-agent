#!/usr/bin/env python3
"""
Quick script to view Atomberg SoV analysis results with advanced insights
"""

import json
import os
from datetime import datetime

def load_and_display_results():
    """Load and display the analysis results"""
    
    # Check if results file exists
    if not os.path.exists('atomberg_sov_results.json'):
        print("❌ Results file not found. Run main.py first!")
        return
    
    # Load results
    with open('atomberg_sov_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("🎯 ATOMBERG SHARE OF VOICE ANALYSIS RESULTS")
    print("=" * 60)
    
    # Overall metrics
    overall = results.get('overall_metrics', {})
    print(f"\n📊 OVERALL METRICS:")
    print(f"   • Total Results Analyzed: {overall.get('total_results_analyzed', 0)}")
    print(f"   • Atomberg Mentions: {overall.get('atomberg_mentions', 0)}")
    print(f"   • Competitor Mentions: {overall.get('competitor_mentions', 0)}")
    print(f"   • Share of Voice: {overall.get('mention_based_sov', 0):.1f}%")
    print(f"   • Weighted SoV: {overall.get('weighted_sov', 0):.1f}%")
    print(f"   • Share of Positive Voice: {overall.get('share_of_positive_voice', 0):.1f}%")
    
    # Sentiment distribution
    sentiment_dist = overall.get('atomberg_sentiment_distribution', {})
    if sentiment_dist:
        print(f"\n😊 ATOMBERG SENTIMENT BREAKDOWN:")
        for sentiment, percentage in sentiment_dist.items():
            emoji = {"POSITIVE": "🟢", "NEUTRAL": "🟡", "NEGATIVE": "🔴"}.get(sentiment, "⚪")
            print(f"   {emoji} {sentiment}: {percentage*100:.1f}%")
    
    # Keyword performance
    keyword_metrics = results.get('keyword_metrics', {})
    if keyword_metrics:
        print(f"\n🎯 KEYWORD PERFORMANCE:")
        sorted_keywords = sorted(keyword_metrics.items(), 
                               key=lambda x: x[1].get('mention_based_sov', 0), 
                               reverse=True)
        
        for keyword, metrics in sorted_keywords:
            sov = metrics.get('mention_based_sov', 0)
            atomberg_mentions = metrics.get('atomberg_mentions', 0)
            print(f"   • '{keyword}': {sov:.1f}% SoV ({atomberg_mentions} mentions)")
    
    # Insights
    insights = results.get('insights', {})
    
    print(f"\n💡 KEY FINDINGS:")
    for finding in insights.get('key_findings', []):
        print(f"   • {finding}")
    
    print(f"\n🚀 PRIORITY ACTION ITEMS:")
    for i, action in enumerate(insights.get('action_items', []), 1):
        print(f"   {i}. {action}")
    
    # Business Insights
    business_insights = insights.get('business_insights', {})
    if business_insights:
        print(f"\n💼 BUSINESS INTELLIGENCE:")
        
        if 'content_gaps' in business_insights:
            print(f"   📊 Content Gap Analysis:")
            for gap in business_insights['content_gaps']:
                print(f"      • '{gap['keyword']}': {gap['sov']:.1f}% SoV ({gap['opportunity']} opportunity)")
        
        if 'platform_performance' in business_insights:
            print(f"   📱 Platform Performance:")
            for platform, perf in business_insights['platform_performance'].items():
                print(f"      • {platform}: {perf['sov']:.1f}% SoV ({perf['atomberg_mentions']} mentions)")
        
        if 'revenue_impact' in business_insights:
            print(f"   💰 Revenue Impact: {business_insights['revenue_impact']}")
    
    # Advanced Analysis
    if 'theme_analysis' in results:
        theme_data = results['theme_analysis']
        print(f"\n🎭 CONTENT THEME ANALYSIS:")
        
        atomberg_themes = theme_data.get('atomberg_themes', [])[:5]
        competitor_themes = theme_data.get('competitor_themes', [])[:5]
        
        if atomberg_themes:
            print(f"   📝 Atomberg focuses on: {', '.join([t[0] for t in atomberg_themes])}")
        if competitor_themes:
            print(f"   🏢 Competitors focus on: {', '.join([t[0] for t in competitor_themes])}")
        
        content_gaps = theme_data.get('content_gap_analysis', [])
        if content_gaps:
            print(f"   🕳️ Content Gaps (High Priority):")
            for gap in content_gaps[:3]:
                print(f"      • {gap['feature']}: {gap['gap_size']} mention gap ({gap['priority']} priority)")
    
    if 'influencer_analysis' in results:
        influencer_data = results['influencer_analysis']
        print(f"\n🌟 INFLUENCER IMPACT ANALYSIS:")
        
        coverage = influencer_data.get('atomberg_influencer_coverage', 0)
        total_reach = influencer_data.get('total_influencer_reach', 0)
        print(f"   📊 Atomberg Influencer Coverage: {coverage} mentions")
        print(f"   🎯 Total Influencer Reach: {total_reach:,} subscribers")
        
        partnerships = influencer_data.get('partnership_opportunities', [])
        if partnerships:
            print(f"   🤝 Top Partnership Opportunities:")
            for opp in partnerships[:3]:
                print(f"      • {opp['channel'].title()}: {opp['subscribers']:,} subscribers (Score: {opp['influence_score']:.0f})")
    
    if 'price_feature_analysis' in results:
        price_data = results['price_feature_analysis']
        print(f"\n💰 PRICE & FEATURE POSITIONING:")
        
        price_analysis = price_data.get('price_analysis', {})
        atomberg_price = price_analysis.get('atomberg_avg_price', 0)
        competitor_pricing = price_analysis.get('competitor_pricing', {})
        
        if atomberg_price > 0:
            print(f"   💵 Atomberg Average Price: ₹{atomberg_price:,}")
        
        if competitor_pricing:
            print(f"   🏢 Competitor Pricing:")
            for competitor, data in competitor_pricing.items():
                print(f"      • {competitor}: ₹{data['avg_price']:,} (Range: ₹{data['price_range'][0]:,}-₹{data['price_range'][1]:,})")
        
        advantages = price_data.get('competitive_advantages', [])
        improvements = price_data.get('improvement_areas', [])
        
        if advantages:
            print(f"   🏆 Feature Advantages: {', '.join(advantages)}")
        if improvements:
            print(f"   🔧 Improvement Areas: {', '.join(improvements)}")
    
    # Advanced insights from the insights section
    advanced_analysis = insights.get('advanced_analysis', {})
    if advanced_analysis:
        print(f"\n🎯 STRATEGIC INTELLIGENCE:")
        
        pricing_insights = advanced_analysis.get('pricing_insights', {})
        if pricing_insights:
            positioning = pricing_insights.get('positioning', 'Unknown')
            print(f"   💰 Market Positioning: {positioning}")
        
        theme_insights = advanced_analysis.get('theme_insights', {})
        if theme_insights and theme_insights.get('content_gaps'):
            print(f"   📋 Priority Content Gaps: {len(theme_insights['content_gaps'])} identified")
        
        influencer_insights = advanced_analysis.get('influencer_insights', {})
        if influencer_insights:
            coverage_score = influencer_insights.get('coverage_score', 0)
            total_reach = influencer_insights.get('total_reach', 0)
            print(f"   🌟 Influencer Coverage Score: {coverage_score}/10")
            if total_reach > 0:
                print(f"   📺 Potential Reach: {total_reach:,} subscribers")

    print(f"\n📋 RECOMMENDATIONS:")
    for rec in insights.get('recommendations', []):
        print(f"   • {rec}")
    
    # Competitive analysis
    comp_analysis = insights.get('competitive_analysis', {})
    if comp_analysis:
        print(f"\n🏆 COMPETITIVE LANDSCAPE:")
        print(f"   • Top Competitor: {comp_analysis.get('top_competitor', 'N/A')}")
        print(f"   • Their Mentions: {comp_analysis.get('top_competitor_mentions', 0)}")
        print(f"   • Gap to Close: {comp_analysis.get('competitive_gap', 0)} mentions")
    
    # Content opportunities
    content_opps = insights.get('content_opportunities', [])
    if content_opps:
        print(f"\n📝 CONTENT OPPORTUNITIES:")
        for opp in content_opps[:3]:
            print(f"   • {opp}")
    
    # Analysis timestamp
    timestamp = results.get('timestamp', '')
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            print(f"\n⏰ Analysis completed: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            print(f"\n⏰ Analysis completed: {timestamp}")
    
    print(f"\n📈 Files generated:")
    print(f"   📊 Dashboard: atomberg_sov_dashboard.png")
    print(f"   📋 Executive Report: atomberg_executive_report.html")
    print(f"   💾 Detailed Data: atomberg_sov_results.json")

if __name__ == "__main__":
    load_and_display_results()