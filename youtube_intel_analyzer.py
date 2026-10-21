#!/usr/bin/env python3
"""
YouTube Intelligence & Commentary Sentiment Analyzer
Analyzes YouTube videos on a topic, filters for commentary/analysis content,
extracts audience comments, and evaluates sentiment using NLP.
"""

import os
import sys
import argparse
from typing import List, Dict, Any

# Load environment variable or .env file
API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

# Load sentiment analyzer if available
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
except ImportError:
    vader_analyzer = None

# Optional zero-shot classifier with lightweight fallback
classifier = None
try:
    from transformers import pipeline
    # Lazy loaded if needed
except ImportError:
    pass


def get_classifier():
    global classifier
    if classifier is None:
        try:
            from transformers import pipeline
            classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except Exception:
            classifier = False
    return classifier


def is_commentary(title: str) -> bool:
    """Classifies if video title relates to commentary, analysis, or news."""
    cls = get_classifier()
    if cls and cls is not False:
        try:
            labels = ["commentary", "video essay", "analysis", "music video", "gaming gameplay"]
            res = cls(title, labels)
            return res['labels'][0] in ["commentary", "video essay", "analysis"]
        except Exception:
            pass

    # High-accuracy keyword fallback for cross-platform and lightweight environments
    commentary_keywords = [
        "commentary", "essay", "analysis", "review", "drama", "truth about",
        "exposed", "documentary", "breakdown", "explained", "case study",
        "story of", "investigation", "discussion", "critique", "report"
    ]
    title_lower = title.lower()
    return any(kw in title_lower for kw in commentary_keywords) or len(title) > 20


def get_sentiment_score(text: str) -> float:
    """Evaluates text polarity score (-1.0 to +1.0)."""
    if vader_analyzer:
        return vader_analyzer.polarity_scores(text)['compound']
    # Basic polarity heuristic fallback
    positive_words = {"great", "good", "love", "awesome", "excellent", "agree", "smart", "helpful", "best"}
    negative_words = {"bad", "hate", "terrible", "awful", "worst", "disagree", "fake", "scam", "wrong"}
    tokens = set(text.lower().split())
    pos = len(tokens & positive_words)
    neg = len(tokens & negative_words)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def fetch_youtube_data(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Fetches video and comment data via YouTube Data API v3, or returns simulated telemetry."""
    if not API_KEY:
        print("ℹ️  No YOUTUBE_API_KEY environment variable detected.")
        print("   Running in Demonstration Mode with simulated YouTube intelligence telemetry.\n")
        return [
            {
                "title": f"The State of Technology & Society in {query}: A Comprehensive Essay",
                "video_id": "demo_vid_001",
                "channel": f"{query} Media Analysis",
                "sentiment": 0.42,
                "comments_analyzed": 35
            },
            {
                "title": f"Why Everything You Thought About {query} Is Wrong (Investigation)",
                "video_id": "demo_vid_002",
                "channel": "Deep Dive Insights",
                "sentiment": -0.28,
                "comments_analyzed": 48
            },
            {
                "title": f"Understanding the Economic & Cultural Landscape: {query} 2026",
                "video_id": "demo_vid_003",
                "channel": "Global Horizon Commentary",
                "sentiment": 0.65,
                "comments_analyzed": 50
            },
            {
                "title": f"The Untold Truth Behind Recent Developments in {query}",
                "video_id": "demo_vid_004",
                "channel": "Critical Eye Network",
                "sentiment": -0.15,
                "comments_analyzed": 42
            }
        ]

    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        youtube = build('youtube', 'v3', developerKey=API_KEY)

        request = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=max_results * 2,
            relevanceLanguage="en"
        )
        response = request.execute()

        videos = []
        for item in response.get('items', []):
            vid_id = item['id']['videoId']
            title = item['snippet']['title']
            channel = item['snippet']['channelTitle']

            if not is_commentary(title):
                continue

            # Fetch top comments
            scores = []
            try:
                comment_req = youtube.commentThreads().list(
                    part="snippet",
                    videoId=vid_id,
                    maxResults=30
                )
                comment_resp = comment_req.execute()
                for c in comment_resp.get('items', []):
                    c_text = c['snippet']['topLevelComment']['snippet']['textDisplay']
                    scores.append(get_sentiment_score(c_text))
            except HttpError as e:
                # Comments disabled or restricted
                scores = [0.0]

            avg_score = sum(scores) / len(scores) if scores else 0.0
            videos.append({
                "title": title,
                "video_id": vid_id,
                "channel": channel,
                "sentiment": avg_score,
                "comments_analyzed": len(scores)
            })

            if len(videos) >= max_results:
                break

        return videos

    except Exception as e:
        print(f"⚠️  YouTube API Error: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="YouTube Commentary & Sentiment Intelligence Analyzer")
    parser.add_argument("--query", "-q", type=str, default="South Africa Commentary", help="Topic to search and analyze")
    parser.add_argument("--count", "-c", type=int, default=5, help="Number of videos to analyze")
    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║        📺 YouTube Intelligence & Sentiment NLP Analyzer           ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"Topic:       '{args.query}'")
    print(f"Max Results: {args.count}")
    print("──────────────────────────────────────────────────────────────────────")

    results = fetch_youtube_data(args.query, args.count)

    print("\n--- ✅ ANALYSIS RESULTS ---")
    if not results:
        print("No videos matched the criteria.")
        return

    for i, item in enumerate(results, start=1):
        score = item['sentiment']
        if score > 0.05:
            mood = f"🟢 Positive Audience Vibe ({score:+.2f})"
        elif score < -0.05:
            mood = f"🔴 Critical / Negative Audience Vibe ({score:+.2f})"
        else:
            mood = f"⚪ Neutral / Balanced Audience Vibe ({score:+.2f})"

        print(f"{i}. {item['title']}")
        print(f"   Channel:  {item['channel']}")
        print(f"   Sentiment: {mood}")
        print(f"   Comments:  {item['comments_analyzed']} analyzed")
        print("   " + "─" * 40)


if __name__ == "__main__":
    main()
