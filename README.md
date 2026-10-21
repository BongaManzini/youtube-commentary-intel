# YouTube Intelligence & Sentiment Analyzer

An automated NLP pipeline that queries YouTube for commentary, essays, and news discussions on any topic, extracts audience comments, and performs sentiment analysis using VADER and zero-shot transformer classification.

## Features

- **Topic Scouting**: Discovers videos matching specific queries via YouTube Data API v3.
- **Commentary Classifier**: Distinguishes analysis and commentary from gameplay or music videos.
- **Audience Sentiment Extraction**: Scrapes top audience comments and scores sentiment polarity (-1.0 to +1.0).
- **Cross-Platform**: Works across Windows, macOS, and Linux, with both live API and offline demonstration modes.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API Key (optional - runs in demo mode without key)
cp .env.example .env
export YOUTUBE_API_KEY="your_api_key"

# 3. Run analysis
python youtube_intel_analyzer.py --query "Cybersecurity" --count 5
```
