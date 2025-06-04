import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import torch
import time
from typing import List, Dict
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from tqdm import tqdm

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Game Review Analyzer", layout="wide")

# Cache models and heavy resources
@st.cache_resource
def load_models():
    # Load theme classifier
    theme_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    # Load custom sentiment model
    repo_id = "hdhili/distilbert-sentiment"
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        sentiment_classifier = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        st.error(f"Failed to load custom model: {str(e)} - Falling back to default model")
        sentiment_classifier = pipeline("sentiment-analysis")
    
    return theme_classifier, sentiment_classifier

theme_classifier, sentiment_classifier = load_models()

# ----- Game Search -----
@st.cache_data(ttl=3600)
def search_games_steam_store(query: str, limit: int = 5) -> List[Dict]:
    """Search for games on Steam Store API"""
    try:
        url = f"https://store.steampowered.com/api/storesearch/?term={query}&cc=us&l=en"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return [
            {"name": item["name"], "appid": item["id"]}
            for item in response.json().get("items", [])[:limit]
        ]
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []

# ----- Review Scraping (using direct Steam API) -----
@st.cache_data(ttl=3600)
def scrape_reviews(app_id: str, num_reviews: int = 200) -> List[str]:
    """Scrape reviews directly from Steam API"""
    try:
        url = f"https://store.steampowered.com/appreviews/{app_id}"
        params = {
            'json': 1,
            'filter': 'recent',
            'language': 'english',
            'review_type': 'all',
            'purchase_type': 'all',
            'num_per_page': 100,
            'cursor': '*'
        }
        
        all_reviews = []
        cursor = '*'
        total_fetched = 0
        
        with st.spinner(f"Fetching reviews (0/{num_reviews})..."):
            status_text = st.empty()
            
            while total_fetched < num_reviews:
                params['cursor'] = cursor
                response = requests.get(url, params=params)
                if response.status_code != 200:
                    st.error(f"Request failed with status {response.status_code}")
                    break

                data = response.json()
                reviews = data.get('reviews', [])
                if not reviews:
                    st.warning("No more reviews found.")
                    break

                for review in reviews:
                    all_reviews.append(review['review'])
                    if len(all_reviews) >= num_reviews:
                        break

                total_fetched = len(all_reviews)
                cursor = data.get('cursor')
                status_text.text(f"Fetching reviews ({total_fetched}/{num_reviews})...")
                time.sleep(0.5)

        return all_reviews[:num_reviews]
        
    except Exception as e:
        st.error(f"Review scraping failed: {str(e)}")
        return []

def split_into_quotes(text: str) -> List[str]:
    """Split text into meaningful quotes"""
    connectors = ["but", "however", "although", "still", "yet"]
    for conn in connectors:
        text = text.replace(f" {conn} ", f". {conn} ")
    return [quote.strip() for quote in text.split(".") if quote.strip()]

def analyze_reviews(reviews: List[str], candidate_labels: List[str]) -> pd.DataFrame:
    """Analyze reviews for themes and sentiment"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, review in enumerate(tqdm(reviews, desc="Analyzing reviews")):
        try:
            quotes = split_into_quotes(review)
            for quote in quotes:
                # Theme classification
                theme_out = theme_classifier(quote, candidate_labels)
                top_theme = theme_out["labels"][0]
                theme_score = theme_out["scores"][0]
                
                # Sentiment analysis
                sentiment_out = sentiment_classifier(quote)[0]
                
                results.append({
                    "review_id": i,
                    "quote": quote,
                    "predicted_theme": top_theme,
                    "theme_confidence": round(theme_score, 3),
                    "predicted_sentiment": sentiment_out["label"].lower(),
                    "sentiment_score": round(sentiment_out["score"], 3),
                })
            
            # Update progress
            progress = (i + 1) / len(reviews)
            progress_bar.progress(progress)
            status_text.text(f"Processed {i+1}/{len(reviews)} reviews ({progress:.0%})")
            
        except Exception as e:
            st.warning(f"Skipping review due to error: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    df = pd.DataFrame(results)
    df['predicted_sentiment'] = df['predicted_sentiment'].replace({
    'label_1': 'positive',
    'label_0': 'negative'
    })
    
    return df

def create_visualizations(df: pd.DataFrame):
    """Create and display visualizations with pie charts"""
    if df.empty:
        st.warning("No data to visualize")
        return
    
    st.subheader("📊 Theme Analysis")
    
    # Calculate theme statistics
    theme_counts = df['predicted_theme'].value_counts()
    sentiment_by_theme = df.groupby(['predicted_theme', 'predicted_sentiment']).size().unstack().fillna(0)
    
    # Create two columns for the pie charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Theme distribution pie chart
        st.write("### Theme Distribution")
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.pie(
            theme_counts,
            labels=theme_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.Pastel1.colors
        )
        ax1.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
        st.pyplot(fig1)
    
    with col2:
        # Sentiment distribution pie chart
        st.write("### Overall Sentiment")
        sentiment_counts = df['predicted_sentiment'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=["#72fc65", "#DE6B6B"]  # Green for positive, orange for negative
        )
        ax2.axis('equal')
        st.pyplot(fig2)
    
    # Create donut charts for each theme's sentiment breakdown
    st.write("### Sentiment Breakdown by Theme")
    
    # Get a list of colors for the themes
    theme_colors = plt.cm.tab20.colors
    
    # Create a grid of subplots
    n_themes = len(theme_counts)
    n_cols = 3
    n_rows = (n_themes + n_cols - 1) // n_cols
    
    fig3, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()  # Flatten in case it's a 2D array
    
    for i, (theme, counts) in enumerate(sentiment_by_theme.iterrows()):
        ax = axes[i]
        # Create donut chart
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=["#DE6B6B", "#72fc65"],
            wedgeprops=dict(width=0.7)  # This makes it a donut chart
        )
        
        # Set title with theme name and total count
        ax.set_title(f"{theme}\n({counts.sum()} mentions)", fontsize=10)
        
        # Make the percentages more visible
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(8)
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig3)

def main():
    st.title("🎮 Game Review Theme Analyzer")
    st.markdown("Using custom sentiment model: `hdhili/distilbert-sentiment`")
    
    # Game search
    query = st.text_input("Enter game name:", placeholder="e.g. Cyberpunk 2077")
    
    if query:
        with st.spinner("Searching for games..."):
            games = search_games_steam_store(query)
        
        if games:
            selected_game = st.selectbox(
                "Select the game:",
                games,
                format_func=lambda x: x["name"]
            )
            
            # Analysis parameters
            col1, col2 = st.columns(2)
            with col1:
                num_reviews = st.slider("Number of reviews", 50, 20000, 1000)
            with col2:
                candidate_labels = st.text_input(
                    "Themes to analyze (comma separated):",
                    value="graphics, gameplay, story, performance, price"
                )
                candidate_labels = [lbl.strip() for lbl in candidate_labels.split(",") if lbl.strip()]
            
            if st.button("Analyze Reviews", type="primary"):
                reviews = scrape_reviews(selected_game["appid"], num_reviews)
                
                if reviews:
                    start_time = time.time()
                    df = analyze_reviews(reviews, candidate_labels)
                    elapsed = time.time() - start_time
                    
                    st.success(f"Analyzed {len(df)} quotes from {len(reviews)} reviews in {elapsed:.1f} seconds")
                    
                    # Show results
                    st.dataframe(df.head(100))
                    create_visualizations(df)
                    
                    # Download option
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download full results",
                        csv,
                        f"{selected_game['name']}_review_analysis.csv",
                        "text/csv"
                    )
                else:
                    st.warning("No reviews found or error fetching reviews.")
        else:
            st.warning("No games found matching your search.")

if __name__ == "__main__":
    main()