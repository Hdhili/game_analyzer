import streamlit as st
import pandas as pd
import requests
import torch
import time
from typing import List, Dict
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.express as px
import math

st.set_page_config(page_title="Fast Parallel Game Review Analyzer", layout="wide")

# -------------------------
# Helpers / Utilities
# -------------------------
def split_into_quotes(text: str) -> List[str]:
    connectors = [
        "but", "however", "although", "still", "yet",
        "mais", "cependant", "pourtant", "toutefois"
    ]
    for conn in connectors:
        text = text.replace(f" {conn} ", f". {conn} ")
    return [q.strip() for q in text.split(".") if q.strip()]

@st.cache_data(ttl=3600)
def search_games_steam_store(query: str, limit: int = 8) -> List[Dict]:
    try:
        url = f"https://store.steampowered.com/api/storesearch/?term={query}&cc=us&l=en"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        items = r.json().get("items", [])[:limit]
        return [{"name": it["name"], "appid": it["id"]} for it in items]
    except Exception:
        return []

@st.cache_data(ttl=3600)
def fetch_steam_reviews(app_id: str, num_reviews: int, lang: str, progress_step: int = 10) -> List[str]:
    """
    Fetch reviews from Steam. progress_step controls how often cursor updates are saved into cache,
    but Streamlit progress will be updated outside via callback. Returns list of review texts.
    """
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        "json": 1,
        "filter": "recent",
        "language": "french" if lang == "fr" else "english",
        "review_type": "all",
        "purchase_type": "all",
        "num_per_page": 100,
        "cursor": "*"
    }
    reviews = []
    cursor = "*"
    # Note: This function is cached; the progress UI will be handled by the caller via progress_callback.
    while len(reviews) < num_reviews:
        params["cursor"] = cursor
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            break
        data = r.json()
        batch = data.get("reviews", [])
        if not batch:
            break
        for item in batch:
            reviews.append(item.get("review", ""))
            if len(reviews) >= num_reviews:
                break
        cursor = data.get("cursor", "")
        if not cursor:
            break
        time.sleep(0.08)  # polite pause
    return reviews[:num_reviews]

@st.cache_resource
def load_models(lang_code: str, theme_model_name: str):
    """
    Returns theme_classifier, sentiment_classifier.
    Cached by Streamlit so repeated runs are fast.
    """
    device = 0 if torch.cuda.is_available() else -1

    # Load theme zero-shot model
    try:
        theme_classifier = pipeline("zero-shot-classification", model=theme_model_name, device=device)
    except Exception:
        theme_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

    # Sentiment model selection
    if lang_code == "fr":
        sentiment_repo = "tblard/tf-allocine"
    else:
        sentiment_repo = "hdhili/distilbert-sentiment"

    try:
        tok = AutoTokenizer.from_pretrained(sentiment_repo)
        mdl = AutoModelForSequenceClassification.from_pretrained(sentiment_repo)
        sentiment_classifier = pipeline("sentiment-analysis", model=mdl, tokenizer=tok, device=device)
    except Exception:
        # fallback to default HF sentiment
        sentiment_classifier = pipeline("sentiment-analysis", device=device)

    return theme_classifier, sentiment_classifier

# -------------------------
# UI - Sidebar (controls)
# -------------------------
st.title("🎮 Fast Parallel Game Review Analyzer")

with st.sidebar:
    st.header("Settings")

    # Language
    lang_choice = st.selectbox("Review language", options=["English", "French"])
    lang_code = "fr" if lang_choice.lower() == "french" else "en"

    # Theme model choice
    st.subheader("Theme model (zero-shot)")
    theme_model_choice = st.selectbox(
        "Pick model",
        [
            "facebook/bart-large-mnli (accurate, heavy)",
            "MoritzLaurer/deberta-v3-base-mnli-fever-anli (faster)",
            "joeddav/xlm-roberta-large-xnli (multilingual)"
        ]
    )
    theme_model_map = {
        "facebook/bart-large-mnli (accurate, heavy)": "facebook/bart-large-mnli",
        "MoritzLaurer/deberta-v3-base-mnli-fever-anli (faster)": "MoritzLaurer/deberta-v3-base-mnli-fever-anli",
        "joeddav/xlm-roberta-large-xnli (multilingual)": "joeddav/xlm-roberta-large-xnli",
    }
    theme_model_name = theme_model_map[theme_model_choice]

    # candidate labels default by language (user editable)
    default_labels = (
        "graphismes, jouabilité, histoire, performance, prix"
        if lang_code == "fr"
        else "graphics, gameplay, story, performance, price"
    )
    candidate_labels_input = st.text_input("Themes (comma separated)", value=default_labels)
    candidate_labels = [lbl.strip() for lbl in candidate_labels_input.split(",") if lbl.strip()]

    # performance tuning
    st.markdown("**Performance & batching**")
    device_msg = "GPU" if torch.cuda.is_available() else "CPU"
    st.write(f"Detected device: **{device_msg}**")

    # sensible defaults depending on device
    if torch.cuda.is_available():
        batch_default = 32
        workers_default = 8
    else:
        batch_default = 8
        workers_default = 4

    batch_size = st.number_input("Batch size (quotes per model call)", min_value=1, max_value=256, value=batch_default, step=1)
    max_workers = st.number_input("Thread workers (parallel batches)", min_value=1, max_value=64, value=workers_default, step=1)

    st.markdown("---")
    st.write("Tips:")
    st.write("- Use larger batch size on GPU for speed.")
    st.write("- Decrease batch size on CPU to avoid OOM.")

# -------------------------
# Main inputs
# -------------------------
col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input("Search Steam games by name (press Enter to search):")
    games = []
    if query:
        with st.spinner("Searching Steam store..."):
            games = search_games_steam_store(query)
        if not games:
            st.warning("No games found for that query.")
        else:
            game_choice = st.selectbox("Select a game", options=games, format_func=lambda x: x["name"])
with col2:
    num_reviews = st.number_input("Number of reviews to fetch", min_value=50, max_value=20000, value=1000, step=50)
    analyze_btn = st.button("Analyze Reviews", type="primary")

# -------------------------
# Run analysis when button clicked
# -------------------------
if analyze_btn and query and games:
    # Load models
    with st.spinner("Loading models (cached if already downloaded)..."):
        theme_classifier, sentiment_classifier = load_models(lang_code, theme_model_name)
    st.success(f"Models ready — theme model: {theme_model_name}")

    # Scrape reviews with progress
    scrape_progress = st.progress(0.0)
    scrape_text = st.empty()
    start_scrape = time.time()

    # Because fetch_steam_reviews is cached and doesn't accept a progress callback safely,
    # we'll fetch in a loop here and update progress UI manually (uncached path).
    def fetch_with_progress(app_id: str, n: int, lang: str):
        url = f"https://store.steampowered.com/appreviews/{app_id}"
        params = {
            "json": 1,
            "filter": "recent",
            "language": "french" if lang == "fr" else "english",
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": 100,
            "cursor": "*"
        }
        reviews = []
        cursor = "*"
        while len(reviews) < n:
            params["cursor"] = cursor
            try:
                r = requests.get(url, params=params, timeout=15)
                if r.status_code != 200:
                    break
                data = r.json()
            except Exception:
                break
            batch = data.get("reviews", [])
            if not batch:
                break
            for item in batch:
                reviews.append(item.get("review", ""))
                # update progress every 5 reviews to avoid UI thrashing
                if len(reviews) % 5 == 0 or len(reviews) == n:
                    scrape_progress.progress(min(len(reviews) / n, 1.0))
                    scrape_text.text(f"Fetched {len(reviews)}/{n} reviews...")
                if len(reviews) >= n:
                    break
            cursor = data.get("cursor", "")
            if not cursor:
                break
            time.sleep(0.06)
        scrape_progress.progress(1.0)
        scrape_text.text(f"Fetched {len(reviews)}/{n} reviews (took {time.time()-start_scrape:.1f}s)")
        return reviews[:n]

    reviews = fetch_with_progress(game_choice["appid"], int(num_reviews), lang_code)

    if not reviews:
        st.warning("No reviews fetched. Try a smaller number or check the App ID.")
    else:
        # Extract quotes
        quotes: List[str] = []
        review_ids: List[int] = []
        for i, rv in enumerate(reviews):
            qs = split_into_quotes(rv)
            quotes.extend(qs)
            review_ids.extend([i] * len(qs))

        st.info(f"Extracted {len(quotes)} quotes from {len(reviews)} reviews.")

        # Prepare batches
        total_quotes = len(quotes)
        total_batches = math.ceil(total_quotes / batch_size)
        batches = [quotes[i:i+batch_size] for i in range(0, total_quotes, batch_size)]

        # classification progress UI
        classify_progress = st.progress(0.0)
        classify_text = st.empty()
        start_cls = time.time()

        # Function to process batch: theme + sentiment
        def process_batch(batch_quotes: List[str]):
            # theme predictions
            theme_out = theme_classifier(batch_quotes, candidate_labels)
            if isinstance(theme_out, dict):
                theme_out = [theme_out]
            # sentiment predictions
            sent_out = sentiment_classifier(batch_quotes)
            if isinstance(sent_out, dict):
                sent_out = [sent_out]

            merged = []
            for t, s, q in zip(theme_out, sent_out, batch_quotes):
                merged.append({
                    "quote": q,
                    "predicted_theme": (t.get("labels") or [None])[0],
                    "theme_confidence": round((t.get("scores") or [0])[0], 3),
                    "predicted_sentiment": str(s.get("label", "")).lower(),
                    "sentiment_score": round(float(s.get("score", 0)), 3)
                })
            return merged

        # Run batches in parallel threads
        results = []
        with ThreadPoolExecutor(max_workers=int(max_workers)) as exe:
            future_to_idx = {exe.submit(process_batch, b): idx for idx, b in enumerate(batches)}
            completed = 0
            for future in as_completed(future_to_idx):
                batch_res = future.result()
                results.extend(batch_res)
                completed += 1
                classify_progress.progress(min(completed / total_batches, 1.0))
                classify_text.text(f"Classified {completed}/{total_batches} batches")

        classify_progress.progress(1.0)
        classify_text.text(f"Classification finished in {time.time()-start_cls:.1f}s — {len(results)} quote predictions")

        # Build final DataFrame
        df_out = pd.DataFrame(results)
        df_out["predicted_sentiment"] = df_out["predicted_sentiment"].replace({
            "label_1": "positive",
            "label_0": "negative",
            "positif": "positive",
            "négatif": "negative"
        }).str.lower()

        # -------------------------
        # Filters & Charts
        # -------------------------
        st.subheader("Interactive analysis & filtering")

        # Filters
        unique_themes = sorted(df_out["predicted_theme"].dropna().unique().tolist())
        unique_sents = sorted(df_out["predicted_sentiment"].dropna().unique().tolist())

        left, right = st.columns([3, 1])
        with left:
            theme_filter = st.multiselect("Filter by theme", options=unique_themes, default=unique_themes)
            sentiment_filter = st.multiselect("Filter by sentiment", options=unique_sents, default=unique_sents)
            keyword = st.text_input("Search keyword in quote", value="")
        with right:
            st.write("Results")
            st.metric("Quotes", len(df_out))
            st.metric("Unique themes", len(unique_themes))

        # Apply filters
        filtered = df_out[
            df_out["predicted_theme"].isin(theme_filter) &
            df_out["predicted_sentiment"].isin(sentiment_filter)
        ]
        if keyword:
            filtered = filtered[filtered["quote"].str.contains(keyword, case=False, na=False)]

        st.write(f"Showing {len(filtered)} quotes after filters")

        # Interactive plots using plotly
        fig1 = px.histogram(filtered, x="predicted_sentiment", color="predicted_sentiment",
                            title="Sentiment Distribution", template="plotly_white")
        fig2 = px.histogram(filtered, x="predicted_theme", color="predicted_theme",
                            title="Theme Distribution (filtered)", template="plotly_white", height=450)

        col_a, col_b = st.columns(2)
        col_a.plotly_chart(fig1, use_container_width=True)
        col_b.plotly_chart(fig2, use_container_width=True)

        # Sentiment by theme sunburst for quick insight
        if not filtered.empty:
            sun = px.sunburst(filtered, path=["predicted_theme", "predicted_sentiment"], values=None,
                              title="Theme → Sentiment breakdown (sunburst)")
            st.plotly_chart(sun, use_container_width=True)

        # Data table and download
        st.subheader("Filtered quotes")
        st.dataframe(filtered.head(300))

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", csv, file_name=f"{game_choice['name']}_filtered_quotes.csv", mime="text/csv")

        st.success("Analysis complete ✅")
