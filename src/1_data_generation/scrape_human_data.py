import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import re

def smart_truncate(text, target_words=150):
    if not isinstance(text, str):
        return text
        
    words = text.split()
    if len(words) <= target_words:
        return text
        
    base_text = " ".join(words[:target_words])
    remainder = " ".join(words[target_words:])
    match = re.search(r'[.!?]', remainder)
    if match:
        end_idx = match.end()
        return base_text + " " + remainder[:end_idx]
    else:
        return base_text + "."


def scrape_huffpost_article(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = " ".join([p.get_text().strip() for p in paragraphs])
        if len(article_text.split()) < 50:
            return None
            
        return smart_truncate(article_text, target_words=150)
        
    except Exception as e:
        print(f"Fehler beim Scrapen von {url}: {e}")
        return None

def main():
    print("Lade generierte KI-Texte...")
    df_ai = pd.read_csv("data/processed/ai_generated_local_news.csv")
    df_ai = df_ai.rename(columns={"ai_generated_text": "text"})
    used_headlines = df_ai['original_headline'].unique()
    
    print("Loading original Kaggle data (JSON)...")
    df_kaggle = pd.read_json("data/raw/News_Category_Dataset_v3.json", lines=True)
    df_human_links = df_kaggle[df_kaggle['headline'].isin(used_headlines)].copy()
    
    scraped_data = []
    
    print(f"Starting web scraping for {len(df_human_links)} original articles...")
    
    for i, row in df_human_links.iterrows():
        url = row['link']
        headline = row['headline']
        
        print(f"[{len(scraped_data)+1}/{len(df_human_links)}] Scraping: {headline[:40]}...")
        
        text = scrape_huffpost_article(url)
        
        if text:
            scraped_data.append({
                "original_headline": headline,
                "text": text,
                "model_used": "human",
                "temperature": 0.0,
                "label": 0
            })
        else:
            print("   -> Skipped (dead link or paywall/unreadable format)")
            
        time.sleep(1)

    df_human_final = pd.DataFrame(scraped_data)
    print(f"\nSuccessfully extracted {len(df_human_final)} real articles with clean sentence endings!")

    cols = ['original_headline', 'text', 'model_used', 'temperature', 'label']
    df_human_final = df_human_final[cols]
    df_ai = df_ai[cols]
    
    print("Mixing AI texts and human texts...")
    df_final = pd.concat([df_human_final, df_ai], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_file = "FINAL_local_news_validation.csv"
    df_final.to_csv(output_file, index=False)
    print(f"Done! Your final length-balanced dataset is saved at '{output_file}'.")

if __name__ == "__main__":
    main()