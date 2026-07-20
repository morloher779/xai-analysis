import os
import random
import time
import pandas as pd
import ollama

SYSTEM_PROMPT = """You are a local community reporter writing for a hyper-local neighborhood app. 
Your tone is informative, grounded, and straightforward. Avoid overly flowery language, complex metaphors, 
or typical AI filler words (like 'moreover', 'delve', 'tapestry')."""

def generate_local_text(model_name, headline, temp):
    """Generate text via the local Ollama instance on your GPU."""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Write a short local news update (approx. 150 words) based entirely on this headline: '{headline}'. Focus on typical local community details."}
            ],
            options={
                "temperature": temp
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Error with model {model_name}: {e}")
        return None

def main():
    input_json = "News_Category_Dataset_v3.json" 
    output_csv = "ai_generated_local_news.csv"
    
    processed_headlines = set()
    if os.path.exists(output_csv):
        try:
            df_existing = pd.read_csv(output_csv)
            if 'original_headline' in df_existing.columns:
                processed_headlines = set(df_existing['original_headline'].tolist())
                print(f"Found history: {len(processed_headlines)} headlines already processed.")
        except Exception as e:
            print(f"Could not read existing CSV: {e}")

    print(f"Loading dataset from {input_json}...")
    try:
        df_kaggle = pd.read_json(input_json, lines=True)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    df_kaggle = df_kaggle[df_kaggle['headline'].str.strip() != '']
    
    df_kaggle = df_kaggle[~df_kaggle['headline'].isin(processed_headlines)]
    print(f"Remaining fresh headlines in dataset: {len(df_kaggle)}")
    
    LIMIT = 5
    
    headlines = df_kaggle['headline'].head(LIMIT).tolist()

    if len(headlines) == 0:
        print("No new headlines left! You have processed the entire dataset.")
        return

    models = ["llama3", "mistral", "gemma"]
    
    print(f"Starting local GPU generation for {len(headlines)} NEW headlines...")

    for i, headline in enumerate(headlines):
        print(f"\n--- [{i+1}/{len(headlines)}] Processing headline: {headline[:40]}... ---")
        
        for model_name in models:
            temp = round(random.uniform(0.4, 0.9), 1)
            print(f"  -> Generating with {model_name} (Temp: {temp})...", end="", flush=True)
            
            start_time = time.time()
            generated_text = generate_local_text(model_name, headline, temp)
            
            if generated_text:
                calc_time = round(time.time() - start_time, 2)
                print(f" Fertig in {calc_time}s.")
                
                row_data = {
                    "original_headline": headline,
                    "ai_generated_text": generated_text,
                    "model_used": model_name,
                    "temperature": temp,
                    "label": 1 # 1 = AI generated
                }
                
                df_temp = pd.DataFrame([row_data])
                df_temp.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
                
            else:
                print(f" ERROR/Skipped.")

    print(f"Run complete! Your data has been appended to {output_csv}.")

if __name__ == "__main__":
    main()