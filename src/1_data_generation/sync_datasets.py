import pandas as pd

def main():
    df_human = pd.read_csv("data/processed/FINAL_local_news_validation.csv") 
    df_human = df_human[df_human['label'] == 0]
    
    df_ai = pd.read_csv("data/processed/ai_generated_local_news.csv")
    df_ai = df_ai.rename(columns={"ai_generated_text": "text"})
    
    successful_headlines = df_human['original_headline'].unique()
    print(f"{len(successful_headlines)} erfolgreiche menschliche Headlines gefunden.")
    
    df_ai_synced = df_ai[df_ai['original_headline'].isin(successful_headlines)].copy()
    
    cols = ['original_headline', 'text', 'model_used', 'temperature', 'label']
    df_ai_synced = df_ai_synced[cols]
    df_human = df_human[cols]
    
    print(f"Füge {len(df_human)} Menschen-Texte und {len(df_ai_synced)} KI-Texte zusammen...")
    df_final_synced = pd.concat([df_human, df_ai_synced], ignore_index=True)
    df_final_synced = df_final_synced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_file = "data/processed/OOD_Test_Dataset_Synced.csv"
    df_final_synced.to_csv(output_file, index=False)
    print(f"Fertig! Dein finaler, synchronisierter OOD-Test-Datensatz liegt in '{output_file}'.")

if __name__ == "__main__":
    main()