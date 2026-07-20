from datasets import load_dataset
import pandas as pd

#print("Loading HC3 dataset...")
#dataset_hc3 = load_dataset("Hello-SimpleAI/HC3", name="all", trust_remote_code=True)

#df_hc3 = pd.DataFrame(dataset_hc3['train'])
#print("\n--- HC3 preview ---")
#print(df_hc3.head(3))

print("Loading artem9k dataset...")
ds = load_dataset("artem9k/ai-text-detection-pile")