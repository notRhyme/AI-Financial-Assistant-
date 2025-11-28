# main.py
import os
from dotenv import load_dotenv

# 1. Load environment variables (The API Key)
load_dotenv()

# 2. Check Key
if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY not found. Please check your .env file.")
    exit()

# 3. Import your pipeline
# This works because main.py is in the root, so it can see the 'src' folder
from src.pipeline import run_pipeline

if __name__ == "__main__":
    print("## 🏦 Initializing Banking Agent System ##")
    
    # 4. Define test question
    question = "What does the document say about interest rate risks?"
    
    print(f"\n❓ Question: {question}")
    print("-" * 50)
    
    # 5. Run the Agents
    try:
        result = run_pipeline(question)
        print("\n\n✅ FINAL RESULT:\n")
        print(result)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")