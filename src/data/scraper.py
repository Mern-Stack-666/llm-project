import requests
from bs4 import BeautifulSoup
import os
import argparse

def scrape_urls(urls, output_file):
    """
    Scrapes a list of URLs and appends content to output_file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        for url in urls:
            try:
                print(f"Scraping {url}...")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text from paragraphs to avoid menu items/footers
                paragraphs = soup.find_all('p')
                text = '\n'.join([p.get_text() for p in paragraphs])
                
                if text:
                    f.write(f"\n\n--- Source: {url} ---\n\n")
                    f.write(text)
                    print(f"Saved {len(text)} characters.")
                else:
                    print(f"No text found for {url}")
                    
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")

if __name__ == "__main__":
    # Comprehensive list of computer science and AI related topics
    default_urls = [
        # AI & ML
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://en.wikipedia.org/wiki/Neural_network",
        "https://en.wikipedia.org/wiki/Large_language_model",
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://en.wikipedia.org/wiki/Reinforcement_learning",
        "https://en.wikipedia.org/wiki/Generative_artificial_intelligence",
        "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
        "https://en.wikipedia.org/wiki/GPT-3",
        "https://en.wikipedia.org/wiki/Common_Objects_in_Context",
        # Computing Fundamentals
        "https://en.wikipedia.org/wiki/Computer_science",
        "https://en.wikipedia.org/wiki/Algorithm",
        "https://en.wikipedia.org/wiki/Data_structure",
        "https://en.wikipedia.org/wiki/Computer_architecture",
        "https://en.wikipedia.org/wiki/Operating_system",
        "https://en.wikipedia.org/wiki/Memory_management",
        "https://en.wikipedia.org/wiki/Parallel_computing",
        # Programming Languages
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://en.wikipedia.org/wiki/C_(programming_language)",
        "https://en.wikipedia.org/wiki/C%2B%2B",
        "https://en.wikipedia.org/wiki/JavaScript",
        "https://en.wikipedia.org/wiki/Compiler",
        "https://en.wikipedia.org/wiki/Just-in-time_compilation",
        # Mathematics for AI
        "https://en.wikipedia.org/wiki/Linear_algebra",
        "https://en.wikipedia.org/wiki/Calculus",
        "https://en.wikipedia.org/wiki/Probability_theory",
        "https://en.wikipedia.org/wiki/Statistics",
        "https://en.wikipedia.org/wiki/Optimization_problem",
        # Internet & Data
        "https://en.wikipedia.org/wiki/World_Wide_Web",
        "https://en.wikipedia.org/wiki/Internet_Protocol_Suite",
        "https://en.wikipedia.org/wiki/Cloud_computing",
        "https://en.wikipedia.org/wiki/Database",
        "https://en.wikipedia.org/wiki/Big_data"
    ]
    
    output_path = os.path.join("data", "raw", "scraped_data.txt")
    scrape_urls(default_urls, output_path)
    print(f"Done! Data saved to {output_path}")
