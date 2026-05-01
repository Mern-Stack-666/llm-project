import requests
from bs4 import BeautifulSoup
import os
import re
import argparse


def clean_text(text):
    """
    Clean scraped text: remove source markers, excessive whitespace, 
    citation brackets, and other noise.
    """
    # Remove source markers like --- Source: URL ---
    text = re.sub(r'---\s*Source:.*?---', '', text)
    # Remove Wikipedia citation brackets like [1], [2], [edit], etc.
    text = re.sub(r'\[[\d\w\s,]+\]', '', text)
    # Remove excessive newlines (3+ → 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove excessive spaces
    text = re.sub(r' {2,}', ' ', text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()


def scrape_urls(urls, output_file):
    """
    Scrapes a list of URLs and appends cleaned content to output_file.
    No source markers are injected — only clean text.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        for url in urls:
            try:
                print(f"Scraping {url}...")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Extract text from paragraphs to avoid menu items/footers
                paragraphs = soup.find_all('p')
                text = '\n'.join([p.get_text() for p in paragraphs])
                
                # Clean the extracted text
                text = clean_text(text)
                
                if text and len(text) > 100:  # Skip very short pages
                    f.write(f"\n\n{text}\n\n")
                    print(f"  Saved {len(text)} characters.")
                else:
                    print(f"  Skipped (too short or empty): {url}")
                    
            except Exception as e:
                print(f"  Failed to scrape {url}: {e}")


def clean_existing_file(filepath):
    """Clean an existing scraped data file by removing source markers and noise."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    print(f"Cleaning existing file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    text = clean_text(text)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"  Cleaned. Final size: {len(text):,} characters.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web Scraper for LLM training data")
    parser.add_argument("--clean-only", action="store_true",
                        help="Only clean existing scraped data, don't scrape new data")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    args = parser.parse_args()
    
    output_path = args.output or os.path.join("data", "raw", "scraped_data.txt")
    
    if args.clean_only:
        clean_existing_file(output_path)
    else:
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
        
        scrape_urls(default_urls, output_path)
        print(f"Done! Data saved to {output_path}")
