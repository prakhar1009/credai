# news1.py
import os
import time
import sys
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
import requests
from datetime import datetime
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()

# Configure API keys
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# Suppress litellm verbose logging
os.environ["LITELLM_LOG"] = "ERROR"
import litellm
litellm.set_verbose = False

# Rate limiting configuration
class RateLimiter:
    def __init__(self, requests_per_minute=8):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        
    def wait_if_needed(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            print(f"⏳ Rate limiting: Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()

# Global rate limiter - 8 requests per minute (safe for free tier)
rate_limiter = RateLimiter(requests_per_minute=8)

# Initialize the Gemini LLM
llm = LLM(
    model="gemini/gemini-1.5-flash",  # Using stable model
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.5,
    max_retries=3,
    timeout=60,
    max_rpm=8  # Rate limit per minute
)

# Hook into LLM's call method to add rate limiting
original_call = llm.call

def rate_limited_call(*args, **kwargs):
    rate_limiter.wait_if_needed()
    try:
        return original_call(*args, **kwargs)
    except Exception as e:
        if "429" in str(e) or "rate limit" in str(e).lower():
            print("⚠️  Rate limit hit! Waiting 60 seconds...")
            time.sleep(60)
            rate_limiter.wait_if_needed()
            return original_call(*args, **kwargs)
        raise

llm.call = rate_limited_call

# Initialize search tool
search_tool = SerperDevTool()

# Define Agents with minimal configuration
news_analyst = Agent(
    role="News Analyst Pro",
    goal="Summarizes the article into clear, concise points",
    backstory="Expert news analyst who distills complex information",
    llm=llm,
    verbose=True,
    memory=False,
    allow_delegation=False,
    max_iter=1,
    max_rpm=8
)

cred_checker = Agent(
    role="CredChecker AI",
    goal="Assesses credibility and bias",
    backstory="Impartial fact-checker analyzing reliability",
    llm=llm,
    verbose=True,
    memory=False,
    tools=[search_tool],
    allow_delegation=False,
    max_iter=1,
    max_rpm=8
)

source_hunter = Agent(
    role="Source Hunter",
    goal="Validates factual claims",
    backstory="Meticulous researcher who verifies information",
    llm=llm,
    tools=[search_tool],
    verbose=True,
    memory=False,
    allow_delegation=False,
    max_iter=1,
    max_rpm=8
)

bias_detector = Agent(
    role="Bias Detection Engine",
    goal="Identifies linguistic bias",
    backstory="Expert in media analysis",
    llm=llm,
    verbose=True,
    memory=False,
    allow_delegation=False,
    max_iter=1,
    max_rpm=8
)

# Simplified tasks
analysis_task = Task(
    description="Create 5 key bullet points summarizing the main facts of the article. Be concise.",
    expected_output="5 bullet points",
    agent=news_analyst
)

credibility_task = Task(
    description="Rate credibility 0-100 with 2-3 sentence explanation.",
    expected_output="Score and brief reasoning",
    agent=cred_checker
)

source_validation_task = Task(
    description="List top 3 claims with verification status (verified/unverified/disputed).",
    expected_output="3 claims with status",
    agent=source_hunter
)

bias_detection_task = Task(
    description="Rate bias 0-100 (0=neutral) with 2 examples if biased.",
    expected_output="Bias score and examples",
    agent=bias_detector
)

def fetch_url(url: str) -> str:
    """Fetch content from URL"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 403:
            print("❌ Access forbidden. The website blocks automated access.")
            print("💡 Try copying and pasting the article text instead.")
        else:
            print(f"❌ HTTP error: {e}")
        return None
    except Exception as e:
        print(f"❌ Failed to fetch URL: {e}")
        return None

def get_multiline_input():
    """Get multi-line input from user"""
    print("\n📝 Paste your article below.")
    print("When done, press Enter twice (two blank lines):\n")
    
    lines = []
    consecutive_empty = 0
    
    while True:
        try:
            line = input()
            
            if line == "":
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    # Remove trailing empty lines
                    while lines and lines[-1] == "":
                        lines.pop()
                    break
                lines.append(line)
            else:
                consecutive_empty = 0
                lines.append(line)
                
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n\nInput cancelled.")
            return ""
    
    return "\n".join(lines)

def smart_truncate(content: str, max_chars: int = 6000) -> str:
    """Intelligently truncate content"""
    if len(content) <= max_chars:
        return content
    
    # Remove extra whitespace
    content = ' '.join(content.split())
    
    if len(content) <= max_chars:
        return content
    
    # Try to find a good break point
    truncated = content[:max_chars]
    
    # Look for sentence endings
    for delimiter in ['. ', '.\n', '! ', '? ', '\n\n']:
        last_pos = truncated.rfind(delimiter)
        if last_pos > max_chars * 0.8:  # Within 80% of limit
            return truncated[:last_pos + len(delimiter)].strip()
    
    # Fallback: break at word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:
        return truncated[:last_space].strip()
    
    return truncated.strip()

def extract_article_text(html_content: str) -> str:
    """Extract readable text from HTML content"""
    # Simple extraction - remove obvious HTML
    import re
    
    # Remove script and style elements
    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', html_content)
    
    # Clean up whitespace
    text = ' '.join(text.split())
    
    # Remove common boilerplate patterns
    text = re.sub(r'(Cookie Policy|Privacy Policy|Terms of Service|Subscribe|Newsletter|Advertisement).*?(?=\.|$)', '', text, flags=re.IGNORECASE)
    
    return text

def main():
    # Check API keys
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ ERROR: GOOGLE_API_KEY not found!")
        print("\nPlease create a .env file with:")
        print("GOOGLE_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Serper is optional
    if not os.getenv("SERPER_API_KEY"):
        print("⚠️  Warning: SERPER_API_KEY not found")
        print("Web search features will be limited\n")
    
    print("\n" + "="*60)
    print("🤖 CredAI News Analysis System")
    print("="*60)
    print("✅ Using Gemini 1.5 Flash with automatic rate limiting")
    print("⏱️  Processing takes 3-5 minutes to respect API limits\n")
    
    # Get input
    choice = input("Enter 1 to paste article text, 2 to enter URL: ").strip()
    
    if choice == "1":
        content = get_multiline_input()
        if not content.strip():
            print("\n❌ No content provided!")
            sys.exit(1)
    elif choice == "2":
        url = input("\nEnter article URL: ").strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        print("🔄 Fetching content...")
        html_content = fetch_url(url)
        if not html_content:
            print("\n💡 Tip: Copy the article text from your browser and use option 1")
            sys.exit(1)
        
        # Extract text from HTML
        content = extract_article_text(html_content)
        if len(content) < 100:
            print("❌ Could not extract article text from webpage")
            print("💡 Try copying and pasting the article text instead")
            sys.exit(1)
    else:
        print("❌ Invalid choice.")
        sys.exit(1)
    
    # Content statistics
    original_length = len(content)
    print(f"\n📊 Original content: {original_length:,} characters")
    
    # Smart truncation for API limits
    MAX_CHARS = 6000  # Conservative limit
    if original_length > MAX_CHARS:
        content = smart_truncate(content, MAX_CHARS)
        print(f"✂️  Truncated to: {len(content):,} characters")
        print("💡 For full analysis of long articles, consider:")
        print("   - Breaking into sections")
        print("   - Upgrading to paid API plan")
    
    # Verify content quality
    if len(content) < 100:
        print("\n❌ Article too short for meaningful analysis")
        sys.exit(1)
    
    # Create crew
    crew = Crew(
        agents=[news_analyst, cred_checker, source_hunter, bias_detector],
        tasks=[analysis_task, credibility_task, source_validation_task, bias_detection_task],
        process=Process.sequential,
        verbose=True,
        memory=False,
        cache=False,
        max_rpm=5,
        share_crew=False
    )
    
    try:
        print("\n" + "="*60)
        print("🔄 Starting Analysis")
        print("="*60)
        print("📍 Progress: Each dot represents an API call")
        print("⏱️  Estimated time: 3-5 minutes\n")
        
        start_time = time.time()
        
        # Initial delay
        time.sleep(2)
        
        # Run analysis
        result = crew.kickoff(inputs={"article": content})
        
        elapsed = time.time() - start_time
        
        # Display results
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print("="*60)
        print(f"⏱️  Total time: {elapsed//60:.0f}m {elapsed%60:.0f}s\n")
        
        print("📋 FINAL REPORT:")
        print("-"*60)
        print(result)
        print("-"*60)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"news_analysis_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("CredAI News Analysis Report\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing time: {elapsed//60:.0f}m {elapsed%60:.0f}s\n")
            f.write(f"Content analyzed: {len(content):,} characters\n")
            f.write("="*60 + "\n\n")
            f.write("ANALYSIS RESULTS:\n")
            f.write("-"*60 + "\n")
            f.write(str(result))
        
        print(f"\n💾 Report saved: {filename}")
        print("✅ Analysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error: {error_msg}")
        
        if "429" in error_msg or "rate limit" in error_msg.lower():
            print("\n🚨 RATE LIMIT ERROR DETECTED")
            print("\nImmediate solutions:")
            print("1. ⏰ Wait 60 seconds and try again")
            print("2. 📄 Use a shorter article")
            print("3. 🕒 Try again in 1-2 minutes")
            print("\nLong-term solutions:")
            print("4. 💳 Upgrade to Google AI Studio paid plan")
            print("5. 🔑 Use a different API key")
            print("6. 🌐 Try during off-peak hours (late night/early morning)")
        elif "API key" in error_msg:
            print("\n🔑 API KEY ERROR")
            print("1. Verify your key at: https://aistudio.google.com/apikey")
            print("2. Make sure it's a Google AI Studio key (not Cloud)")
            print("3. Check if the key has expired")
        else:
            print("\n💡 Troubleshooting tips:")
            print("1. Check your internet connection")
            print("2. Verify API keys in .env file")
            print("3. Try with a shorter article")
            print("4. Check API status at Google AI Studio")
        
        if "--debug" in sys.argv:
            print("\n📝 Debug trace:")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()