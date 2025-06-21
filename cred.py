# news.py - Fixed version
import os
import time
import sys
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
import requests
from datetime import datetime
import warnings
import re
from bs4 import BeautifulSoup

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

# Add manager configuration for better coordination
manager_llm = LLM(
    model="gemini/gemini-2.0-flash-exp",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,  # Lower temperature for more consistent management
    max_retries=3,
    timeout=60,
    max_rpm=8
)

# Initialize search tool
search_tool = SerperDevTool()

# Define Agents with enhanced capabilities (removed system_template parameter)
news_analyst = Agent(
    role="Senior News Analyst",
    goal="Extract and summarize the most important facts from news articles with precision and clarity",
    backstory="""You are a veteran news analyst with 20 years of experience at major publications. 
    You excel at quickly identifying the key facts in any article and presenting them clearly. 
    You focus on concrete information: who, what, when, where, why, and how. 
    You never use placeholder text and always analyze the actual content provided.
    
    When analyzing articles:
    1. Always work with the actual article content provided
    2. Extract specific facts, not generalizations
    3. Include names, numbers, dates, and locations when available
    4. Never use placeholder or template text
    5. Be concise but informative""",
    llm=llm,
    verbose=True,
    memory=False,
    allow_delegation=False,
    max_iter=1,
    max_rpm=8
)

cred_checker = Agent(
    role="Media Credibility Expert",
    goal="Evaluate the reliability and trustworthiness of news articles based on journalistic standards",
    backstory="""You are an expert in media literacy and journalism ethics with a background in fact-checking. 
    You evaluate articles based on source quality, attribution, evidence, and journalistic standards. 
    You provide balanced assessments backed by specific observations from the text.
    
    When assessing articles:
    1. Look for specific source attribution and quotes
    2. Check if claims are supported by evidence
    3. Evaluate the balance and fairness of reporting
    4. Consider the publication and author credibility
    5. Always base your assessment on the actual article content""",
    llm=llm,
    verbose=True,
    memory=False,
    tools=[search_tool],
    allow_delegation=False,
    max_iter=1,
    max_rpm=8
)

source_hunter = Agent(
    role="Advanced Fact Verification Specialist",
    goal="Verify key claims in articles using reliable sources and evidence, providing source URLs",
    backstory="""You are an elite fact-checker with experience at leading verification organizations like Snopes, 
    PolitiFact, and Reuters Fact Check. You excel at verifying claims using multiple credible sources and providing 
    direct evidence links. You're known for your thoroughness and ability to find primary sources.
    
    When verifying claims:
    1. Extract specific, verifiable factual claims from the article
    2. Use search tools to find supporting or conflicting evidence from reputable sources
    3. ALWAYS include the full URLs to verification sources for each claim
    4. Prioritize primary sources, official statistics, and reputable news outlets
    5. For each claim, provide a confidence level along with your verification
    6. If conflicting information exists, present both sides with their respective sources
    7. Never make verification statements without linking to specific sources
    8. For claims that touch on scientific or technical matters, seek peer-reviewed research""",
    llm=llm,
    tools=[search_tool],
    verbose=True,
    memory=False,
    allow_delegation=True,  # Allow delegation to other agents for complex cases
    max_iter=5,  # Increased iterations for more thorough search
    max_rpm=8
)

bias_detector = Agent(
    role="Media Bias Analyst",
    goal="Identify and analyze potential bias in news reporting through language and framing analysis",
    backstory="""You are a linguistics expert specializing in media analysis and bias detection. 
    You have studied how language choices, framing, and narrative structure can influence perception. 
    You provide objective analysis of potential bias while recognizing that all reporting has some perspective.
    
    When analyzing for bias:
    1. Examine specific word choices and their connotations
    2. Look for loaded language or emotional appeals
    3. Check if all relevant perspectives are included
    4. Identify what might be missing from the story
    5. Provide specific examples from the text""",
    llm=llm,
    verbose=True,
    memory=False,
    allow_delegation=False,
    max_iter=1,
    max_rpm=8
)

# Enhanced tasks with better context handling and examples
analysis_task = Task(
    description="""Analyze the provided article content and create 5 key bullet points summarizing the main facts.
    
    Article content: {article}
    
    Instructions:
    1. Read the entire article carefully
    2. Identify the 5 most important facts or developments
    3. Write each fact as a clear, concise bullet point
    4. Focus on concrete information (numbers, names, places, events)
    5. Each bullet should be 1-2 sentences maximum
    
    Format your response as:
    • [First key fact]
    • [Second key fact]
    • [Third key fact]
    • [Fourth key fact]
    • [Fifth key fact]
    
    Example output:
    • Canadian tourism to US border states has dropped by 50% since spring according to Kingdom Trails Association
    • Jay Peak Resort reports cancellations started after Trump's "51st state" comments about Canada
    • Vermont businesses are offering "at-par" pricing to attract Canadian customers back
    • Maine Governor installed bilingual "Bienvenue Canadiens" signs along interstates
    • 60% of Jay Peak's summer traffic typically comes from Canadian visitors""",
    expected_output="5 clear bullet points summarizing the article's main facts",
    agent=news_analyst
)

credibility_task = Task(
    description="""Assess the credibility of the article provided below.
    
    Article content: {article}
    
    Instructions:
    1. Evaluate the sources mentioned (are they named and reputable?)
    2. Check for specific data, quotes, and attribution
    3. Look for balanced perspectives or potential bias
    4. Assess factual accuracy based on your knowledge
    5. Rate credibility from 0-100 (0=not credible, 100=highly credible)
    
    Provide:
    - Credibility Score: [0-100]
    - Explanation: [2-3 sentences explaining your rating]""",
    expected_output="Credibility score (0-100) with 2-3 sentence explanation",
    agent=cred_checker
)

source_validation_task = Task(
    description="""Identify and verify the top 3 most important factual claims in the article with source links.
    
    Article content: {article}
    
    Instructions:
    1. Identify the 3 most significant factual claims (statistics, quotes, events) from the article
    2. Use web search to verify each claim using multiple credible sources
    3. For EACH claim, you MUST provide: 
       - Verification status: Verified (strong evidence), Partially Verified (some evidence), Disputed (conflicting information), or Unverified (insufficient evidence)
       - Confidence level (High/Medium/Low)
       - AT LEAST 2 specific verification source URLs for each claim
       - URLs must be complete and functional (e.g., https://www.example.com/article-path)
    4. If sources contradict each other, explain the discrepancy and cite both perspectives
    
    Format each claim verification as:
    
    ### Claim 1: [Exact claim from article]
    **Status:** [Verification status] (Confidence: [Level])
    **Analysis:** [2-3 sentence explanation of verification findings]
    **Sources:**
    - [Source name 1]: [Full URL 1]
    - [Source name 2]: [Full URL 2]
    - [Additional sources if relevant]: [Full URL]
    
    ### Claim 2: [Exact claim from article]
    **Status:** [Verification status] (Confidence: [Level])
    **Analysis:** [2-3 sentence explanation of verification findings]
    **Sources:**
    - [Source name 1]: [Full URL 1]
    - [Source name 2]: [Full URL 2]
    
    ### Claim 3: [Exact claim from article]
    **Status:** [Verification status] (Confidence: [Level])
    **Analysis:** [2-3 sentence explanation of verification findings]
    **Sources:**
    - [Source name 1]: [Full URL 1]
    - [Source name 2]: [Full URL 2]
    
    Remember: NEVER assert verification without providing specific source URLs. If you cannot find verification sources, state this explicitly.""",
    expected_output="Detailed verification of 3 key claims with source links",
    agent=source_hunter
)

bias_detection_task = Task(
    description="""Conduct a comprehensive analysis of potential bias in the article's language, framing, and perspective.
    
    Article content: {article}
    
    Instructions:
    1. Examine specific word choices, tone, and framing techniques
    2. Evaluate representation of multiple perspectives and viewpoints
    3. Identify missing context, one-sided reporting, or narrative techniques
    4. Analyze source selection and how quotes are presented
    5. Rate overall bias from 0-100 (0=completely neutral, 100=extremely biased)
    
    Provide your analysis in this format:
    
    ## Bias Assessment
    **Bias Score:** [0-100]
    
    **Bias Direction:** [Left-leaning/Right-leaning/Corporate/Anti-establishment/Other specific bias type, or "Balanced" if neutral]
    
    **Language Analysis:**
    - Evaluate emotional vs. neutral language
    - Note any loaded terms, euphemisms, or framing devices
    - Identify if language appeals to emotion over facts
    
    **Perspective Analysis:**
    - Are multiple perspectives presented fairly?
    - Which viewpoints receive more coverage or favorable treatment?
    - Are any critical perspectives missing?
    
    **Specific Examples:**
    1. [First example - quote specific text and explain bias]
    2. [Second example - quote specific text and explain bias]
    3. [Third example - if applicable]
    
    If the article scores ≤ 20 on the bias scale, you may state "The article appears largely neutral in tone and presentation" and provide brief explanation of what makes it balanced.""",
    expected_output="Detailed bias analysis with specific examples and explanation of bias type",
    agent=bias_detector
)

def extract_article_text(html_content: str) -> str:
    """Extract article text from HTML content"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Try to find main content area
        content_selectors = [
            'article',
            '[role="main"]',
            '.article-content',
            '.story-content',
            '.post-content',
            '.entry-content',
            '.content',
            'main'
        ]
        
        content_text = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    content_text += element.get_text()
                break
        
        # Fallback to body if no content area found
        if not content_text:
            body = soup.find('body')
            if body:
                content_text = body.get_text()
        
        # Clean up text
        lines = (line.strip() for line in content_text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        content_text = ' '.join(chunk for chunk in chunks if chunk)
        
        return content_text
    
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def preprocess_article(content: str) -> str:
    """Clean and preprocess article content"""
    if not content or len(content.strip()) < 50:
        raise ValueError("Article content is too short or empty")
    
    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content)
    
    # Remove common website elements
    content = re.sub(r'(subscribe|newsletter|advertisement|cookie policy)', '', content, flags=re.IGNORECASE)
    
    # Basic cleaning
    content = content.strip()
    
    return content

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

def format_analysis_output(crew_result) -> dict:
    """Format the analysis output for better readability"""
    try:
        # Extract individual task results
        tasks_output = crew_result.tasks_output if hasattr(crew_result, 'tasks_output') else []
        
        formatted_result = {
            'key_points': '',
            'credibility': '',
            'fact_verification': '',
            'bias_analysis': ''
        }
        
        # Map each task output to the correct section
        task_names = ['analysis_task', 'credibility_task', 'source_validation_task', 'bias_detection_task']
        result_keys = ['key_points', 'credibility', 'fact_verification', 'bias_analysis']
        
        for i, task_output in enumerate(tasks_output):
            if i < len(result_keys):
                formatted_result[result_keys[i]] = str(task_output.raw)
        
        return formatted_result
    except:
        # Fallback to parsing the string output
        result_str = str(crew_result)
        return {
            'key_points': result_str,
            'credibility': '',
            'fact_verification': '',
            'bias_analysis': ''
        }

def print_colored_text(text: str, color: str = 'white'):
    """Print colored text to terminal"""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bold': '\033[1m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def create_markdown_report(formatted_result: dict, content_length: int, processing_time: float, article_url: str = None) -> str:
    """Create an enhanced formatted markdown report with better source link formatting"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown_content = f"""# 📰 CredAI News Analysis Report

**Generated:** {timestamp}  
**Processing Time:** {processing_time//60:.0f}m {processing_time%60:.0f}s  
**Content Analyzed:** {content_length:,} characters  
"""
    
    if article_url:
        markdown_content += f"**Source URL:** [{article_url}]({article_url})\n"
    
    markdown_content += "\n---\n\n"
    
    # Key Points Section
    if formatted_result['key_points']:
        markdown_content += "## 📌 Key Points Summary\n\n"
        markdown_content += formatted_result['key_points'] + "\n\n"
    
    # Credibility Assessment
    if formatted_result['credibility']:
        markdown_content += "## 🔍 Credibility Assessment\n\n"
        markdown_content += formatted_result['credibility'] + "\n\n"
    
    # Fact Verification - Enhanced to preserve markdown links
    if formatted_result['fact_verification']:
        markdown_content += "## ✅ Fact Verification\n\n"
        fact_content = formatted_result['fact_verification']
        # Make sure URLs are properly formatted as markdown links if they aren't already
        url_pattern = r'(https?://[^\s\)]+)(?![\)\]])'  # Match URLs not already part of markdown links
        fact_content = re.sub(url_pattern, r'[\1](\1)', fact_content)
        markdown_content += fact_content + "\n\n"
    
    # Bias Analysis
    if formatted_result['bias_analysis']:
        markdown_content += "## ⚖️ Bias Analysis\n\n"
        markdown_content += formatted_result['bias_analysis'] + "\n\n"
    
    # Add source reliability section
    markdown_content += "## 📊 Source Reliability Guide\n\n"
    markdown_content += """The following guide explains our source reliability ratings:

| Source Type | Reliability |
|-------------|-------------|
| Academic Research | Very High - Peer-reviewed studies and research from established institutions |
| Government Data | High - Official statistics and information from government agencies |
| Major News Outlets | Medium to High - Established news organizations with editorial standards |
| Specialized Publications | Medium to High - Industry-specific publications with subject matter expertise |
| Opinion Pieces | Low to Medium - May contain factual information but with significant bias |
| Social Media | Low - Unverified information requiring additional confirmation |
| Anonymous Sources | Very Low - Cannot be independently verified |

"""
    
    # Add methodology section
    markdown_content += "## 🧪 Methodology\n\n"
    markdown_content += """This analysis was performed using a team of AI agents:

1. **News Analyst** extracted key facts from the article
2. **Credibility Expert** evaluated the article's reliability
3. **Fact Verification Specialist** verified claims using multiple external sources
4. **Bias Detector** analyzed the article for potential bias in language and framing

Each claim verification includes confidence ratings and multiple source links to support transparency and enable readers to verify information independently.
"""
    
    markdown_content += "\n---\n\n*Report generated by CredAI Enhanced News Analysis System v2.0*"""
    
    return markdown_content

def main():
    # Check API keys
    if not os.getenv("GOOGLE_API_KEY"):
        print_colored_text("❌ ERROR: GOOGLE_API_KEY not found!", 'red')
        print_colored_text("\nPlease create a .env file with:", 'yellow')
        print_colored_text("GOOGLE_API_KEY=your_api_key_here", 'cyan')
        sys.exit(1)
    
    # Serper is optional
    if not os.getenv("SERPER_API_KEY"):
        print_colored_text("⚠️  Warning: SERPER_API_KEY not found", 'yellow')
        print_colored_text("Web search features will be limited\n", 'yellow')
    
    # Create reports directory
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Enhanced header
    print_colored_text("\n" + "="*70, 'cyan')
    print_colored_text("🤖 CredAI News Analysis System v2.0", 'bold')
    print_colored_text("="*70, 'cyan')
    print_colored_text("✅ Using Gemini 1.5 Flash with automatic rate limiting", 'green')
    print_colored_text("⏱️  Processing takes 3-5 minutes to respect API limits", 'blue')
    print_colored_text("📁 Reports saved in markdown format\n", 'purple')
    
    # Get input with better prompts
    print_colored_text("📝 Input Options:", 'bold')
    print_colored_text("  1️⃣  Paste article text", 'white')
    print_colored_text("  2️⃣  Enter article URL", 'white')
    
    choice = input("\n🔽 Choose option (1 or 2): ").strip()
    article_url = None
    
    if choice == "1":
        content = get_multiline_input()
        if not content.strip():
            print_colored_text("\n❌ No content provided!", 'red')
            sys.exit(1)
    elif choice == "2":
        url = input("\n🌐 Enter article URL: ").strip()
        article_url = url
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        print_colored_text("🔄 Fetching content...", 'blue')
        html_content = fetch_url(url)
        if not html_content:
            print_colored_text("\n💡 Tip: Copy the article text from your browser and use option 1", 'yellow')
            sys.exit(1)
        
        # Extract text from HTML
        content = extract_article_text(html_content)
        if len(content) < 100:
            print_colored_text("❌ Could not extract article text from webpage", 'red')
            print_colored_text("💡 Try copying and pasting the article text instead", 'yellow')
            sys.exit(1)
    else:
        print_colored_text("❌ Invalid choice.", 'red')
        sys.exit(1)
    
    # Content statistics with better formatting
    original_length = len(content)
    print_colored_text(f"\n📊 Content Analysis:", 'bold')
    print_colored_text(f"   📄 Original length: {original_length:,} characters", 'white')
    
    # Smart truncation for API limits
    MAX_CHARS = 6000  # Conservative limit
    if original_length > MAX_CHARS:
        content = smart_truncate(content, MAX_CHARS)
        print_colored_text(f"   ✂️  Truncated to: {len(content):,} characters", 'yellow')
        print_colored_text("   💡 For full analysis of long articles:", 'cyan')
        print_colored_text("      - Break into sections", 'cyan')
        print_colored_text("      - Upgrade to paid API plan", 'cyan')
    
    # Verify content quality
    if len(content) < 100:
        print_colored_text("\n❌ Article too short for meaningful analysis", 'red')
        sys.exit(1)
    
    # Preprocess the article
    try:
        processed_content = preprocess_article(content)
    except ValueError as e:
        print_colored_text(f"\n❌ {e}", 'red')
        sys.exit(1)
    
    # Create crew with enhanced configuration
    crew = Crew(
        agents=[news_analyst, cred_checker, source_hunter, bias_detector],
        tasks=[analysis_task, credibility_task, source_validation_task, bias_detection_task],
        process=Process.sequential,
        verbose=False,  # Reduce verbose output for cleaner display
        memory=False,
        max_rpm=5,
        planning=False
    )
    
    try:
        print_colored_text("\n" + "="*70, 'cyan')
        print_colored_text("🔄 Starting Analysis Pipeline", 'bold')
        print_colored_text("="*70, 'cyan')
        
        # Progress indicators
        steps = [
            "📊 Analyzing key facts",
            "🔍 Assessing credibility", 
            "✅ Verifying claims",
            "⚖️  Detecting bias"
        ]
        
        for step in steps:
            print_colored_text(f"⏳ {step}...", 'blue')
        
        print_colored_text("\n⏱️  Estimated time: 3-5 minutes", 'yellow')
        print()
        
        start_time = time.time()
        
        # Run analysis with processed content
        result = crew.kickoff(inputs={"article": processed_content})
        
        elapsed = time.time() - start_time
        
        # Format results
        formatted_result = format_analysis_output(result)
        
        # Display results with enhanced formatting
        print_colored_text("\n" + "="*70, 'green')
        print_colored_text("✅ ANALYSIS COMPLETE", 'bold')
        print_colored_text("="*70, 'green')
        print_colored_text(f"⏱️  Processing time: {elapsed//60:.0f}m {elapsed%60:.0f}s", 'cyan')
        print()
        
        # Display each section with colors
        sections = [
            ("📌 KEY POINTS SUMMARY", 'key_points', 'blue'),
            ("🔍 CREDIBILITY ASSESSMENT", 'credibility', 'green'), 
            ("✅ FACT VERIFICATION", 'fact_verification', 'yellow'),
            ("⚖️ BIAS ANALYSIS", 'bias_analysis', 'purple')
        ]
        
        for title, key, color in sections:
            if formatted_result[key]:
                print_colored_text(f"\n{title}", color)
                print_colored_text("-" * 50, color)
                print(formatted_result[key])
        
        print_colored_text("\n" + "="*70, 'green')
        
        # Save report as markdown
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"news_analysis_{timestamp}.md"
        filepath = os.path.join(reports_dir, filename)
        
        markdown_content = create_markdown_report(
            formatted_result, 
            len(content), 
            elapsed,
            article_url
        )
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        print_colored_text(f"💾 Report saved: {filepath}", 'green')
        print_colored_text("✨ Analysis completed successfully!", 'bold')
        
    except KeyboardInterrupt:
        print_colored_text("\n\n⚠️  Analysis interrupted by user", 'yellow')
        sys.exit(0)
    except Exception as e:
        error_msg = str(e)
        print_colored_text(f"\n❌ Error: {error_msg}", 'red')
        
        if "429" in error_msg or "rate limit" in error_msg.lower():
            print_colored_text("\n🚨 RATE LIMIT ERROR DETECTED", 'red')
            print_colored_text("\nImmediate solutions:", 'yellow')
            print_colored_text("1. ⏰ Wait 60 seconds and try again", 'white')
            print_colored_text("2. 📄 Use a shorter article", 'white')
            print_colored_text("3. 🕒 Try again in 1-2 minutes", 'white')
            print_colored_text("\nLong-term solutions:", 'cyan')
            print_colored_text("4. 💳 Upgrade to Google AI Studio paid plan", 'white')
            print_colored_text("5. 🔑 Use a different API key", 'white')
            print_colored_text("6. 🌐 Try during off-peak hours", 'white')
        elif "API key" in error_msg:
            print_colored_text("\n🔑 API KEY ERROR", 'red')
            print_colored_text("1. Verify your key at: https://aistudio.google.com/apikey", 'cyan')
            print_colored_text("2. Make sure it's a Google AI Studio key (not Cloud)", 'white')
            print_colored_text("3. Check if the key has expired", 'white')
        else:
            print_colored_text("\n💡 Troubleshooting tips:", 'yellow')
            print_colored_text("1. Check your internet connection", 'white')
            print_colored_text("2. Verify API keys in .env file", 'white')
            print_colored_text("3. Try with a shorter article", 'white')
            print_colored_text("4. Check API status at Google AI Studio", 'white')
        
        if "--debug" in sys.argv:
            print_colored_text("\n📝 Debug trace:", 'cyan')
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()