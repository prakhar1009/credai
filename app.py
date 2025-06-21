# app.py - Streamlit interface for CredAI News Analysis
import streamlit as st
import os
import time
import requests
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
import re
from bs4 import BeautifulSoup
import warnings
import sys

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress litellm verbose logging
os.environ["LITELLM_LOG"] = "ERROR"
import litellm
litellm.set_verbose = False

# Load environment variables
load_dotenv()

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
            st.info(f"⏳ Rate limiting: Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()

# Global rate limiter - 8 requests per minute (safe for free tier)
rate_limiter = RateLimiter(requests_per_minute=8)

# Configure API keys
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# Configure Streamlit page
st.set_page_config(
    page_title="CredAI News Analysis",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: blue;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .analysis-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .metric-card {
        background: blue;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #007bff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Helper functions from news.py
def extract_article_text(html_content: str) -> str:
    """Extract article text from HTML content"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Try to find main content area
        content_selectors = [
            'article', '[role="main"]', '.article-content', '.story-content',
            '.post-content', '.entry-content', '.content', 'main'
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
        st.error(f"Error extracting text: {e}")
        return ""

def preprocess_article(content: str) -> str:
    """Clean and preprocess article content"""
    if not content or len(content.strip()) < 50:
        raise ValueError("Article content is too short or empty")
    
    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content)
    
    # Remove common website elements
    content = re.sub(r'(subscribe|newsletter|advertisement|cookie policy)', '', content, flags=re.IGNORECASE)
    
    return content.strip()

def fetch_url(url: str) -> str:
    """Fetch content from URL"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        st.error(f"Failed to fetch URL: {e}")
        return None

def smart_truncate(content: str, max_chars: int = 6000) -> str:
    """Intelligently truncate content"""
    if len(content) <= max_chars:
        return content
    
    content = ' '.join(content.split())
    if len(content) <= max_chars:
        return content
    
    truncated = content[:max_chars]
    
    # Look for sentence endings
    for delimiter in ['. ', '.\n', '! ', '? ', '\n\n']:
        last_pos = truncated.rfind(delimiter)
        if last_pos > max_chars * 0.8:
            return truncated[:last_pos + len(delimiter)].strip()
    
    # Fallback: break at word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:
        return truncated[:last_space].strip()
    
    return truncated.strip()

def format_analysis_output(crew_result) -> dict:
    """Format the analysis output for better readability"""
    try:
        tasks_output = crew_result.tasks_output if hasattr(crew_result, 'tasks_output') else []
        
        formatted_result = {
            'key_points': '',
            'credibility': '',
            'fact_verification': '',
            'bias_analysis': ''
        }
        
        result_keys = ['key_points', 'credibility', 'fact_verification', 'bias_analysis']
        
        for i, task_output in enumerate(tasks_output):
            if i < len(result_keys):
                formatted_result[result_keys[i]] = str(task_output.raw)
        
        return formatted_result
    except:
        result_str = str(crew_result)
        return {
            'key_points': result_str,
            'credibility': '',
            'fact_verification': '',
            'bias_analysis': ''
        }

@st.cache_resource
def initialize_agents():
    """Initialize and cache the AI agents and LLMs"""
    # Configure API keys
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ GOOGLE_API_KEY not found! Please add it to your .env file.")
        return None, None
    
    # Initialize our agents with the LLM
    llm = Gemini(
        model="gemini-1.5-flash", 
        request_timeout=120,
        callbacks=[lambda: rate_limiter.wait_if_needed()]
    )
    
    # Manager LLM for better coordination
    manager_llm = Gemini(
        model="gemini-2.0-flash-exp",
        temperature=0.3,  # Lower temperature for more consistent management
        request_timeout=120,
        callbacks=[lambda: rate_limiter.wait_if_needed()]
    )
    
    # Initialize search tool
    search_tool = SerperDevTool() if os.getenv("SERPER_API_KEY") else None
    
    # Create agents
    # 1. News Analyst - Focuses on extracting key information
    news_analyst = Agent(
        role="Senior News Analyst",
        goal="Extract and summarize the most important facts from news articles with precision and clarity",
        backstory="""You are a veteran news analyst with 20 years of experience at top news outlets. 
        Your specialty is identifying key facts and important developments in complex news stories.
        You have a talent for finding the signal in the noise and can quickly determine what matters most in any article.
        You pride yourself on extracting concrete information and presenting it in a clear, concise manner.
        You focus on answering the key journalistic questions: Who, What, When, Where, Why, and How.""",
        llm=llm,
        verbose=True,
        memory=False,
        allow_delegation=False,
        max_iter=1,
        max_rpm=8
    )
    
    # 2. Credibility Checker - Evaluates source reliability
    cred_checker = Agent(
        role="Credibility Assessment Specialist",
        goal="Evaluate the credibility and reliability of news articles based on journalistic standards",
        backstory="""You are a media literacy expert and former fact-checker who specializes in evaluating news credibility.
        You have worked for media watchdog organizations and journalism ethics boards for over 15 years.
        You know the hallmarks of reliable reporting: named sources, specific data, balanced perspectives,
        and transparent attribution. You can spot red flags like vague attributions, emotional language,
        and missing context. Your assessments are methodical and evidence-based, never partisan or ideological.
        You've developed a comprehensive framework for credibility scoring that's used by news organizations worldwide.""",
        llm=llm,
        verbose=True,
        memory=False,
        allow_delegation=False,
        max_iter=1,
        max_rpm=8
    )
    
    # 3. Source Hunter - Verifies article claims
    source_hunter = Agent(
        role="Investigative Fact Checker",
        goal="Verify factual claims in news articles through independent research and provide evidence",
        backstory="""You are a seasoned investigative journalist who specializes in fact-checking and verification.
        You have won prestigious awards for exposing misinformation and your meticulous research skills.
        You know how to find primary sources, cross-reference information, and determine the veracity of claims.
        You believe in evidence-based reporting and have a vast knowledge of reliable information sources.
        You never make assertions without backing them up with concrete evidence and specific sources.
        You understand that proper attribution is essential, and you always provide full URLs to your sources.
        When you cannot verify something, you clearly state that rather than making assumptions.
        
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
        tools=[search_tool] if search_tool else [],
        verbose=False,
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
        verbose=False,
        memory=False,
        allow_delegation=False,
        max_iter=1,
        max_rpm=8
    )
    
    agents = {
        'news_analyst': news_analyst,
        'cred_checker': cred_checker,
        'source_hunter': source_hunter,
        'bias_detector': bias_detector
    }
    
    return agents, manager_llm

def create_tasks(agents):
    """Create analysis tasks"""
    if not agents:
        st.error("❌ Cannot create tasks: agents not initialized")
        return []
        
    news_analyst = agents.get('news_analyst')
    cred_checker = agents.get('cred_checker')
    source_hunter = agents.get('source_hunter')
    bias_detector = agents.get('bias_detector')
    
    if not all([news_analyst, cred_checker, source_hunter, bias_detector]):
        st.warning("⚠️ Some agents could not be initialized properly")
        return []
    
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
    
    return [analysis_task, credibility_task, source_validation_task, bias_detection_task]

def run_analysis(content: str):
    """Run the news analysis"""
    agents, manager_llm = initialize_agents()
    if not agents:
        return None
    
    tasks = create_tasks(agents)
    
    # Create crew with manager LLM for better coordination
    crew = Crew(
        agents=list(agents),
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
        memory=False,
        max_rpm=8,  # Match our rate limiter setting
        manager_llm=manager_llm,  # Use specialized manager for task coordination
        planning=True  # Enable planning for better agent collaboration
    )
    
    # Run analysis with error handling
    try:
        result = crew.kickoff(inputs={"article": content})
        return format_analysis_output(result)
    except Exception as e:
        st.error(f"🛑 Error during analysis: {str(e)}")
        st.info("Try with a different article or check your API keys.")
        return None

# Main Streamlit App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 CredAI News Analysis System</h1>
        <p>AI-powered news credibility and bias analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API key status
        google_key = os.getenv("GOOGLE_API_KEY")
        serper_key = os.getenv("SERPER_API_KEY")
        
        if google_key:
            st.success("✅ Google API Key configured")
        else:
            st.error("❌ Google API Key missing")
            st.info("Add GOOGLE_API_KEY to your .env file")
        
        if serper_key:
            st.success("✅ Serper API Key configured")
        else:
            st.warning("⚠️ Serper API Key missing (optional)")
        
        st.markdown("---")
        
        # Analysis options
        st.header("📊 Analysis Options")
        max_chars = st.slider("Max characters to analyze", 1000, 10000, 6000)
        
        # Reports folder info
        st.markdown("---")
        st.header("📁 Reports")
        reports_dir = "reports"
        if os.path.exists(reports_dir):
            files = [f for f in os.listdir(reports_dir) if f.endswith('.md')]
            st.info(f"📄 {len(files)} reports saved")
        else:
            st.info("📄 No reports yet")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Article")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["📄 Paste article text", "🌐 Enter article URL"],
            horizontal=True
        )
        
        content = ""
        article_url = None
        
        if input_method == "📄 Paste article text":
            content = st.text_area(
                "Paste your article here:",
                height=300,
                placeholder="Paste the full article text here..."
            )
        else:
            article_url = st.text_input(
                "Enter article URL:",
                placeholder="https://example.com/article"
            )
            
            if article_url and st.button("🔄 Fetch Article"):
                with st.spinner("Fetching article content..."):
                    if not article_url.startswith(('http://', 'https://')):
                        article_url = 'https://' + article_url
                    
                    html_content = fetch_url(article_url)
                    if html_content:
                        content = extract_article_text(html_content)
                        if content:
                            st.success(f"✅ Fetched {len(content):,} characters")
                        else:
                            st.error("❌ Could not extract article text")
                    else:
                        st.error("❌ Failed to fetch article")
        
        # Display content stats
        if content:
            original_length = len(content)
            st.info(f"📊 Content length: {original_length:,} characters")
            
            if original_length > max_chars:
                st.warning(f"⚠️ Content will be truncated to {max_chars:,} characters")
        
        # Analysis button
        if content and len(content) > 100:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                st.session_state.processing = True
                
                # Preprocess content
                try:
                    processed_content = preprocess_article(content)
                    if len(processed_content) > max_chars:
                        processed_content = smart_truncate(processed_content, max_chars)
                    
                    # Run analysis
                    with st.spinner("🔄 Running AI analysis... This may take 3-5 minutes"):
                        progress_bar = st.progress(0)
                        
                        # Simulate progress
                        for i in range(100):
                            time.sleep(0.05)
                            progress_bar.progress(i + 1)
                        
                        result = run_analysis(processed_content)
                        
                        if result:
                            st.session_state.analysis_result = result
                            st.session_state.processing = False
                            st.success("✅ Analysis completed!")
                            st.rerun()
                        else:
                            st.error("❌ Analysis failed")
                            st.session_state.processing = False
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.processing = False
        
        elif content and len(content) <= 100:
            st.warning("⚠️ Article too short for analysis (minimum 100 characters)")
    
    with col2:
        st.header("📊 Analysis Results")
        
        if st.session_state.analysis_result:
            result = st.session_state.analysis_result
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["📌 Key Points", "🔍 Credibility", "✅ Fact Check", "⚖️ Bias"])
            
            with tab1:
                if result['key_points']:
                    st.markdown(f"""
                    <div class="analysis-card">
                        {result['key_points']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No key points analysis available")
            
            with tab2:
                if result['credibility']:
                    st.markdown(f"""
                    <div class="analysis-card">
                        {result['credibility']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No credibility analysis available")
            
            with tab3:
                if result['fact_verification']:
                    st.markdown(f"""
                    <div class="analysis-card">
                        {result['fact_verification']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No fact verification available")
            
            with tab4:
                if result['bias_analysis']:
                    st.markdown(f"""
                    <div class="analysis-card">
                        {result['bias_analysis']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No bias analysis available")
            
            # Download options
            st.markdown("---")
            st.subheader("💾 Export Results")
            
            # Create markdown report
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            markdown_content = f"""# 📰 CredAI News Analysis Report

**Generated:** {timestamp}
**Source URL:** {article_url if article_url else "Text input"}

---

## 📌 Key Points Summary
{result['key_points']}

## 🔍 Credibility Assessment
{result['credibility']}

## ✅ Fact Verification
{result['fact_verification']}

## ⚖️ Bias Analysis
{result['bias_analysis']}

---

*Report generated by CredAI News Analysis System*
"""
            
            st.download_button(
                label="📄 Download as Markdown",
                data=markdown_content,
                file_name=f"news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
            
            # Save to reports folder
            if st.button("💾 Save to Reports Folder", use_container_width=True):
                os.makedirs("reports", exist_ok=True)
                filename = f"news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                filepath = os.path.join("reports", filename)
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
                
                st.success(f"✅ Report saved to {filepath}")
        
        elif st.session_state.processing:
            st.info("🔄 Analysis in progress...")
        else:
            st.info("👆 Enter an article above to start analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🤖 CredAI News Analysis System v2.0 | Powered by CrewAI & Gemini</p>
        <p>⚡ Real-time news credibility and bias detection</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()