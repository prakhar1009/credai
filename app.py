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

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()

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
        color: white;
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
        background: white;
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
    """Initialize and cache the AI agents"""
    # Configure API keys
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ GOOGLE_API_KEY not found! Please add it to your .env file.")
        return None
    
    # Initialize LLM
    llm = LLM(
        model="gemini/gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.5,
        max_retries=3,
        timeout=60,
        max_rpm=8
    )
    
    # Initialize search tool
    search_tool = SerperDevTool() if os.getenv("SERPER_API_KEY") else None
    
    # Define Agents
    news_analyst = Agent(
        role="Senior News Analyst",
        goal="Extract and summarize the most important facts from news articles with precision and clarity",
        backstory="""You are a veteran news analyst with 20 years of experience at major publications. 
        You excel at quickly identifying the key facts in any article and presenting them clearly. 
        You focus on concrete information: who, what, when, where, why, and how.""",
        llm=llm,
        verbose=False,
        memory=False,
        allow_delegation=False,
        max_iter=1,
        max_rpm=8
    )
    
    cred_checker = Agent(
        role="Media Credibility Expert",
        goal="Evaluate the reliability and trustworthiness of news articles based on journalistic standards",
        backstory="""You are an expert in media literacy and journalism ethics with a background in fact-checking. 
        You evaluate articles based on source quality, attribution, evidence, and journalistic standards.""",
        llm=llm,
        verbose=False,
        memory=False,
        tools=[search_tool] if search_tool else [],
        allow_delegation=False,
        max_iter=1,
        max_rpm=8
    )
    
    source_hunter = Agent(
        role="Fact Verification Specialist",
        goal="Verify key claims in articles using reliable sources and evidence",
        backstory="""You are a meticulous fact-checker who specializes in verifying claims using multiple sources. 
        You have extensive experience in investigative research and cross-referencing information.""",
        llm=llm,
        tools=[search_tool] if search_tool else [],
        verbose=False,
        memory=False,
        allow_delegation=False,
        max_iter=3,
        max_rpm=8
    )
    
    bias_detector = Agent(
        role="Media Bias Analyst",
        goal="Identify and analyze potential bias in news reporting through language and framing analysis",
        backstory="""You are a linguistics expert specializing in media analysis and bias detection. 
        You have studied how language choices, framing, and narrative structure can influence perception.""",
        llm=llm,
        verbose=False,
        memory=False,
        allow_delegation=False,
        max_iter=1,
        max_rpm=8
    )
    
    return news_analyst, cred_checker, source_hunter, bias_detector

def create_tasks(agents):
    """Create analysis tasks"""
    news_analyst, cred_checker, source_hunter, bias_detector = agents
    
    analysis_task = Task(
        description="""Analyze the provided article content and create 5 key bullet points summarizing the main facts.
        Article content: {article}
        
        Format your response as clear bullet points focusing on concrete information.""",
        expected_output="5 clear bullet points summarizing the article's main facts",
        agent=news_analyst
    )
    
    credibility_task = Task(
        description="""Assess the credibility of the article provided.
        Article content: {article}
        
        Provide a credibility score (0-100) with explanation.""",
        expected_output="Credibility score (0-100) with explanation",
        agent=cred_checker
    )
    
    source_validation_task = Task(
        description="""Identify and verify the top 3 most important factual claims in the article.
        Article content: {article}
        
        Mark each as: Verified, Unverified, or Disputed with explanations.""",
        expected_output="List of 3 claims with verification status and explanations",
        agent=source_hunter
    )
    
    bias_detection_task = Task(
        description="""Analyze the article for potential bias in language, framing, or perspective.
        Article content: {article}
        
        Provide a bias score (0-100) with examples if biased.""",
        expected_output="Bias score (0-100) with examples if biased",
        agent=bias_detector
    )
    
    return [analysis_task, credibility_task, source_validation_task, bias_detection_task]

def run_analysis(content: str):
    """Run the news analysis"""
    agents = initialize_agents()
    if not agents:
        return None
    
    tasks = create_tasks(agents)
    
    # Create crew
    crew = Crew(
        agents=list(agents),
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
        memory=False,
        max_rpm=5,
        planning=False
    )
    
    # Run analysis
    result = crew.kickoff(inputs={"article": content})
    return format_analysis_output(result)

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