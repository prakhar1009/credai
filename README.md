# CredAI News Analysis

CredAI is a powerful news analysis tool that leverages AI agents to provide in-depth analysis of news articles. It extracts key facts, evaluates the credibility of the source, identifies potential bias, and provides a comprehensive summary to help you stay informed with reliable information.

## Features

- **Fact Extraction:** Identifies and extracts the most important facts from a news article (who, what, when, where, why, and how).
- **Credibility Analysis:** Assesses the trustworthiness of news articles based on journalistic standards with a 0-100 score.
- **Fact Verification:** Verifies key claims in the article through external sources and web searches with full source URLs.
- **Bias Detection:** Analyzes language, framing, and perspective to identify potential bias with a 0-100 score and bias direction.
- **Dual Interface:** 
  - CLI version for direct terminal use with colored output
  - Web interface built with Streamlit for easy article analysis via URL or text input
- **Smart Content Processing:** Intelligently extracts article text from URLs and handles content truncation for API limits.
- **Automated Report Generation:** Creates detailed markdown reports with timestamps stored in the reports directory.
- **Intelligent Rate Limiting:** Built-in rate limiter ensures API compliance (8 RPM) with automatic throttling.
- **Robust URL Handling:** Handles website access restrictions with user-friendly fallbacks.

## Project Structure

- `app.py`: The Streamlit web application providing a user-friendly interface for article analysis.
- `news.py`: The core module that defines the AI agents, tasks, and orchestration logic. Can be run as a standalone CLI tool.
- `cred.py`: Enhanced version of news.py with improved rate limiting, better error handling, and content extraction capabilities.
- `requirements.txt`: Lists all the Python dependencies needed to run the application.
- `reports/`: Directory where analysis reports are saved as markdown files with timestamps (e.g., `news_analysis_20250621_213608.md`).

## AI Agents

CredAI uses a crew of four specialized AI agents, each with specific expertise:

1. **News Analyst**
   - Role: Senior News Analyst
   - Goal: Extract and summarize the most important facts with precision and clarity
   - Expertise: 20 years of experience at major publications, focuses on concrete who/what/when/where/why/how
   - Output: 5 clear, concise bullet points summarizing the article's main facts

2. **Credibility Checker**
   - Role: Media Credibility Expert
   - Goal: Evaluate reliability based on journalistic standards
   - Expertise: Media literacy and journalism ethics with fact-checking background
   - Output: Credibility score (0-100) with detailed explanation of sources, attribution, evidence, and balance

3. **Source Hunter**
   - Role: Advanced Fact Verification Specialist
   - Goal: Verify key claims using reliable sources and evidence with source URLs
   - Expertise: Elite fact-checker with experience at leading verification organizations
   - Tools: Uses Serper web search for verification
   - Output: Detailed verification of 3 key claims with verification status, confidence level (High/Medium/Low), analysis, and multiple source URLs

4. **Bias Detector**
   - Role: Media Bias Analyst
   - Goal: Identify and analyze potential bias through language and framing analysis
   - Expertise: Linguistics specialist in media analysis and bias detection
   - Output: Bias score (0-100), bias direction (Left/Right/Corporate/etc.), language analysis, perspective analysis, and specific examples from the text

## Technical Details

- **LLMs**:
  - Primary: Google's Gemini 1.5 Flash model for agent reasoning
  - Management: Gemini 2.0 Flash Exp for better coordination
  - Parameters: 0.3-0.5 temperature settings for balanced creativity/consistency
  - Configuration: 3 max retries, 60-second timeouts

- **Rate Limiting**:
  - Custom RateLimiter class managing 8 requests per minute (RPM)
  - Automatic wait periods when approaching rate limits
  - Visual feedback during throttling periods

- **Search Capability**:
  - Web search via Serper API for fact verification
  - Source validation with full URL tracking
  - Tools configured for source_hunter agent only

- **Content Processing**:
  - BeautifulSoup-based intelligent article extraction
  - Smart content selectors for various webpage types
  - Regular expression cleaning and preprocessing
  - Smart truncation algorithm to handle API token limits

- **Error Handling**:
  - Robust handling of API rate limits
  - Connection issues detection with user-friendly messages
  - Fallback mechanisms for content extraction failures

## Setup and Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/prakhar1009/credai.git
    cd credai
    ```

2. **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**

    Create a `.env` file in the root directory and add your API keys:

    ```
    GOOGLE_API_KEY="Your-Google-API-Key"
    SERPER_API_KEY="Your-Serper-API-Key"  # Optional but recommended for fact verification
    ```

   Get your API keys from:
   - Google AI Studio: https://aistudio.google.com/apikey
   - Serper: https://serper.dev/

## How to Run

### Web Interface

To start the Streamlit web application:

```bash
python news.py
 or
streamlit run app.py
```

This will open the CredAI News Analysis tool in your web browser where you can:
- Paste article text directly
- Enter an article URL for automatic extraction
- View results with color-coded sections for each analysis aspect

### Command Line Interface

To run the CLI version:

```bash
python news.py
```
```bash
python cred.py
``` 

The CLI version offers:
- Option to paste article text or provide a URL
- Progress indicators during analysis
- Colored terminal output for results
- Automatic markdown report generation

## Troubleshooting

### API Rate Limits
If you encounter rate limit errors:
- Wait 60 seconds and try again
- Use a shorter article
- Upgrade to Google AI Studio paid plan

### API Key Issues
If experiencing API key errors:
- Verify your Google AI Studio key at: https://aistudio.google.com/apikey
- Ensure you're using an AI Studio key (not Google Cloud)
- Check if your key has expired

### Content Extraction
If URL content extraction fails:
- Copy the article text directly from your browser
- Ensure the website doesn't block automated access
- Try a different browser if experiencing issues
