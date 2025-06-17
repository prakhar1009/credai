# CredAI News Analysis

CredAI is a powerful news analysis tool that leverages AI agents to provide in-depth analysis of news articles. It extracts key facts, evaluates the credibility of the source, and provides a comprehensive summary to help you stay informed with reliable information.

## Features

-   **Fact Extraction:** Identifies and extracts the most important facts from a news article (who, what, when, where, why, and how).
-   **Credibility Analysis:** Assesses the trustworthiness of news articles based on journalistic standards.
-   **Web Interface:** A user-friendly web interface built with Streamlit to easily analyze articles by simply providing a URL.
-   **Powered by CrewAI:** Utilizes a crew of AI agents to perform specialized tasks, ensuring high-quality analysis.

## Project Structure

-   `app.py`: This file contains the Streamlit web application. It provides the user interface for interacting with the news analysis tool.
-   `news.py`: This is the core of the project. It defines the AI agents (`news_analyst`, `cred_checker`), the tasks they perform, and the overall `Crew` that orchestrates the analysis process.

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/prakhar1009/credai.git
    cd credai
    ```

2.  **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**

    Create a `.env` file in the root directory and add your API keys:

    ```
    GOOGLE_API_KEY="Your-Google-API-Key"
    SERPER_API_KEY="Your-Serper-API-Key"
    ```

## How to Run

To start the application, run the following command in your terminal:

```bash
streamlit run app.py
```

This will open the CredAI News Analysis tool in your web browser. Simply paste the URL of a news article you want to analyze and click the "Analyze" button.
