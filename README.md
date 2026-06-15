#  Congressional Bills Intelligence System

This repository contains an end-to-end Machine Learning and AI project that tracks, analyzes, and predicts the passage of U.S. Congressional bills from the 118th Congress. 

The system is split into two distinct parts: an automated data pipeline that performs Exploratory Data Analysis (EDA) and trains an XGBoost model, and a Retrieval-Augmented Generation (RAG) chatbot interface built with Streamlit and LLaMA-3.

##  Project Structure

This repository is organized into a monorepo containing two main sub-projects:

### [1. Data Pipeline & ML Training](./1_data_pipeline/)
*An automated pipeline to scrape Congress.gov, extract data journalism insights, and train passage prediction models.*
* **Data Collection**: Scrapes legislative data and bill texts from the Congress.gov REST API.
* **Exploratory Data Analysis (EDA)**: Uses interactive **Plotly** visualizations to uncover real-world mechanics of Congress (e.g., the "Naming Bill" effect, fast-tracking rules, and committee gatekeepers).
* **Feature Engineering**: NLP metrics (word counts, naming bill flags), categorical encoding, and sponsor tracking.
* **Machine Learning**: Trains an **XGBoost** classifier to predict if a bill will pass into law.
* **Evaluation**: Outputs ROC-AUC metrics and utilizes **SHAP** values for feature importance interpretability.

### [2. AI RAG Chatbot](./2_rag_app/)
*A front-end Streamlit application allowing users to search legislation via natural language.*
* **Vector Database**: Indexes legislative chunks using `sentence-transformers` and **ChromaDB**.
* **LLM Integration**: Connects to the **Groq API** (LLaMA-3.3-70B) to generate global answers and dynamic 1-sentence summaries for every retrieved bill.
* **Dynamic UI**: Features a modern, clean light-themed dashboard with probability progress bars, XGBoost signal indicators, and metadata filtering.

##  Live Demo
*(Insert your Hugging Face Spaces link here once deployed!)*

---

**Tech Stack**: Python, Pandas, Plotly, XGBoost, SHAP, ChromaDB, Hugging Face (Sentence-Transformers), Groq (LLaMA-3), Streamlit, BeautifulSoup.
