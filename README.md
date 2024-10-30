# group3-final-project
# Email Responder for Denver City Council District 4

## Overview
This application is an AI-powered email responder designed to assist Denver City Council District 4 in managing constituent communications. It leverages the LangChain framework and OpenAI's language models to generate context-specific responses, additional resources about the issue, details on next steps for resolution, and a feedback loop to improve answers over time; helping streamline responses to community inquiries quickly and efficiently.

## Features
- **Automated Email Responses**: Uses AI to generate concise, context-specific responses to constituent emails.
- **Customizable Templates**: Two main response templates are provided, one for general inquiries and one specific to District 4.
- **Chroma Database Integration**: Manages context retrieval for question answering using Chroma.
- **Webscrapping Additional Info**: Provides additioal information from devergov.org that can be used as suplimentary info to response
- **Library of Necessary Next Steps**: Additional issue specific info to help constituents file formal isssues with Denver 311
- **RLHF (Reinforcement Learning from Human Feedback)**: Recieve real-time feedback from the user to improve answers ovver time and usage.

## Project Structure
- **`EmailResponder` Class**: Central class that manages different response templates and context retrieval.
- **Templates**: Includes general-purpose and District 4-specific templates to tailor responses.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/JimGile/group3-final-project.git
2.  **Install dependencies**:
    ```bash
    pip install -r requirements_app.txt
3.  **Set-up API Key
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"

## Usage
1.  Run the application: The application uses Gradio for the interface. To start the app, use:
    ```bash
    python app.py

## Additional Notebooks
**load_vectorstore_docs.ipynb**<br>
This notebook prepares email data specifically for integration into a vector database, enabling efficient search and retrieval of relevant information.

Key Steps
1. Data Preprocessing:
- Loads and cleans email data by removing unnecessary columns (name, email_address, d4_staff_member, etc.).
- Outputs a refined CSV file for use in the vector store.
2. Text Splitting:
- Chunks email content into manageable segments for better embedding and retrieval.
3. Embedding Creation:
- Uses OpenAI embeddings to convert text chunks into vector format for efficient similarity search.
4. Vector Store Population:
- Utilizes the Chroma database to store the vectorized data, making it accessible for rapid query-based retrieval.

Prerequisites<br>
LangChain and Chroma dependencies should be installed, along with an OpenAI API key configured in the environment.

Running the Notebook<br>
Execute the notebook cells sequentially to preprocess, vectorize, and store emails into the Chroma vector database.

**topic_modeling.ipynb**<br>
This notebook conducts topic modeling on constituent emails to identify recurring themes and key issues that constituents raise. By analyzing the content of emails, this model aids in understanding prevalent topics, helping to streamline responses and identify priority areas.

Key Steps
1. Data Loading and Preprocessing:
- Imports the dataset containing constituent emails and applies initial data cleaning.
- Prepares the data for topic modeling, ensuring relevant fields are included.
2. Topic Modeling:
- Uses Latent Dirichlet Allocation (LDA) from Scikit-Learn to perform topic modeling.
- Extracts main topics across constituent emails, highlighting key issues.
3. Natural Language Processing (NLP) Enhancements:
- Utilizes the SpaCy library for advanced text processing, including tokenization and lemmatization, to improve topic accuracy.
4. Result Visualization:
- Visualizes topics and common words for each identified theme, providing insights into constituent concerns.

Prerequisites<br>
Scikit-Learn and SpaCy should be installed, with SpaCy’s English language model (en_core_web_sm) available.

Running the Notebook<br>
Execute each cell sequentially to load data, preprocess text, perform topic modeling, and visualize results. The insights from this notebook can guide targeted responses and resource allocation for common community issues.

**topic_modeling.ipynb**<br>
This notebook uses topic modeling techniques to analyze constituent emails, identifying common themes and areas of concern. It applies natural language processing to classify and group email content, aiding in understanding key topics for better engagement and resource allocation.

Key Steps
1. Data Import and Preparation:
- Loads email data into a structured format and cleans it for analysis.
- Prepares fields relevant to topic modeling, focusing on message content.
2. Text Processing:
- Tokenizes, lemmatizes, and pre-processes text using SpaCy’s English language model for improved topic modeling accuracy.
3. Latent Dirichlet Allocation (LDA) Topic Modeling:
- Employs LDA to identify clusters of themes across constituent emails.
- Extracts and categorizes topics, highlighting recurring issues.
4. Topic Visualization:
- Visualizes the identified topics and their top words, providing insights into community priorities.

Prerequisites<br>
Install Scikit-Learn and SpaCy with the en_core_web_sm language model for full functionality.

Usage<br>
Run the notebook sequentially to load, process, and analyze email data, ultimately identifying and visualizing main topics to inform data-driven decision-making.

**sample_rag_chain.ipynb**<br>
This notebook sets up a Retrieval-Augmented Generation (RAG) chain to answer questions using relevant context from a pre-built vector database. It’s designed to facilitate precise, context-aware responses to user queries, leveraging the power of language models and vector stores for enhanced information retrieval.

Key Steps
1. Environment and Model Setup:
- Configures OpenAI embeddings and the Chroma vector store, loading the necessary environment variables (e.g., OPENAI_API_KEY).
- Loads the RAG prompt template from LangChain Hub.
2. Vector Store Integration:
- Connects to an existing Chroma vector store, which holds the embedded documents for context retrieval.
- Uses a retriever to fetch context relevant to a given query, based on similarity matching.
3. RAG Chain Definition:
- Establishes a RAG chain that combines the retrieved context and query with a pre-defined prompt template.
- Passes the formatted context and query to the language model (e.g., GPT-4) to generate concise and contextually informed responses.
4. Prompt and Model Execution:
- Executes the RAG chain by providing a question, retrieving relevant context from the vector store, and generating a concise answer.

Prerequisites<br>
LangChain, Chroma, and OpenAI API access are required.
Ensure the Chroma vector store is pre-populated with relevant embeddings.

Usage<br>
To run the notebook:
- Load the cells sequentially, setting up environment variables and verifying vector store connectivity.
- Input a query to see the RAG chain retrieve relevant context and generate a response.

## Team Members
- Anna Fine
- Carl Peterson
- Jim Gile
- Tim Willard
