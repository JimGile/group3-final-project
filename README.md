# group3-final-project
# Email Responder for Denver City Council District 4

## Overview
This application is an AI-powered email responder designed to assist Denver City Council District 4 in managing constituent communications. It leverages the LangChain framework and OpenAI's language models to generate context-specific responses, additional resources about the issue, and details on next steps for resolution, and a feedback loop to improve answers over time; helping streamline responses to community inquiries quickly and efficiently.

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

## Team Members
- Anna Fine
- Carl Peterson
- Jim Gile
- Tim Willard
