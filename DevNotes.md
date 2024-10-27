# Developer notes

## Python Requirements

Steps to create the requirements.txt file

1. activate the virtual environment
2. pip freeze > requirements.txt

Steps to install the requirements.txt file

1. activate the virtual environment
2. pip install -r requirements.txt

## Gradio and Hugging Face Secrets

To securely use an API key in a Gradio app hosted on Hugging Face Spaces, you can use Hugging Face Secrets. Hugging Face Secrets allow you to store sensitive information, like API keys, without hardcoding them directly in your code. These secrets will only be accessible to your app and will not be exposed publicly.

Steps to Upload and Use an API Key in a Gradio App on Hugging Face:

1. Add the API Key as a Hugging Face Secret
   - Go to your Hugging Face Spaces dashboard.
   - Select the space where your Gradio app is hosted.
   - On the space's page, click the Settings tab.
   - Scroll down to the Secrets section.
   - Add a new secret with the name of your choice (e.g., MY_API_KEY) and paste your API key in the value field.
   - Save the secret.

2. Access the API Key in Your Gradio App

   In your Gradio app code, you can retrieve the secret using the os.getenv() function from Python’s os module. This function will fetch the environment variable you set in the Hugging Face Secrets.

        Here's an example of how to do this:

        ```python
        import os
        import gradio as gr

        # Access the API key from Hugging Face Secrets
        api_key = os.getenv("MY_API_KEY")

        # Function that uses the API key
        def call_api(input_text):
            if api_key is None:
                return "API key not found"

            # Example API call logic (replace with actual API call)
            response = f"API key used: {api_key}, input was: {input_text}"
            return response

        # Gradio app
        demo = gr.Interface(
            fn=call_api,
            inputs="text",
            outputs="text",
            title="API Key Example"
        )

        demo.launch()
        ```

3. Deploy the Gradio App to Hugging Face

    After you’ve updated your app to use the API key via os.getenv(), commit your changes and push the code to your Hugging Face Space repository.

    Hugging Face Spaces will automatically read the secret and set it as an environment variable. Your Gradio app will now have access to the API key securely.

### Key Points

- Secrets: Hugging Face Secrets store sensitive data such as API keys, ensuring they are not exposed in the source code.
- Environment Variables: You access the secrets using os.getenv('SECRET_NAME'), which is a standard way to handle environment variables in Python.
- Security: The API key will not be visible in your code or the app interface and is securely stored by Hugging Face.
