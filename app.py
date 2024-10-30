from bs4 import BeautifulSoup
from chroma_db import D4EmailChromaDb
from langchain import hub
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

import requests
import textwrap
import gradio as gr
import os


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class EmailResponder:

    D4_URL = "https://www.denvergov.org/Government/Agencies-Departments-Offices/Agencies-Departments-Offices-Directory/Denver-City-Council/Council-Members-Websites-Info/District-4"

    TOPICS_TO_311_DICT = {
        "Homeless": {
            'link': "Encampment Reporting: https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "You will need to report the issue through 311. To do this you'll need to answer the following questions: Are there any children present? How long has the encampment been there? How many tents/structures are present? Are needles, feces or trash present?",
        },
        "Graffiti": {
            'link': "Graffiti: https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "You will need to report the issue through 311. To do this you'll need to answer the following questions: Is the property public, a park, private, or RTD property? Where is the graffiti located? Is graffiti higher than the first floor? Are you the owner or tenant of this property? Is graffiti profane or racist? What is tagged? Is there any identifying features? And if it's on RTD property, please provide as much of the following as possible: the 5 digit bus stop number, the route number, bench number, and direction of travel for the bus.",
        },
        "Pothole": {
            'link': "Pothole: https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "You will need to report the issue through 311. To do this you'll need to answer the following questions: Is the pothole located in an alley, gutter or street? What is the pothole surface? What is the direction of travel of the side it's located on, North, South, East, West, or in an alley? What lane? Can you see the bottom? You'll also need to describe, specifically, the level of damage.",
        },
        "Animal": {
            'link': "Animal Complaint: https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "For this complaint, you just need to describe the issue with as much detail as you can provide. Picture evidence will also strengthen your case.",
        },
        "Vegetation": {
            'link': "Weeds and Vegetation: https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "You will need to report the issue through 311. To do this you'll need to answer the following questions: Where on property is the violation? Is the violation outdoor storage, trash, vegetation, or something else?",
        },
        "Neighborhood": {
            'link': "Neighborhood Issue: https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "For this complaint, you just need to describe the issue with as much detail as you can provide. Picture evidence will also strengthen your case.",
        },
        "Other": {
            'link': "Other: https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "For this complaint, you just need to describe the issue with as much detail as you can provide. Picture evidence will also strengthen your case.",
        },
        "Snow Removal": {
            'link': "Snow on Sidewalk: https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "You will need to report the issue through 311. To do this you'll need to answer the following questions: Is the property business or residential?",
        },
        "Vehicle": {
            'link': "Abandoned Vehicle: https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "You will need to report the issue through 311. To do this you'll need to answer the following questions: Is the property public or private? What is the exact location of the vehicle? What is the plate number? What state is the plate from?",
        },
        "Parking": {
            'link': "Illegal Parking: https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "You will need to report the issue through 311. To do this you'll need to answer the following questions: Is the vehicle blocking a driveway? How long has vehicle been parked? What is the plate number? What state is the plate from? What is the color, make, and style of vehicle? What type of vehicle is it?",
        },
        "Police": {
            'link': "Police (Non-Emergency): https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "You will need to report the issue through 311. To do this you'll need to answer the following questions: You will need to confirm that this is, in fact, a non-emergency before proceeding to describe the issue with as much detail as you can provide. Picture evidence will also strengthen your case.",
        },
        "Fireworks": {
            'link': "Fireworks: https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "You will need to report the issue through 311. To do this you'll need to answer the following questions: What was the date of the occurence? At what time did this happen? What did you see or hear?",
        },
        "Dumping": {
            'link': "Illegal Dumping: https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "You will need to report the issue through 311. To do this you'll need to answer the following questions: What is the color, make, and style of vehicle? What is the plate number? What state is the plate from? At what time did this happen? What type of items are being dumped? Describe the person doing the disposal.",
        },
        "Trash": {
            'link': "Missed Trash Pickup: https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "You will need to report the issue through 311. To do this you'll need to answer the following questions: Did you receive a notice on your cart? What time were your carts/large items set out? Are your carts/large items out and accessible right now? Was there anything blocking access to the cart/items? What service was not collected?",
        },
        "Tree": {
            'link': "Damaged/Fallen Tree: https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "You will need to report the issue through 311. To do this you'll need to answer the following questions: What size is the branch? Where is it located exactly? Is tree blocking street access or right of way? And if a street sign is blocked, is it partial or total obstruction?",
        },
        "Micromobility": {
            'link': "Shared Micromobility: https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "You will need to report the issue through 311. To do this you'll need to answer the following questions: What is the scooter/bike ID number (and/or color)? Is the scooter/bike laying on the ground (not standing)? Is the scooter/bike allowing 5ft of pedestrian clearance? Is the scooter/bike allowing 4ft of utilities clearance? Is the scooter/bike parked within 1.5ft of the curb or building? Is the scooter/bike blocking sight triangle near an intersection/alley/driveway? Is the scooter/bike damaged? If so describe the damage.",
        },
        "Utilities": {
            'link': "No Heat No Water No Electricity: https://denvergov.org/Online-Services-Hub/Report-an-Issue/issue/description",
            'response': "You will need to report the issue through 311. To do this you'll need to answer the following questions: Has management/landlord/property manager been notified? Is this a home, apartment, condo, townhouse, motel or hotel? Are you the owner, tenant or other? What is the unit or room number? Please disclose if you have any pets or weapons in the home, they will need to be secured if an investigation is scheduled.",
        },
    }

    RAG_TEMPLATE_TEXT_GENERIC = textwrap.dedent("""
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    """)

    RAG_TEMPLATE_TEXT_D4_SPECIFIC = textwrap.dedent("""
    You are an assistant for question-answering tasks specifically for generating responses to constituent emails.
    You work as a senior aide to Councilwoman Diana Romero Campbell, a Denver City Council member for District 4.
    You represent the South East Region of the city and county of Denver Colorado USA.
    Use the following pieces of retrieved context to help answer the question and generate a response to the constituent.
    If you don't know the answer, just say that you don't know but we will get the information and get back to you.
    Use three to four sentences maximum, keep the answer concise, and be specific to the city and county of Denver.

    Please concluder your response with the following format:
    Thank you,
    Office of Councilwoman Diana Romero Campbell, District 4.
    """)

    RAG_TEMPLATE_TEXT_SFX = textwrap.dedent("""

    Question: {question}
    Context: {context}
    Answer:
    """)

    TEMPLATE_TEXT_DICT = {
        "D4 Specific Prompt": RAG_TEMPLATE_TEXT_D4_SPECIFIC + RAG_TEMPLATE_TEXT_SFX,
        "Generic Prompt": RAG_TEMPLATE_TEXT_GENERIC + RAG_TEMPLATE_TEXT_SFX,
    }

    # Define the topics and their descriptions
    TOPICS_PROMPT_DICT = {
        'Homeless': 'words similar to homeless, shelter, encampment',
        'Graffiti': 'words similar to graffiti, paint, tagging',
        'Pothole': 'words similar to pothole, holes',
        'Animal': 'words similar to animal, pest, pets, barking',
        'Vegetation': 'words similar to weeds, trees, limbs, overgrown',
        'Neighborhood': 'words similar to HOA, RNO, sidewalk, fence',
        'Snow Removal': 'words similar to snow, ice, plows',
        'Vehicle': 'words similar to vehicle, car, motorcycle, automobile',
        'Parking': 'words similar to parking',
        'Police': 'words similar to police, gang, loud, drugs, crime',
        'Fireworks': 'words similar to fireworks',
        'Dumping': 'words similar to dumping',
        'Trash': 'words similar to trash, garbage, compost, recycling',
        'Housing': 'words similar to rent, rental, apartments, housing',
        'Policy': 'words similar to policy, tax, taxes, mayor, council, environmental, environment, rezoning, rezone, government, politics',
        'Street Racing': 'words similar to racing',
        'Transit': 'words similar to transit, traffic, pedestrian, intersection, bicycle, bike, speed, pavement',
        'Parks': 'words similar to park, playground, trails, pool, gym, medians',
    }

    # Convert the dictionary to a string
    TOPIC_DESCRIPTIONS = "\n".join(
        [f"{key}: {desc}" for key, desc in TOPICS_PROMPT_DICT.items()])

    # Define the topics prompt
    TOPICS_PROMPT = PromptTemplate(
        input_variables=["email", "sentiment"],
        output_key="topics",
        template=f"""
        You are an email classification assistant. Based on the content of the email, please classify it into one or more of the following topics:

        {TOPIC_DESCRIPTIONS}

        Email content:
        {{email}}

        Please list the relevant topics based on the email content. If multiple topics apply, separate them with commas. Only return the topic names.
        If you do not know the topic, return 'Other'.

        Format:
        <your topics here>
        """
    )

    FUN_FACT_PROMPT = PromptTemplate(
        input_variables=["topics"],
        output_key="fun_fact",
        template="""
        Please find a random fun fact about the city of Denver related to any one of the following topics:
        {topics}
        Keep your response short and to the point.
        """
    )

    def __init__(self, embedding_function, persist_directory):
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.vector_store = Chroma(
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
        )
        self.rag_prompt_text_choices: list[str] = [
            "D4 Specific Prompt", "Generic Prompt",]
        self.gpt_model_choices: list[str] = [
            "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"]
        self.current_rag_prompt_text_choice: str = self.rag_prompt_text_choices[0]
        self.current_gpt_model_choice: str = self.gpt_model_choices[0]
        self.current_gpt_model_temp: float = 1
        self.llm = self.create_gpt_llm(
            self.current_gpt_model_choice, self.current_gpt_model_temp)
        self.retriever: VectorStoreRetriever = self.create_retriever()
        self.rag_prompt = self.create_generic_rag_prompt()
        self.update_rag_prompt_text(self.current_rag_prompt_text_choice)
        self.rag_chain = self.create_rag_cahin()
        self.sentiment_chain = self.create_sentiment_chain()
        self.topic_and_fun_fact_chain = self.create_topic_and_fun_fact_chain()

    def create_retriever(self) -> VectorStoreRetriever:
        """
        Create a retriever that uses cosine similarity to search the vector store.

        The retrieved documents are the ones that are most similar to the input query.
        The search results are returned as a list of Document objects, sorted by
        relevance, with the most relevant documents first.

        Returns:
            A VectorStoreRetriever object
        """
        return self.vector_store.as_retriever()

    def create_generic_rag_prompt(self):
        """
        Create a generic prompt template to perform simple queries on the vector store.
        Uses the predefined "rlm/rag-prompt" prompt template from the LangChain hub.

        Returns:
            A LLMChainPromptTemplate object with a generic prompt template
        """
        return hub.pull("rlm/rag-prompt")

    def update_rag_prompt_text(self, prompt_template_choice: str):
        """
        Update the prompt template to use when generating text responses.

        Args:
            prompt_template_choice (str): The name of the prompt template to use.
                Must be one of the keys in the TEMPLATE_TEXT_DICT dictionary.

        Returns:
            None
        """
        self.rag_prompt[0].prompt.template = self.TEMPLATE_TEXT_DICT[prompt_template_choice]

    def create_gpt_llm(self, model_name: str = "gpt-4o-mini", temperature: float = 1):
        """
        Create a GPT-based LLM for generating text responses.

        Args:
            model_name (str): The name of the GPT model to use. Defaults to "gpt-4o-mini".
            temperature (float): The temperature to use when generating text. Defaults to 1.

        Returns:
            A ChatOpenAI object that can be used to generate text responses
        """
        return ChatOpenAI(model=model_name, temperature=temperature)

    def create_sentiment_chain(self):
        # Define a sentiment prompt that returns a sentiment response
        sentiment_prompt = PromptTemplate(
            input_variables=["email"],
            output_key="sentiment",
            template="""
            Given the following email text:
            {email}

            Please provide a sentiment analysis with a response value of 'Negative', 'Neutral', or 'Positive'.

            Format:
            <your sentiment here>
            """
        )
        return sentiment_prompt | self.llm | StrOutputParser()

    def create_topic_and_fun_fact_chain(self):
        topic_chain = LLMChain(
            llm=self.llm, prompt=self.TOPICS_PROMPT, output_key="topics")
        fun_fact_chain = LLMChain(
            llm=self.llm, prompt=self.FUN_FACT_PROMPT, output_key="fun_fact")
        sequential_chain = SequentialChain(
            chains=[topic_chain, fun_fact_chain],
            input_variables=["email"],  # Initial input needed
            # Final output of the chain
            output_variables=["topics", "fun_fact"]
        )
        return sequential_chain

    def create_rag_cahin(self):
        # Define the chain
        rag_chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def get_topic_311_details(self, topics):
        topic_311_details = []
        for keyword, details_dict in self.TOPICS_TO_311_DICT.items():
            if keyword in topics:
                topic_311_details.append(textwrap.dedent(
                    f"""Topic: {keyword}\nLink: {details_dict['link']}\nDetails: {details_dict['response']}

                    """
                ))
        return "".join(topic_311_details).rstrip()

    def get_html_summaries(self, topics: str):
        url = self.D4_URL
        keywords = [topic.strip() for topic in topics.split(',')]
        try:
            # Fetch and parse the page content
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find relevant sections and generate summaries
            relevant_articles = {}
            for keyword in keywords:
                relevant_articles[keyword] = []
                for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'a']):
                    if keyword.lower() in element.get_text(strip=True).lower():
                        # Capture the parent section for full context
                        section = element.find_parent()
                        full_text = section.get_text(
                            strip=True, separator="\n") if section else element.get_text(strip=True)

                        # Generate a short summary (first 1-2 sentences)
                        summary = '. '.join(full_text.split('. ')[:2]) + '...'

                        # Add link if it's a URL element
                        link = url if element.name != 'a' else element.get('href', url)

                        # Avoid duplicates
                        if summary not in [item['summary'] for item in relevant_articles[keyword]]:
                            relevant_articles[keyword].append({'summary': summary, 'link': link})

            # Format the output with summaries and links
            display_output = "### Summaries with Links to Full Articles:\n\n"
            for keyword, articles in relevant_articles.items():
                display_output += f"**Keyword: {keyword.capitalize()}**\n\n"
                for i, article in enumerate(articles, 1):
                    display_output += f"{i}. {article['summary']} "
                    display_output += f"[Read More]({article['link']})\n\n"
                    display_output += "---\n"  # Separator between summaries

                if not articles:
                    display_output += f"No articles found for **{keyword}**.\n\n"

            return display_output

        except requests.RequestException as e:
            return f"An error occurred: {e}"

    def generate_response(
            self,
            email,
            prompt_template_choice: str,
            gpt_model_choce: str,
            gpt_model_temp: float):
        """
        Generate a response to the given email.

        Args:
            email (str): The text of the email to generate a response to.

        Returns:
            str: The generated response text.
        """
        if prompt_template_choice != self.current_rag_prompt_text_choice:
            self.current_rag_prompt_text_choice = prompt_template_choice
            self.update_rag_prompt_text(prompt_template_choice)
        if gpt_model_choce != self.current_gpt_model_choice or gpt_model_temp != self.current_gpt_model_temp:
            self.current_gpt_model_choice = gpt_model_choce
            self.current_gpt_model_temp = gpt_model_temp
            self.llm = self.create_gpt_llm(
                model_name=gpt_model_choce, temperature=gpt_model_temp)

        # Invoke all of the chains
        response = self.rag_chain.invoke(email)
        sentiment = self.sentiment_chain.invoke(email)
        results = self.topic_and_fun_fact_chain(email)
        topics = results["topics"]
        topic_details = self.get_topic_311_details(topics)
        fun_fact = results["fun_fact"]
        html_summary = self.get_html_summaries(topics)

        return (sentiment, topics, response, topic_details, fun_fact, html_summary)

    # Define a function to format the documents retrieved from the vector store
    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)


# Define the gradio interface
def create_gradio_app(responder: EmailResponder) -> gr.Blocks:
    """
    Create a Gradio interface that wraps the given EmailResponder object.

    Args:
        responder: The EmailResponder object to wrap.

    Returns:
        A Gradio Blocks object that can be launched with app.launch()
    """

    example_emails = [
        textwrap.dedent("""
        The lack of police presence and code enforcement is sending a growing message that these violations
        are not important. Second item: affordable denver and wanting more information about how the tax will
        accomplish the goals set by Mayor.
        """).strip(),
        "I want information on getting a compost bin. I have submitted case number: 9578014",
        "I saw a downed tree at the intersection of Yale and Clayton. One of the limbs is blocking the street."
    ]

    # Define examples with all inputs specified for each example
    examples = [
        [example_emails[0], "D4 Specific Prompt", "gpt-4o-mini", 0.1],
        [example_emails[0], "Generic Prompt", "gpt-4o-mini", 1.0],
        [example_emails[1], "D4 Specific Prompt", "gpt-4o-mini", 1.0],
        [example_emails[1], "Generic Prompt", "gpt-4o-mini", 0.3],
        [example_emails[2], "D4 Specific Prompt", "gpt-4o-mini", 0.6],
    ]

    def handle_thumbs_up(sentiment, topics, response):
        return f"Positive feedback received for:\n{response}"

    def handle_thumbs_down(sentiment, topics, response):
        return f"Negative feedback received for:\n{response}"

    with gr.Blocks() as email_app:
        gr.Markdown("# Denver City Council District 4 Email Assistant")
        gr.Markdown(
            "Enter a constituent email and the app will generate a sample response.")

        with gr.Row():
            with gr.Column():
                # Define input components for Column 1
                email_input = gr.Textbox(
                    label="Constituent Email",
                    placeholder="Enter email here..."
                )
                prompt_template = gr.Dropdown(
                    choices=responder.rag_prompt_text_choices,
                    value=responder.current_rag_prompt_text_choice,
                    label="Prompt Template",
                )
                gpt_model = gr.Dropdown(
                    choices=responder.gpt_model_choices,
                    value=responder.current_gpt_model_choice,
                    label="GPT Model",
                )
                specificity_scale = gr.Slider(
                    value=responder.current_gpt_model_temp,
                    label="D4 Scale: 0 more specific - 1 less specific",
                    minimum=0,
                    maximum=1,
                    step=0.1
                )

                # Place the Generate Response button under the inputs
                generate_button = gr.Button("Generate Response")

            with gr.Column():
                # Define output components for Column 2
                sentiment_output = gr.Textbox(
                    label="Sentiment:",
                    placeholder="The sentiment of the email will show here..."
                )
                topics_output = gr.Textbox(
                    label="Topics:",
                    placeholder="Email topics will show here..."
                )
                response_output = gr.Textbox(
                    label="Sample response:",
                    placeholder="Generated response will show here..."
                )
                topic_311_details_output = gr.Textbox(
                    label="311 Topic Specific Details",
                    placeholder="Topic based 311 details will show here..."
                )
                fun_fact_output = gr.Textbox(
                    label="Denver Fun Fact:",
                    placeholder="Random Denver fun fact will show here..."
                )
                html_summary_output = gr.Textbox(
                    label="Additional Denvergov.org Info",
                    placeholder="Any additional Denvergov.org info will show here..."
                )

                # Thumbs up and thumbs down buttons
                with gr.Row():
                    thumbs_up = gr.Button("👍 Thumbs Up")
                    thumbs_down = gr.Button("👎 Thumbs Down")

                feedback_output = gr.Textbox(
                    label="Feedback:",
                    placeholder="Feedback on the response will show here...",
                    interactive=False
                )

        # Display examples
        gr.Examples(
            examples=examples,
            inputs=[email_input, prompt_template,
                    gpt_model, specificity_scale],
        )

        # Connect generate button to responder function
        generate_button.click(
            fn=responder.generate_response,
            inputs=[email_input, prompt_template,
                    gpt_model, specificity_scale],
            outputs=[sentiment_output, topics_output,
                     response_output, topic_311_details_output,
                     fun_fact_output, html_summary_output]
        )

        # Connect thumbs up and thumbs down buttons to feedback functions
        thumbs_up.click(
            fn=handle_thumbs_up,
            inputs=[sentiment_output, topics_output, response_output],
            outputs=feedback_output
        )
        thumbs_down.click(
            fn=handle_thumbs_down,
            inputs=[sentiment_output, topics_output, response_output],
            outputs=feedback_output
        )

    return email_app


# Run the application
if __name__ == "__main__":
    # Define variables
    d4_emails_file = './resources/d4_emails_topics.csv'
    d4_emails_responses_file = './resources/d4_emails_responses.csv'
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_directory = "chroma_db"

    # Initialize the vector store everytime at startup b/c we can't save it to the huggingface repo
    chroma = D4EmailChromaDb(csv_file_in=d4_emails_file,
                             csv_file_out=d4_emails_responses_file)
    chroma.init_vector_store(embedding_function, persist_directory)

    # Initialize the email responder and run the gradio app
    responder = EmailResponder(embedding_function, persist_directory)
    email_app = create_gradio_app(responder)
    email_app.launch()
