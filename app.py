from langchain import hub
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from chroma_db import D4EmailChromaDb
import textwrap
import gradio as gr
import os


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class EmailResponder:

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
        'Vegitation': 'words similar to weeds, trees, limbs, overgrown',
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
        self.rag_prompt_text_choices: list[str] = ["D4 Specific Prompt", "Generic Prompt",]
        self.gpt_model_choices: list[str] = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"]
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
        fun_fact = results["fun_fact"]

        return (sentiment, topics, response, fun_fact)

    # Define a function to format the documents retrieved from the vector store
    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)


# Define the gradio interface
def create_gradio_app(responder: EmailResponder) -> gr.Interface:
    """
    Create a Gradio interface that wraps the given EmailResponder object.

    This interface has a single text box input for the constituent email
    and a single text box output for the generated response.
    The interface also includes a title, description, and examples.

    Args:
        responder: The EmailResponder object to wrap.

    Returns:
        A Gradio Interface object that can be launched with app.launch()
    """

    example_emails = [
        textwrap.dedent("""
            The lack of police presence and code enforcement is sending a growing message that these violations
            are not important. Second item: affordable denver and wanting more information about how the tax will
            accomplish the goals set by Mayor.
        """),
        "I want information on getting a compost bin. I have submitted case number: 9578014",
        "I saw a downed tree at the intersection of Yale and Clayton. One of the limbs is blocking the street."
    ]

    # Define examples with all inputs specified for each example
    examples = [
        [example_emails[0], "D4 Specific Prompt", "gpt-4o-mini", 0.3],
        [example_emails[0], "Generic Prompt", "gpt-4o-mini", 1.0],
        [example_emails[1], "D4 Specific Prompt", "gpt-4o-mini", 1.0],
        [example_emails[1], "Generic Prompt", "gpt-4o-mini", 0.3],
        [example_emails[2], "D4 Specific Prompt", "gpt-4o-mini", 0.6],
    ]

    email_app = gr.Interface(
        responder.generate_response,
        [
            gr.Textbox(
                label="Constituent Email",
                placeholder="Enter email here..."
            ),
            gr.Dropdown(
                choices=responder.rag_prompt_text_choices,
                value=responder.current_rag_prompt_text_choice,
                label="Prompt Template",
            ),
            gr.Dropdown(
                choices=responder.gpt_model_choices,
                value=responder.current_gpt_model_choice,
                label="GPT Model",
            ),
            gr.Slider(
                value=responder.current_gpt_model_temp,
                label="D4 Specificity Scale - 0 is most specific, 1 is least specific",
                minimum=0,
                maximum=1,
                step=0.1
            ),
        ],
        [
            gr.Textbox(
                label="Sentiment:",
                placeholder="The sentiment of the email will show here..."
            ),
            gr.Textbox(
                label="Topics:",
                placeholder="Email topics will show here..."
            ),
            gr.Textbox(
                label="Sample response:",
                placeholder="Generated response will show here..."
            ),
            gr.Textbox(
                label="Denver Fun Fact:",
                placeholder="Random Denver fun fact will show here..."
            ),
        ],
        title="Denver City Council District 4 Email Assistant",
        description="Enter a constituent email and the app will generate a sample response.",
        examples=examples,
        cache_examples=False
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
