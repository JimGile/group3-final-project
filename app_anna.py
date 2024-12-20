from langchain import hub
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI
from chroma_db import D4EmailChromaDb
import textwrap
import gradio as gr
import os


# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load the API key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
print(f"API key loaded: {OPENAI_API_KEY}")


class EmailResponder:

    TOPICS_311_DICT = {
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


    TEMPLATE_TEXT_GENERIC = textwrap.dedent("""
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    """)

    TEMPLATE_TEXT_D4_SPECIFIC = textwrap.dedent("""
    You are an assistant for question-answering tasks specifically for generating responses to constituent emails.
    You work as a senior aide to Councilwoman Diana Romero Campbell, a Denver City Council member for District 4.
    You represent the South East Region of the city and county of Denver Colorado USA.
    Use the following pieces of retrieved context to help answer the question and generate a response to the constituent.
    If you don't know the answer, just say that you don't know but we will get the information and get back to you.
    Use three to four sentences maximum, keep the answer concise, and be specific to the city and county of Denver.
    In your response, please include a fun fact about the city of Denver.
    """)

    TEMPLATE_TEXT_SFX = textwrap.dedent("""
    Question: {question}
    Context: {context}
    Answer:
    """)

    TEMPLATE_TEXT_DICT = {
        "D4 Specific Prompt": TEMPLATE_TEXT_D4_SPECIFIC + TEMPLATE_TEXT_SFX,
        "Generic Prompt": TEMPLATE_TEXT_GENERIC + TEMPLATE_TEXT_SFX,
    }

    def __init__(self, embedding_function, persist_directory):
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.vector_store = Chroma(
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
        )
        self.prompt_text_choices: list[str] = ["D4 Specific Prompt", "Generic Prompt",]
        self.gpt_model_choices: list[str] = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"]
        self.current_prompt_text_choice: str = self.prompt_text_choices[0]
        self.current_gpt_model_choice: str = self.gpt_model_choices[0]
        self.current_gpt_model_temp: float = 1
        self.retriever: VectorStoreRetriever = self.create_retriever()
        self.prompt = self.create_generic_prompt()
        self.update_prompt_text(self.current_prompt_text_choice)
        self.llm = self.create_gpt_llm(self.current_gpt_model_choice, self.current_gpt_model_temp)
        self.rag_chain = self.create_rag_cahin()

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

    def create_generic_prompt(self):
        """
        Create a generic prompt template to perform simple queries on the vector store.
        Uses the predefined "rlm/rag-prompt" prompt template from the LangChain hub.

        Returns:
            A LLMChainPromptTemplate object with a generic prompt template
        """
        return hub.pull("rlm/rag-prompt")

    def update_prompt_text(self, prompt_template_choice: str):
        """
        Update the prompt template to use when generating text responses.

        Args:
            prompt_template_choice (str): The name of the prompt template to use.
                Must be one of the keys in the TEMPLATE_TEXT_DICT dictionary.

        Returns:
            None
        """
        self.prompt[0].prompt.template = self.TEMPLATE_TEXT_DICT[prompt_template_choice]

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

    def create_rag_cahin(self):
        # Define the chain
        rag_chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain
    
    def classify_issue(self, topics):
        keywords = [topic.strip() for topic in topics.split(',')]
        issues_responses = []
        for keyword, details_dict in self.TOPICS_311_DICT.items():
            if keyword in topics.lower():
                issues_responses.append(
                    {
                        "category": keyword,
                        "link": details_dict['link'],
                        "response": details_dict['response']
                    }
                )
        return issues_responses

    def generate_response(self, email, prompt_template_choice: str, gpt_model_choce: str, gpt_model_temp: float):
        """
        Generate a response to the given email.

        Args:
            email (str): The text of the email to generate a response to.

        Returns:
            str: The generated response text.
        """
        if prompt_template_choice != self.current_prompt_text_choice:
            self.current_prompt_text_choice = prompt_template_choice
            self.update_prompt_text(prompt_template_choice)
        if gpt_model_choce != self.current_gpt_model_choice or gpt_model_temp != self.current_gpt_model_temp:
            self.current_gpt_model_choice = gpt_model_choce
            self.current_gpt_model_temp = gpt_model_temp
            self.llm = self.create_gpt_llm(model_name=gpt_model_choce, temperature=gpt_model_temp)

        return self.rag_chain.invoke(email)

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
        "I want information on getting a compost bim. I have submitted case number: 9578014",
    ]

    # Define examples with all inputs specified for each example
    examples = [
        [example_emails[0], "D4 Specific Prompt", "gpt-4o-mini", 0.3],
        [example_emails[0], "Generic Prompt", "gpt-4o-mini", 1.0],
        [example_emails[1], "D4 Specific Prompt", "gpt-4o-mini", 1.0],
        [example_emails[1], "Generic Prompt", "gpt-4o-mini", 0.3],
    ]    

    email_app = gr.Interface(
        responder.generate_response,
        [
            gr.Textbox(
                label="Constituent Email",
                placeholder="Enter email here..."
            ),
            gr.Dropdown(
                choices=responder.prompt_text_choices,
                value=responder.current_prompt_text_choice,
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
                label="Sample response:",
                placeholder="Generated response will show here..."
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
