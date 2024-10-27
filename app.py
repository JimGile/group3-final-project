from langchain import hub
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI
from chroma_db import D4EmailChromaDb
import gradio as gr
import os


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class EmailResponder:
    def __init__(self, embedding_function, persist_directory):
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.vector_store = Chroma(
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
        )
        self.retriever = self.create_retriever()
        self.prompt = self.create_standard_prompt()
        self.llm = self.create_gpt_llm()
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

    def create_standard_prompt(self):
        """
        Create a standard prompt template to perform simple queries on the vector store.
        Uses the predefined "rlm/rag-prompt" prompt template from the LangChain hub.

        Returns:
            A LLMChainPromptTemplate object with a standard prompt template
        """
        return hub.pull("rlm/rag-prompt")

    def create_gpt_llm(self, model_name: str = "gpt-4", temperature: float = 1):
        """
        Create a GPT-based LLM for generating text responses.

        Args:
            model_name (str): The name of the GPT model to use. Defaults to "gpt-4".
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

    def generate_response(self, email):
        """
        Generate a response to the given email.

        Args:
            email (str): The text of the email to generate a response to.

        Returns:
            str: The generated response text.
        """

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

    email_app = gr.Interface(
        responder.generate_response,
        [
            gr.Textbox(
                label="Constituent Email",
                placeholder="Enter email here..."
            ),
        ],
        [
            gr.Textbox(
                label="Sample response:",
                placeholder="Generated response will show here..."
            ),
        ],
        title="Denver City Council District 4 Email Agent",
        description="Enter a constituent email and the app will generate a sample response.",
        examples=[
            """The lack of police presence and code enforcement is sending a growing message that these violations
            are not important. Second item: affordable denver and wanting more information about how the tax will
            accomplish the goals set by Mayor.""",
            "I want information on getting a compost bim. I have submitted case number: 9578014.",
        ],
        cache_examples=False
    )
    return email_app


# Run the application
if __name__ == "__main__":
    # Define variables
    d4_emails_file = './resources/d4_emails_topics.csv'
    d4_emails_responses_file = './resources/d4_emails_responses.csv'
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
    persist_directory = "chroma_db"

    # Initialize the vector store everytime at startup b/c we can't save it to the huggingface repo
    chroma = D4EmailChromaDb(csv_file_in=d4_emails_file,
                             csv_file_out=d4_emails_responses_file)
    chroma.init_vector_store(embedding_function, persist_directory)

    # Initialize the email responder and run the gradio app
    responder = EmailResponder(embedding_function, persist_directory)
    email_app = create_gradio_app(responder)
    email_app.launch()
