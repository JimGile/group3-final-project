
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from typing import List
import pandas as pd
import os


class D4EmailChromaDb:
    def __init__(self, csv_file_in: str, csv_file_out: str):
        self.csv_file_in = csv_file_in
        self.csv_file_out = csv_file_out

    def init_vector_store(
            self,
            embeddings,
            persist_directory,
            chunk_size=5000,
            chunk_overlap=100) -> None:

        if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            # Split the emails into document chunks
            docs, uuids = self._preprocess_and_split_emails(
                chunk_size, chunk_overlap)

            self._load_vectorstore_docs(docs, uuids, embeddings, persist_directory)

    # Preprocess the emails csv file and split them into document chunks
    def _preprocess_and_split_emails(self, chunk_size, chunk_overlap) -> tuple[List[Document], list[str]]:

        # Remove unwanted columns before loading into vectorstore
        columns_to_drop = ['name', 'email_address',
                           'd4_staff_member', 'constituent_email_2', 'd4_response_2']
        d4_emails_df = pd.read_csv(self.csv_file_in)
        d4_emails_df = d4_emails_df.drop(columns=columns_to_drop)
        d4_emails_df.to_csv(self.csv_file_out, index=False)

        # Create a document loader for D4 Emails
        loader = CSVLoader(self.csv_file_out, encoding='utf-8')

        # Load the document
        data = loader.load()

        # Create an instance of the splitter class with the given chunk size and overlap
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Split the emails into document chunks and create uuids
        docs = splitter.split_documents(data)
        uuids = [
            self._get_csv_id(docs[i]) for i in range(len(docs))
        ]

        return docs, uuids

    def _get_csv_id(self, doc: Document) -> str:
        return f"{str(doc.metadata['source']).split('/')[-1].replace('.csv', '')}_{doc.metadata['row']}"

    def _load_vectorstore_docs(
            self,
            docs: List[Document],
            uuids: list[str],
            embeddings,
            persist_directory: str) -> None:

        # Create the vector_store with the documents and save it to disk
        try:
            Chroma.from_documents(
                docs,
                embeddings,
                persist_directory=persist_directory,
                ids=uuids
            )
        except Exception as e:
            print(
                f"Error creating/updating vector store: {str(e)}", flush=True)
