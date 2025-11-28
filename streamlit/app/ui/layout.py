# app/ui/layout.py
import streamlit as st
from app.api.client import APIClient
from app.ui.widgets import input_textbox, submit_button
from app.ui.display import show_answer
from app.config.settings import Settings

class StreamlitRAGApp:
    def __init__(self, client: APIClient, settings: Settings):
        self.client = client
        self.settings = settings

    def run(self):
        st.title("MyRAG - Streamlit UI")
        st.sidebar.header("RAG Settings")

        project_id = st.sidebar.number_input(
            "Project ID", value=self.settings.DEFAULT_PROJECT_ID, min_value=1
        )
        limit = st.sidebar.number_input(
            "Max Results", value=self.settings.DEFAULT_LIMIT, min_value=1
        )

        user_query = input_textbox("Ask a question:", "Type your question here...", key="query")
        if submit_button("Get Answer"):
            with st.spinner("Fetching answer..."):
                response = self.client.answer_rag(project_id=project_id, text=user_query, limit=limit)
                show_answer(response)
