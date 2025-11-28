# ==========================
# streamlit/streamlit_app.py
# ==========================
import os
import streamlit as st
from app.api.client import APIClient
from app.ui.layout import StreamlitRAGApp


from app.config.settings import Settings




def main():
    # Load config
    settings = Settings()
    client = APIClient(base_url=settings.API_BASE_URL)

    # Run UI
    app = StreamlitRAGApp(client=client, settings=settings)
    app.run()



if __name__ == '__main__':
    main()