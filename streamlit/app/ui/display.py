# app/ui/display.py
import streamlit as st
from ..models.GenerateAnswerRespone import GenerateAnswerRespone

def show_answer(response: GenerateAnswerRespone):
    if response.signal.lower() != "rag_answer_success":
        st.error("Failed to get answer from RAG.")
        return
    
    st.markdown("### Answer")
    st.info(response.answer)

    if response.full_prompt:
        with st.expander("Full Prompt"):
            st.code(response.full_prompt)

    if response.chat_history:
        with st.expander("Chat History"):
            for msg in response.chat_history:
                role = msg.role.capitalize()
                content = msg.content
                st.write(f"**{role}:** {content}")