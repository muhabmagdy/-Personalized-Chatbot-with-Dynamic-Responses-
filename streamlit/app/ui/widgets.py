# app/ui/widgets.py
import streamlit as st

def input_textbox(label: str, placeholder: str = "", key: str = None) -> str:
    return st.text_area(label, placeholder, key=key)

def submit_button(label: str = "Ask") -> bool:
    return st.button(label)
