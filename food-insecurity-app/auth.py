import streamlit as st
import json
import os

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def login_form():
    users = load_users()

    with st.form("login"):
        st.write("### Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if username in users and users[username]["password"] == password:
                st.success(f"Welcome back, {username}!")
                st.session_state.logged_in = True
                st.session_state.username = username
            else:
                st.error("Invalid username or password")
