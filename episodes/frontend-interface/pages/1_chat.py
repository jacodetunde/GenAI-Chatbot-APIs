import os
import re
import json
import time
import streamlit as st
import extra_streamlit_components as stx
from dotenv import load_dotenv
import httpx
import jwt
import logging
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

client_secret = os.getenv("CLIENT_SECRET", "my_client_secret")
bot_uri = "http://127.0.0.1:8000"
chat_uri = bot_uri + "/chat_process"
feedback_uri = "http://127.0.0.1:8000/save_feedback"

st.session_state.update(st.session_state)

user_cookie_name = "chat_user"
auth_cookie_name = "streamlit_session"

timeout = httpx.Timeout(60.0)
chunk_render_time = 0.01
buffer_padding_template = re.compile(r"(<<<) *(>>>)|(<<<) *| *(>>>)")

def hide_controls():
    """Hide Streamlit default menu and footer."""
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def get_new_feedback_key(message_index):
    """Generate a unique feedback key using the message index and UUID."""
    return f"feedback_{message_index}_{uuid.uuid4().hex[:8]}"

def submit_feedback(bot_message_text, headers, message_index, feedback_key):
    user_feedback = st.session_state.get(feedback_key, "")

    if user_feedback:
        feedback_data = {
            "bot_message": bot_message_text,
            "user_feedback": user_feedback,
        }

        try:
            response = httpx.post(feedback_uri, headers=headers, json=feedback_data, timeout=30.0)
            if response.status_code == 200:
                st.toast("Feedback submitted successfully!", icon="✅")
            else:
                st.error(f"Failed to submit feedback: {response.status_code}")
        except httpx.RequestError as e:
            st.error(f"An error occurred: {e}")


def main():
    logger.info("Application started.")
    st.set_page_config(
        initial_sidebar_state="expanded",
        layout="wide",
        page_title="GenAI Chat Interface",
        page_icon=":books:",
    )

    cookie_manager = stx.CookieManager()
    hide_controls()

    login_placeholder = st.empty()
    login_placeholder.markdown("**Please log in to continue:**")
    
    # Initialize session state
    if "feedback_store" not in st.session_state:
        st.session_state["feedback_store"] = {}
     # Initialize session state
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []  # Initialize only if not already set
        logger.info("Session state 'chat_messages' initialized as an empty list.")


    auth_cookie = cookie_manager.get(auth_cookie_name)
    user_cookie = cookie_manager.get(user_cookie_name)

    if (user_cookie and auth_cookie) or st.session_state.get("authentication_status", False):
        logger.info("User authenticated successfully.")
        login_placeholder.empty()

        st.write("# Chat Messenger")
        st.divider()

        if auth_cookie:
            st.session_state["email"] = cookie_manager.get(user_cookie_name)

        user_email = st.session_state.get("email")
        if not user_email:
            logger.error("User email is missing. Unable to proceed.")
            st.error("User email is missing. Please log in again.")
            return

        auth_payload = {"person_id": user_email}
        auth_token = jwt.encode(auth_payload, client_secret, algorithm="HS256")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_token}",
        }
        logger.info("JWT token generated for user.")

        # Display existing messages
        for index, message in enumerate(st.session_state.chat_messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    # Generate a unique feedback key
                    feedback_key = f"feedback_{index}"
                    if feedback_key not in st.session_state:
                        st.session_state[feedback_key] = ""  # Initialize feedback state
                    with st.expander("🗣️ Provide Feedback", expanded=True):
                        st.text_area(
                            "Your feedback here...",
                            key=feedback_key,
                            on_change=submit_feedback,
                            args=(message["content"], headers, index, feedback_key),
                        )

        # Input for user message
        prompt = st.chat_input("What changes would you like to make to the first draft?")
        if prompt:
            # Append user message
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Simulate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    try:
                        data = {"UserInput": prompt}
                        raw_response = ""
                        with httpx.stream(
                            "POST",
                            chat_uri,
                            headers=headers,
                            json=data,
                            timeout=timeout,
                        ) as response:
                            message_placeholder = st.empty()
                            for chunk in response.iter_bytes():
                                if chunk:
                                    decoded_data = chunk.decode("utf-8")
                                    parsed_data = re.sub(buffer_padding_template, "", decoded_data)
                                    raw_response += parsed_data
                                    message_placeholder.markdown(raw_response)
                                    time.sleep(chunk_render_time)
                                    
                        assistant_response = raw_response.strip()

                        # Append assistant response to session state
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": assistant_response}
                        )

                        # Ensure state is saved before rerun
                        st.rerun()
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        logger.error(f"Error during response generation: {e}")

    else:
        logger.warning("Authentication failed. User cookies or session state invalid.")
        st.error("Authentication failed. Please log in again.")
if __name__ == "__main__":
    main()
