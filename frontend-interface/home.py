import os
import re
import io
import json
import time
import datetime
import streamlit as st
import extra_streamlit_components as stx
from dotenv import load_dotenv
import httpx
import jwt
import logging
import tiktoken

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"

load_dotenv()
access_password = os.getenv("ACCESS_PASSWORD", "password")
cookie_expiry_days = 100

gpt_token_limit = 100000
token_encoding = tiktoken.encoding_for_model("gpt-4")

client_secret = os.getenv("CLIENT_SECRET", "my_client_secret")

st.session_state.update(st.session_state)

user_cookie_name = "chat_user"
auth_cookie_name = "streamlit_session"

timeout = httpx.Timeout(60.0)
chunk_render_time = 0.01
buffer_padding_template = re.compile(r"(<<<) *(>>>)|(<<<) *| *(>>>)")


def hide_controls():
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def _get_session_id():
    from streamlit.runtime import get_instance
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    runtime = get_instance()
    session_id = get_script_run_ctx().session_id
    return session_id


def reset_session_state():
    # Only reset feedback-related session state, not chat messages
    keys_to_reset = [key for key in st.session_state.keys() if "_feedback_" in key]
    for key in keys_to_reset:
        st.session_state.pop(key, None)


def check_email_password():
    def email_entered():
        if re.fullmatch(email_regex, st.session_state["email"]) and "@gmail.com" in st.session_state["email"]:
            st.session_state["valid_email"] = {
                "email": st.session_state["email"].lower().strip(),
                "session_id": _get_session_id(),
            }
            logger.info(f"Valid email entered: {st.session_state['email']}")
        else:
            st.session_state["valid_email"] = False
            logger.warning("Invalid email entered.")

    def password_entered():
        if "password" in st.session_state and st.session_state["password"] == access_password:
            st.session_state["password_correct"] = True
            logger.info("Correct password entered.")
        else:
            st.session_state["password_correct"] = False
            logger.warning("Incorrect password entered.")

    if (
        "valid_email" not in st.session_state
        or "password_correct" not in st.session_state
        or not st.session_state["valid_email"]
        or not st.session_state["password_correct"]
    ):

        st.text_input("Email", type="default", on_change=email_entered, key="email")
        st.text_input("Password", type="password", on_change=password_entered, key="password")

        submit = st.button("Login")

        if "valid_email" in st.session_state and not st.session_state["valid_email"]:
            st.error("Invalid Email")
        elif "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("Invalid Password")
        return False
    else:
        # Email Valid.
        return True


def main():
    logger.info("Application started.")
    st.set_page_config(
        initial_sidebar_state="expanded",
        layout="wide",
        page_title="GenAI Chat Interface",
        page_icon=":books:",
    )

    hide_controls()

    # Cookie manager
    cookie_manager = stx.CookieManager()

    # Session state initialization
    if "shared" not in st.session_state:
        st.session_state["shared"] = True

    login_placeholder = st.empty()
    login_placeholder.markdown("**Please log in to continue:**")

    auth_cookie = cookie_manager.get(cookie=auth_cookie_name)
    user_cookie = cookie_manager.get(cookie=user_cookie_name)

    if (user_cookie and auth_cookie) or check_email_password():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write("# GenAI Chat Interface")
        with col2:
            log_out = st.button("Log out")
        # Set user email
        user_email = user_cookie or st.session_state["email"].lower().strip()
        st.session_state["email"] = user_email  # Ensure email is stored in session state

        logger.info(f"User logged in: {user_email}")

        if "authentication_status" in st.session_state and log_out:
            # Clear all session state except chat_messages
            keys_to_clear = [key for key in st.session_state.keys() if key != "chat_messages"]
            for key in keys_to_clear:
                del st.session_state[key]
            expires_at = datetime.datetime.now() + datetime.timedelta(days=cookie_expiry_days)
            cookie_manager.set(user_cookie_name, "", key=user_cookie_name, expires_at=expires_at)
            st.rerun()

        else:
            if "authentication_status" not in st.session_state:
                st.session_state["authentication_status"] = True
                st.session_state.sidebar_state = "expanded"
                st.session_state.layout = "wide"
                st.rerun()

            if auth_cookie and user_cookie:
                st.session_state["valid_email"] = {
                    "email": cookie_manager.get(user_cookie_name),
                    "session_id": cookie_manager.get(auth_cookie_name),
                }
            elif auth_cookie and not user_cookie:
                expires_at = datetime.datetime.now() + datetime.timedelta(days=cookie_expiry_days)
                cookie_manager.set(
                    user_cookie_name,
                    st.session_state["email"].lower().strip(),
                    key="set_user",
                    expires_at=expires_at,
                )
            else:
                expires_at = datetime.datetime.now() + datetime.timedelta(days=cookie_expiry_days)
                cookie_manager.set(
                    auth_cookie_name,
                    _get_session_id(),
                    key="set_session",
                    expires_at=expires_at,
                )
                cookie_manager.set(
                    user_cookie_name,
                    st.session_state["email"].lower().strip(),
                    key="set_user",
                    expires_at=expires_at,
                )

        login_placeholder.markdown("")
        st.switch_page("pages/1_chat.py")

if __name__ == "__main__":
    main()





