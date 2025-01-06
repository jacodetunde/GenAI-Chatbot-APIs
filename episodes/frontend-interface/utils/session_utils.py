import re
from streamlit.runtime import get_instance
from streamlit.runtime.scriptrunner import get_script_run_ctx

email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"

ALLOWED_EMAILS = [
    "jacodetunde@gmail.com",
    "user1@example.com",
    "user2@example.com",
    # Add other emails here
]

def initialize_session_state(st):
    """Initialize shared session state variables."""
    if "email" not in st.session_state:
        st.session_state["email"] = ""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = None
    if "valid_email" not in st.session_state:
        st.session_state["valid_email"] = None
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = False
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

def validate_email(email):
    """Validate email using regex."""
    return re.fullmatch(email_regex, email)

def get_session_id():
    """Retrieve the current Streamlit session ID."""
    runtime = get_instance()
    session_id = get_script_run_ctx().session_id
    return session_id
