import hmac

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def check_password() -> bool:
    try:
        expected_password = st.secrets["APP_PASSWORD"]
    except (KeyError, StreamlitSecretNotFoundError):
        st.error(
            "App password is not configured. Set APP_PASSWORD in "
            "`.streamlit/secrets.toml` or your deployment secrets."
        )
        return False

    if not isinstance(expected_password, str) or not expected_password:
        st.error("APP_PASSWORD must be a non-empty string.")
        return False

    if st.session_state.get("password_authenticated", False):
        return True

    st.subheader("Password protected")
    with st.form("password_form"):
        entered_password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")

    if not submitted:
        return False

    if hmac.compare_digest(entered_password, expected_password):
        st.session_state["password_authenticated"] = True
        st.rerun()
        return True

    st.error("Incorrect password.")
    return False
