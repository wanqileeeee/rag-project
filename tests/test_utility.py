from contextlib import nullcontext
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from utility import check_password


class PasswordProtectionTests(TestCase):
    def setUp(self):
        self.fake_streamlit = SimpleNamespace(
            secrets={"APP_PASSWORD": "correct horse battery staple"},
            session_state={},
            error=Mock(),
            subheader=Mock(),
            form=Mock(return_value=nullcontext()),
            text_input=Mock(return_value=""),
            form_submit_button=Mock(return_value=False),
            rerun=Mock(),
        )
        self.streamlit_patch = patch("utility.st", self.fake_streamlit)
        self.streamlit_patch.start()
        self.addCleanup(self.streamlit_patch.stop)

    def test_requires_a_password_submission(self):
        self.assertFalse(check_password())
        self.fake_streamlit.form.assert_called_once_with("password_form")

    def test_authenticates_matching_password_and_remembers_session(self):
        self.fake_streamlit.text_input.return_value = "correct horse battery staple"
        self.fake_streamlit.form_submit_button.return_value = True

        self.assertTrue(check_password())
        self.assertTrue(self.fake_streamlit.session_state["password_authenticated"])
        self.fake_streamlit.rerun.assert_called_once_with()

    def test_rejects_incorrect_password(self):
        self.fake_streamlit.text_input.return_value = "wrong"
        self.fake_streamlit.form_submit_button.return_value = True

        self.assertFalse(check_password())
        self.fake_streamlit.error.assert_called_once_with("Incorrect password.")

    def test_fails_closed_when_password_is_not_configured(self):
        self.fake_streamlit.secrets = {}

        self.assertFalse(check_password())
        self.fake_streamlit.error.assert_called_once()
