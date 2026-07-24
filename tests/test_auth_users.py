"""Unit tests for SQLite user auth and session tokens."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestAuthUsers(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "users.db"
        self.env = {
            "APP_USERS_DB": str(self.db_path),
            "APP_SEED_USER": "chira",
            "APP_SEED_PASSWORD": "test-seed-pass",
            "APP_AUTH_SECRET": "unit-test-secret",
        }
        self._patcher = patch.dict(os.environ, self.env, clear=False)
        self._patcher.start()
        # Fresh module state against the temp DB.
        import backend.auth_users as auth

        self.auth = auth
        auth.init_user_db()

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_seed_user_can_authenticate(self) -> None:
        user = self.auth.authenticate_user("chira", "test-seed-pass")
        self.assertIsNotNone(user)
        assert user is not None
        self.assertEqual(user.username.lower(), "chira")
        self.assertTrue(self.auth.gradio_auth_credentials("chira", "test-seed-pass"))

    def test_register_and_login_token(self) -> None:
        created = self.auth.register_user("alice", "secret12")
        self.assertEqual(created.username, "alice")
        user = self.auth.authenticate_user("alice", "secret12")
        self.assertIsNotNone(user)
        assert user is not None
        token = self.auth.issue_session_token(user)
        verified = self.auth.verify_session_token(token)
        self.assertIsNotNone(verified)
        assert verified is not None
        self.assertEqual(verified.id, user.id)

    def test_reject_duplicate_username(self) -> None:
        self.auth.register_user("bob", "secret12")
        with self.assertRaises(ValueError):
            self.auth.register_user("Bob", "otherpass")

    def test_bad_password_rejected(self) -> None:
        self.assertIsNone(self.auth.authenticate_user("chira", "wrong"))


if __name__ == "__main__":
    unittest.main()
