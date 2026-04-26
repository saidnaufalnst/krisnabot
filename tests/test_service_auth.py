import unittest

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.service_auth import (
    SERVICE_KEY_HEADER,
    require_admin_service_key,
    require_chat_service_key,
)


class ServiceAuthTests(unittest.TestCase):
    def override_setting(self, name: str, value) -> None:
        previous = getattr(settings, name)
        object.__setattr__(settings, name, value)
        self.addCleanup(object.__setattr__, settings, name, previous)

    @staticmethod
    def client_for_dependency(dependency) -> TestClient:
        app = FastAPI()

        @app.get("/ping", dependencies=[Depends(dependency)])
        def ping() -> dict[str, str]:
            return {"status": "ok"}

        return TestClient(app)

    def test_admin_allows_local_development_without_service_key(self) -> None:
        self.override_setting("environment", "development")
        self.override_setting("krisnabot_service_key", "")

        response = self.client_for_dependency(require_admin_service_key).get("/ping")

        self.assertEqual(response.status_code, 200)

    def test_admin_fails_closed_in_production_without_service_key(self) -> None:
        self.override_setting("environment", "production")
        self.override_setting("krisnabot_service_key", "")

        response = self.client_for_dependency(require_admin_service_key).get("/ping")

        self.assertEqual(response.status_code, 503)

    def test_admin_requires_matching_service_key_when_configured(self) -> None:
        self.override_setting("environment", "production")
        self.override_setting("krisnabot_service_key", "secret-key")
        client = self.client_for_dependency(require_admin_service_key)

        missing = client.get("/ping")
        invalid = client.get("/ping", headers={SERVICE_KEY_HEADER: "wrong"})
        valid = client.get("/ping", headers={SERVICE_KEY_HEADER: "secret-key"})

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(invalid.status_code, 401)
        self.assertEqual(valid.status_code, 200)

    def test_chat_key_is_optional_until_enabled(self) -> None:
        self.override_setting("environment", "production")
        self.override_setting("krisnabot_service_key", "secret-key")
        self.override_setting("krisnabot_require_chat_key", False)

        response = self.client_for_dependency(require_chat_service_key).get("/ping")

        self.assertEqual(response.status_code, 200)

    def test_chat_requires_service_key_when_enabled(self) -> None:
        self.override_setting("environment", "production")
        self.override_setting("krisnabot_service_key", "secret-key")
        self.override_setting("krisnabot_require_chat_key", True)
        client = self.client_for_dependency(require_chat_service_key)

        missing = client.get("/ping")
        valid = client.get("/ping", headers={SERVICE_KEY_HEADER: "secret-key"})

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(valid.status_code, 200)


if __name__ == "__main__":
    unittest.main()
