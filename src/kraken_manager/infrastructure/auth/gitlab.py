"""GitLab Self-Managed OIDC discovery and mandatory live userinfo checks."""

from __future__ import annotations

import json
import base64
import hashlib
import secrets
import ssl
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class GitLabIdentity:
    issuer: str
    subject: str
    name: str
    email: str | None

    @property
    def stable_key(self) -> str:
        return f"{self.issuer}|{self.subject}"


class GitLabUnavailable(RuntimeError):
    pass


class GitLabAuthenticationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OidcAuthorizationRequest:
    authorization_url: str
    code_verifier: str
    state: str
    nonce: str


class GitLabOidcClient:
    """Minimal OIDC transport; OAuth browser orchestration stays in Desktop."""

    def __init__(self, issuer: str, *, ca_file: str | None = None, timeout: float = 5.0) -> None:
        self.issuer = issuer.rstrip("/") + "/"
        self.timeout = timeout
        self._ssl_context = ssl.create_default_context(cafile=ca_file)
        self._configuration: dict[str, Any] | None = None

    def _json_get(self, url: str, *, access_token: str | None = None) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        request = Request(url, headers=headers, method="GET")
        try:
            with urlopen(request, timeout=self.timeout, context=self._ssl_context) as response:
                payload = json.load(response)
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise GitLabAuthenticationError("GitLab rejected the access token") from exc
            raise GitLabUnavailable("GitLab OIDC endpoint is unavailable") from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise GitLabUnavailable("GitLab OIDC endpoint is unavailable") from exc
        if not isinstance(payload, dict):
            raise GitLabUnavailable("GitLab returned an invalid OIDC response")
        return payload

    def _json_post(self, url: str, form: dict[str, str]) -> dict[str, Any]:
        request = Request(
            url,
            data=urlencode(form).encode("ascii"),
            headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout, context=self._ssl_context) as response:
                payload = json.load(response)
        except HTTPError as exc:
            if exc.code in {400, 401, 403}:
                raise GitLabAuthenticationError("GitLab rejected the OAuth request") from exc
            raise GitLabUnavailable("GitLab OAuth endpoint is unavailable") from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise GitLabUnavailable("GitLab OAuth endpoint is unavailable") from exc
        if not isinstance(payload, dict):
            raise GitLabUnavailable("GitLab returned an invalid OAuth response")
        return payload

    def discover(self, *, refresh: bool = False) -> dict[str, Any]:
        if self._configuration is None or refresh:
            endpoint = urljoin(self.issuer, ".well-known/openid-configuration")
            configuration = self._json_get(endpoint)
            if str(configuration.get("issuer", "")).rstrip("/") != self.issuer.rstrip("/"):
                raise GitLabUnavailable("OIDC issuer mismatch")
            self._configuration = configuration
        return dict(self._configuration)

    def userinfo(self, access_token: str) -> GitLabIdentity:
        configuration = self.discover()
        endpoint = str(configuration.get("userinfo_endpoint", ""))
        if not endpoint:
            raise GitLabUnavailable("OIDC discovery has no userinfo endpoint")
        payload = self._json_get(endpoint, access_token=access_token)
        subject = str(payload.get("sub", "")).strip()
        if not subject:
            raise GitLabUnavailable("GitLab userinfo has no subject")
        name = str(payload.get("name") or payload.get("preferred_username") or subject)
        email = payload.get("email")
        return GitLabIdentity(self.issuer.rstrip("/"), subject, name, None if email is None else str(email))

    @staticmethod
    def _base64url(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    def authorization_request(
        self,
        *,
        client_id: str,
        redirect_uri: str,
        state: str | None = None,
        nonce: str | None = None,
    ) -> OidcAuthorizationRequest:
        """Build Authorization Code + PKCE parameters for a system browser."""
        configuration = self.discover()
        endpoint = str(configuration.get("authorization_endpoint", ""))
        if not endpoint:
            raise GitLabUnavailable("OIDC discovery has no authorization endpoint")
        verifier = self._base64url(secrets.token_bytes(48))
        challenge = self._base64url(hashlib.sha256(verifier.encode("ascii")).digest())
        state_value = state or secrets.token_urlsafe(32)
        nonce_value = nonce or secrets.token_urlsafe(32)
        query = urlencode(
            {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scope": "openid profile email",
                "state": state_value,
                "nonce": nonce_value,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        separator = "&" if "?" in endpoint else "?"
        return OidcAuthorizationRequest(endpoint + separator + query, verifier, state_value, nonce_value)

    def exchange_code(
        self,
        *,
        client_id: str,
        redirect_uri: str,
        code: str,
        code_verifier: str,
    ) -> dict[str, Any]:
        configuration = self.discover()
        endpoint = str(configuration.get("token_endpoint", ""))
        if not endpoint:
            raise GitLabUnavailable("OIDC discovery has no token endpoint")
        payload = self._json_post(
            endpoint,
            {
                "grant_type": "authorization_code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code": code,
                "code_verifier": code_verifier,
            },
        )
        if not str(payload.get("access_token", "")):
            raise GitLabUnavailable("GitLab token response has no access token")
        return payload

    def refresh(self, *, client_id: str, refresh_token: str) -> dict[str, Any]:
        configuration = self.discover()
        endpoint = str(configuration.get("token_endpoint", ""))
        if not endpoint:
            raise GitLabUnavailable("OIDC discovery has no token endpoint")
        payload = self._json_post(
            endpoint,
            {"grant_type": "refresh_token", "client_id": client_id, "refresh_token": refresh_token},
        )
        if not str(payload.get("access_token", "")):
            raise GitLabUnavailable("GitLab refresh response has no access token")
        return payload


__all__ = [
    "GitLabAuthenticationError",
    "GitLabIdentity",
    "GitLabOidcClient",
    "GitLabUnavailable",
    "OidcAuthorizationRequest",
]
