from __future__ import annotations

import unittest
from urllib.parse import parse_qs, urlparse

from kraken_manager.infrastructure.auth.gitlab import GitLabIdentity, GitLabOidcClient, GitLabUnavailable
from kraken_server.composition import HybridSessionResolver


class GitLabOidcTests(unittest.TestCase):
    def test_authorization_code_request_uses_pkce_state_nonce_and_oidc_scopes(self) -> None:
        client = GitLabOidcClient("https://gitlab.internal")
        client._configuration = {
            "issuer": "https://gitlab.internal",
            "authorization_endpoint": "https://gitlab.internal/oauth/authorize",
        }
        request = client.authorization_request(
            client_id="kraken",
            redirect_uri="http://127.0.0.1:48123/callback",
            state="state-value",
            nonce="nonce-value",
        )
        query = parse_qs(urlparse(request.authorization_url).query)
        self.assertEqual(["code"], query["response_type"])
        self.assertEqual(["S256"], query["code_challenge_method"])
        self.assertEqual(["openid profile email"], query["scope"])
        self.assertEqual(["state-value"], query["state"])
        self.assertEqual(["nonce-value"], query["nonce"])
        self.assertGreaterEqual(len(request.code_verifier), 43)

    def test_cached_federated_session_keeps_reads_during_gitlab_outage(self) -> None:
        class Accounts:
            def resolve_session(self, token):
                return None

        class Identities:
            def __init__(self):
                self.values = {}

            def save(self, principal):
                self.values[principal.id] = principal

            def get(self, principal_id):
                return self.values.get(principal_id)

        class Oidc:
            unavailable = False

            def userinfo(self, token):
                if self.unavailable:
                    raise GitLabUnavailable("offline")
                return GitLabIdentity("https://gitlab.internal", "42", "User", "u@example.test")

        class Cache:
            def __init__(self):
                self.values = {}

            def save(self, token, principal_id):
                self.values[token] = (principal_id, "gitlab")

            def resolve(self, token):
                return self.values.get(token)

            def revoke(self, token):
                self.values.pop(token, None)

        identities = Identities()
        oidc = Oidc()
        resolver = HybridSessionResolver(Accounts(), identities, oidc, Cache())
        online = resolver("token")
        self.assertIsNotNone(online)
        oidc.unavailable = True
        cached = resolver("token")
        self.assertEqual(online.principal_id, cached.principal_id)
        self.assertFalse(resolver.verify_live(cached))


if __name__ == "__main__":
    unittest.main()
