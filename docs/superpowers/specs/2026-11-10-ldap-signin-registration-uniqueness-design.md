# Direct LDAP (Active Directory) sign-in, explicit local registration, global login uniqueness, profile groundwork

Date: 2026-11-10
Status: draft — pending owner approval
Supersedes: `2026-11-09-gitlab-sso-registration-uniqueness-design.md` (GitLab SSO
approach, dropped on the company IT administrator's recommendation — direct LDAP
is simpler for this architecture)
Scope: `kraken_server` (auth routes, config, composition), `kraken_manager`
(identity domain, LDAP client, account/identity stores), `kraken_hub` (Qt login
flow), `kraken_admin` (CLI), `migrations/`, `tests/`, `config/templates/`

## Intent and why

The company's identity lives in Active Directory: employees have domain logins.
The company IT administrator recommends authenticating Kraken **directly against
AD over LDAP** instead of routing through GitLab. The owner's requirements are
unchanged:

- an employee signs in with domain credentials in the existing login dialog;
  the account appears automatically (no admin creates it);
- local (self-registered) accounts exist **only for users outside the local
  network**;
- one login belongs to one person, across all identity sources;
- user profiles (photo, status) are a **future** feature — groundwork (storage +
  sync + API) now, UI later.

Consequence that shapes everything: Kraken verifies domain credentials itself,
server-side, at login time only, via an LDAP bind to AD. No browser flow, no
OAuth, no GitLab involvement in authentication. The existing GitLab provider is
kept untouched as a dormant fallback path.

## Owner decisions (conversations of 2026-11-09…2026-11-10)

1. **Direct LDAP/AD authentication** per the company IT administrator
   (supersedes the earlier GitLab SSO decision).
2. Domain sign-in uses the **existing username/password dialog** — no new
   authentication UI for internal users.
3. Local account creation becomes **explicit**: no implicit auto-registration on
   a failed login, plus an IP gate that blocks registration from the LAN.
4. Logins are unique globally, across local accounts and domain (LDAP) users.
5. Profile: groundwork only — schema, AD sync, API. No profile UI in this change.

## Goals

1. **Domain sign-in.** In the existing login dialog, a username/password that
   matches no local account is verified against AD (service-account search +
   user bind). On first success an account + principal (`provider=ldap`) is
   JIT-provisioned. Subsequent logins verify against AD the same way. Sessions
   are ordinary opaque server sessions; nothing on an authenticated request
   touches AD.
2. **Explicit local registration.** Failed login no longer auto-creates an
   account. A dedicated "Создать аккаунт" mode in the login UI calls the
   registration endpoint and saves the resulting account.
3. **LAN registration gate.** `POST /api/v1/auth/accounts` returns 403 when the
   client IP falls inside configured LAN CIDR ranges; outside clients are
   unaffected. Login is never blocked by the gate.
4. **Global login uniqueness.** A login (case-insensitive) exists at most once
   across all providers: local registration is rejected when the login is taken
   by any principal; when a domain user's login collides with a local account
   that already *holds the login claim* (its principal was provisioned), that
   local account is disabled (corporate identity wins) with an audit record.
   The detailed claim rules live in §4 — in particular a local account whose
   user never signed in keeps working after the domain user claims the login
   (no lockout in either direction).
5. **Profile groundwork.** `principals` stores `login`, `avatar_url`,
   `status_text` (+ timestamp/source marker). AD attributes
   (`displayName`, `mail`, `thumbnailPhoto` when present) are synced at each
   successful domain login. Minimal API: read profile in existing
   session/principals payloads, `PATCH` own profile (status text, avatar URL).

## Non-goals

- No browser/OAuth/PKCE flow (dropped with the GitLab SSO approach).
- No Kerberos/SPNEGO (password-based LDAP bind only), no SAML.
- No AD verification on authenticated requests (login time only) — sessions are
  local opaque tokens.
- No offline fallback for domain logins when AD is unavailable.
- No profile UI: no profile page, no image upload (avatar is a URL), no
  presence/online status, no directory page.
- No changes to the existing GitLab provider (kept dormant; removal is a
  separate decision).
- No changes to offline desktop-profile accounts (`accounts.sqlite3`).
- No `X-Forwarded-For`/trusted-proxy client-IP mode in v1: the server is
  assumed to accept direct connections (stated assumption; behind a reverse
  proxy this is a small follow-up).

## Constraints

- Clean-architecture layering: the LDAP transport lives in
  `kraken_manager/infrastructure/auth/ldap.py`, mirroring `gitlab.py` (typed
  exceptions, own timeout, injectable transport for tests).
- **New dependency: `ldap3`** (pure Python, no native build; already declared,
  unused, in `plugins/neuralimage`'s `web` extra). Python has no stdlib LDAP
  client; `ldap3` is the de-facto standard. Added to the server profile extra.
- Closed-network deployments: the **server** must reach a domain controller
  (see Operational prerequisites); the client side needs nothing new because
  there is no browser flow.
- The desktop already requires HTTPS for network server connections.
- Alembic migrations are hand-written; the current tree has a **single head**
  (`20261029_0011`). The new revision is a plain single-parent child of it.
- Existing test conventions: `unittest` for pure units, pytest + real
  PostgreSQL + FastAPI `TestClient` for server integration (shared `database`
  fixture), `pytest.importorskip` for optional extras.

## Approaches considered

### Authentication method

- **A. Direct LDAP bind to AD (chosen — company IT administrator's
  recommendation).** Server verifies username/password at login; JIT account
  creation; zero client-side authentication changes; no per-request dependency
  on any IdP; no OAuth app to maintain. Trade-off accepted: no free avatar
  (AD `thumbnailPhoto` is usually absent) — the profile groundwork stores the
  field anyway and a future upload UI fills it.
- **B. GitLab SSO (previous design, dropped).** Required a GitLab OAuth app, a
  browser flow in the desktop, and a live token check on every resolved
  session (GitLab down → mutations 503). More moving parts; the IT
  administrator prefers direct LDAP.
- **C. Kerberos/SPNEGO.** "Click and you're in", but Windows-only, KDC/SPNEGO
  complexity, and a separate desktop flow. Rejected.

### Where domain verification runs

Server-side in the login path — the only sensible placement: the password
already travels to the server for local accounts, and a client-side AD check
would push AD connectivity and credentials handling onto every workstation.

### Service account vs direct user-DN bind

- **Service account (chosen, v1):** the server binds with a read-only AD
  service account, searches the user by `sAMAccountName` (works in any OU),
  reads `displayName`/`mail`/`thumbnailPhoto`, then re-binds as that user with
  the submitted password to verify it. One code path; profile attributes come
  along for free.
- Direct bind by DN template (rejected for v1): assumes a fixed OU layout
  (`user_dn_template`), breaks when users live in different OUs, and profile
  attributes can't be read before the authenticated bind. Remains a possible
  later config addition if the IT administrator objects to a service account.

### Local registration policy (unchanged from previous draft)

- **A. Leave implicit auto-registration as is.** Rejected: the footgun remains —
  an internal user with a mistyped password silently creates a "twin" local
  account instead of going through the domain.
- **B. IP gate only.** Insufficient alone: the implicit auto-registration UX
  stays.
- **C. IP gate + explicit "create account" mode in the desktop (chosen — owner
  pick).** Removes the footgun and implements the original requirement
  literally.

### Login uniqueness (unchanged in substance; "GitLab" → "LDAP")

- Local accounts are **already** unique case-insensitively within their
  provider (`server_accounts.username_key` UNIQUE in Postgres; `username UNIQUE
  NOCASE` in SQLite). The gap is cross-provider: domain users have no stored
  login today, so `local:ivan` and domain user `ivan` can coexist.
- **Chosen:** persist `login`/`login_key` on `principals` for both providers,
  unique index on `login_key`; application-level pre-checks with the collision
  rules in Goal 4; DB index as the safety net.
- Rejected: block domain sign-in on collision (locks out a domain user — an
  anti-goal); rename on collision (surprising data mutation).

### Profile scope (unchanged)

- **Chosen:** groundwork only (columns + sync + small API). The hard part of the
  future feature (schema + auto photo source) is done; UI is a separate change.
- Rejected: nothing at all (owner asked for groundwork); full profile now (that
  is the future feature itself — presence, uploads, directory UI — not
  groundwork).

## Design

### 1. Domain (LDAP/AD) sign-in

**Domain** — `PrincipalProvider` gains `LDAP = "ldap"`; a `Principal.ldap(...)`
factory; `external_key = "ldap:<sAMAccountName casefolded>"`. `Principal` gains
optional fields `login`, `avatar_url`, `status_text` (profile groundwork, §5).

**LdapClient** (`kraken_manager/infrastructure/auth/ldap.py`)

- Constructor: `url` (`ldap://` or `ldaps://`), `base_dn`, `user_filter`
  (default `(&(objectClass=user)(objectCategory=person)(sAMAccountName={username}))`
  — matches the company's existing GLPI LDAP catalog filter plus the login
  attribute), `bind_dn`, `bind_password`,
  `start_tls: bool`, `ca_file: str | None`, `timeout: float = 5.0`,
  `transport` (injectable seam for tests — the client wraps an `ldap3`
  `Connection` behind this seam, mirroring how `GitLabOidcClient` is tested).
- `verify(username, password) -> LdapIdentity | None`:
  1. Bind with the service account (failure → `LdapAuthenticationError` — a
     misconfiguration, surfaced as 503, never as 401).
  2. Search under `base_dn` with `user_filter` and the escaped username:
     zero entries → `None`; exactly one → DN + attributes; more than one →
     `LdapUnavailable` (misconfiguration, 503).
  3. Unbind; re-bind as the found DN with the submitted password. Success →
     `LdapIdentity(sAMAccountName, displayName, mail, thumbnailPhoto | None)`;
     failure → `None`.
- Exceptions: `LdapUnavailable` (network, timeout, TLS handshake, ambiguous
  data), `LdapAuthenticationError` (service-account bind rejected).
- **TLS policy:** `ldaps://` → TLS on connect (typically 636); `ldap://` +
  `start_tls=true` → STARTTLS (typically 389) — the default for our AD
  (`ldap://DC.ad.inme:389`). Plain `ldap://` without STARTTLS is refused unless
  `ldap_allow_insecure = true` is set explicitly (a startup warning is logged)
  — an escape hatch for a closed segment where the DC's certificate cannot be
  trusted. `ca_file` supplies an internal CA (or the DC's self-signed
  certificate) when the server's trust store doesn't include it (same pattern
  as the existing `gitlab_ca_file`).

**Login path** (`POST /api/v1/auth/sessions` + composition)

- New composition-level component (designer's naming: `DomainAuthBroker`):
  `authenticate(username, password) -> LocalSession | None`.
  1. Look up the account by username (new read-only `find_account` on the
     account store).
  2. **Not found, LDAP configured** → `LdapClient.verify`; on success: create
     the account row (`auth_provider='ldap'`, `password_hash` = hash of 32
     random bytes — structurally unusable), create the session through the
     existing session machinery, provision the principal
     (`provider=ldap`, `login=sAMAccountName`, `display_name=displayName`,
     `email=mail`, `avatar_url=thumbnailPhoto`) plus its linked performer —
     generalizing the existing local-account provisioning path.
  3. **Found, `auth_provider='local'`** → existing hash path first. A
     account's credentials are never verified against AD as the local path. If local authentication fails and LDAP is configured, the same attempt is verified against AD (a domain user may hold the same login) and the §4 claim rules then decide what happens to the row — this is the mechanism that keeps the domain user from ever being locked out.
  4. **Found, `auth_provider='ldap'`** → `LdapClient.verify`; on success
     create a session and refresh profile attributes (§4). AD is consulted on
     every domain login, never on authenticated requests.
  5. Verification `None` → 401 "Invalid credentials" (indistinguishable from a
     local failure — no username enumeration). `LdapUnavailable` → 503
     "Доменная авторизация временно недоступна". `LdapAuthenticationError` →
     503 (misconfiguration).
- **Brute force:** the existing per-account `failed_attempts`/`blocked_until`
  lockout applies to LDAP accounts (the row exists after the first successful
  login). Before the first success, AD's own account-lockout policy applies.
  No extra mechanism.
- **AD down:** existing sessions keep working (tokens are local); new logins
  get 503. Stated limitation, no offline fallback (non-goal).

**kraken-admin** — account management lives in the shared
`kraken_admin/application/accounts.py`, used by **both** the CLI and the Qt
administration window, so these changes apply to both surfaces:
`reset-password` refuses LDAP accounts ("пароль управляется доменом");
`account-list` / the Qt account table show the provider; `disable/enable`,
`revoke-sessions`, `grant/revoke-admin` work unchanged (the `enabled` flag
gates domain logins too). `account-create` stays local-only and gains the §4
login pre-check.

**Configuration** (`[identity]`, env `KRAKEN_LDAP_*`, via
`ServerConfig.apply_to_environment`):

- `ldap_url` — e.g. `ldap://dc01.corp.local:389` or `ldaps://dc01.corp.local:636`
- `ldap_base_dn` — e.g. `DC=corp,DC=local`
- `ldap_user_filter` — default
  `(&(objectClass=user)(objectCategory=person)(sAMAccountName={username}))`
  (the full AD filter, matching the company's GLPI catalog — a bare
  `sAMAccountName` filter could match non-user objects)
- `ldap_bind_dn` — service account DN
- `ldap_bind_password` — **never in TOML**: DPAPI-protected secret file (the
  existing `protect_secret` pattern, e.g. `ldap-bind.secret`)
- `ldap_start_tls` — bool (true for `ldap://` URLs, see TLS policy)
- `ldap_allow_insecure` — bool, default false (the only way to allow plain
  `ldap://` without STARTTLS; startup warning logged)
- `ldap_ca_file` — optional internal CA / DC certificate
- `ldap_timeout` — float, default 5.0

LDAP is **enabled only when** `ldap_url` + `ldap_bind_dn` + the bind secret are
all present; otherwise the login path behaves exactly as today (closed-network
servers without AD are unaffected). Templates
(`config/templates/server.production.template.toml`, `server.local.template.toml`)
updated with commented examples.

### 2. Explicit local account creation

- `configure_remote_local` (`kraken_hub/composition/catalog.py`): **remove** the
  401 → `register_local` fallback; a failed login raises a typed error the UI
  can render ("учётная запись не найдена или неверный пароль" — for domain
  users this is the expected message until they realize their account is
  created on first *successful* domain sign-in; see UX note below).
- `RemoteServerProjectService.register_local` (and the server endpoint) are kept
  and called only from the new explicit "Создать аккаунт" mode in the login UI
  (username, display name, password → success → save local program account →
  connect).
- The connect-page hint text (`_server_project_hint`, which currently promises
  "если такой записи ещё нет на сервере, она будет создана") is updated to
  match the explicit flow.
- UX note (designer's call): the first-launch dialog for a fresh internal user
  is simply "enter your domain login and password" — the account is created on
  success. No separate "domain" mode is needed in the UI; the server decides
  local-vs-domain.

### 3. LAN registration gate

- `[identity]` gains `lan_cidrs` (list of CIDR strings, e.g.
  `["10.0.0.0/8", "192.168.0.0/16"]`) → env `KRAKEN_LAN_CIDRS` (JSON list).
  Empty/absent list = gate off (backward compatible).
- Server: client IP via `request.client.host` (v1, no X-Forwarded-For). The
  decision lives in a small pure function `is_lan_client(ip, cidrs)` (stdlib
  `ipaddress`) that is unit-testable; the HTTP context exposes the IP provider
  as an injectable seam so `TestClient`-based tests can override it.
- `register_local_account` (`routes/auth.py`): when the gate is on and the
  client is in-LAN → 403 with a clear RU message ("Создание локальных
  аккаунтов из внутренней сети запрещено. Используйте доменную учётную
  запись."). The gate applies to registration only — login endpoints are
  untouched.

### 4. Global login uniqueness

**Schema (one migration, §6)** — `principals` gains:
`login` (text, null), `login_key` (text, null, **unique index**), `avatar_url`
(text, null), `avatar_source` (text, null — `'ldap'` or `'manual'`),
`status_text` (text, null), `profile_updated_at` (timestamptz, null).
`server_accounts` gains `auth_provider` (text, not null, default `'local'`).

**Domain** — `Principal` gains optional fields `login`, `avatar_url`,
`status_text` (all `None`-defaulted, backward compatible). Local principals:
`login = account username`. LDAP principals: `login = sAMAccountName`.

**Sync** — on every successful domain login the resolver's upsert refreshes the
principal's `login`, `login_key`, `display_name`, `email`, and `avatar_url` —
but `avatar_url` is only overwritten when `avatar_source != 'manual'`. (AD
`thumbnailPhoto` is usually absent; the field is simply left NULL in that case.)

**Login-claim semantics.** `login_key` is a *claim*: whoever holds it owns that
login. A principal that loses a claim keeps its identity and keeps working —
only the login belongs to the other party. (Designer's call resolving the
reverse-direction collision: an admin-CLI-created local account provisions its
principal lazily at first login, so a domain user could legitimately claim the
login in between; the rules below make every interleaving safe.)

**Enforcement**

- Local registration, HTTP path: pre-check the login against the identity
  catalog. Taken by an LDAP principal → 409-style error "логин уже занят".
  Taken by a local account → existing `username_key` behavior (registration
  with the same password still acts as sign-in).
- Local account creation, admin-CLI path (`kraken-admin account-create`):
  gains an identity-catalog dependency for the same pre-check; login taken by
  an LDAP principal → the command fails with a clear error (no half-created
  account). The CLI does **not** start provisioning principals at creation
  time — first-login provisioning stays as is.
- LDAP JIT provisioning: if `login_key` is already held by a **local**
  principal → disable that local account (`enabled=false`), revoke its
  sessions, write an `administration_audit` record
  (`login_conflict_local_disabled` with both principal ids); the account row is
      then converted for the domain user (`auth_provider='ldap'`, structurally
      unusable hash, enabled again — the local identity is what stops
      working, and the domain user's session is a fresh one); then provision the
  LDAP principal normally. The domain user is never locked out; the local
  account that squatted the corporate login is the one that stops working, and
  the audit row records what happened (admin can restore/rename via
  `kraken-admin`).
- Local principal provisioning at first login (existing accounts whose
  principal did not claim the login at creation): if the login is already
  claimed by an LDAP principal, create/keep the local principal with
  `login_key = NULL` (it keeps working — local sign-in goes through
  `server_accounts`, not `principals`) and write an audit row
  (`login_claim_conflict_local_skipped`). No lockout, no IntegrityError.
- Two LDAP principals with the same login cannot occur — `sAMAccountName` is
  unique within the domain; if data ever shows otherwise, fail provisioning
  with a clear error (never silent rename).
- The unique index on `login_key` is the DB safety net; all IntegrityError
  paths are handled explicitly (no silent swallows).

**Backfill (in the migration)** — existing **local** principals get
`login`/`login_key` from `server_accounts` (join via `external_key =
'local:<username>'`). The LDAP provider is new, so no LDAP-held keys exist yet;
the "skip if claimed" check is kept as a safety net with an audit row
(`login_claim_conflict_backfill`). Pre-existing local twins of future domain
logins are resolved lazily by the claim rules above, at the relevant user's
next login. No data is dropped or renamed.

### 5. Profile API (groundwork)

- Session introspection (`GET /api/v1/session`) and principals list items gain
  `login`, `avatar_url`, `status_text`.
- `PATCH /api/v1/principals/me/profile` — body `{ "status_text"?: str,
  "avatar_url"?: str }`, self-only (403 for anyone else). Setting
  `avatar_url` manually marks `avatar_source = 'manual'`; AD sync then never
  overwrites it. Length limits: status text ≤ 255 chars, avatar URL ≤
  2048 chars (validated, no fabricated data).
- No image upload endpoint in v1 (avatars are URLs — for LDAP users the field
  is usually empty until the future upload UI).

### 6. Migration

One new Alembic revision (parent: `20261029_0011`, the current single head):

1. `ALTER TABLE principals` — add the six columns from §4.
2. `ALTER TABLE server_accounts` — add `auth_provider` (text, not null,
   default `'local'`).
3. Create the unique index on `principals.login_key` (partial, `WHERE
   login_key IS NOT NULL`).
4. Backfill local principals' `login`/`login_key` (rule above).
5. Downgrade: drop index + columns (no backfill reversal needed).

The SQLAlchemy `MetaData` copies used by stores/tests
(`postgres/account_store.py`, `postgres/identity_store.py`) are extended to
match. The desktop offline SQLite mirror (`accounts.sqlite3`,
`identity.sqlite3`) is **not** changed — profiles and domain auth are
server-side concepts.

### 7. Configuration summary

```toml
[identity]
gitlab_issuer = ""                    # existing, untouched (dormant fallback)
gitlab_ca_file = ""                   # existing, untouched
ldap_url = "ldap://DC.ad.inme:389"                  # NEW (our values)
ldap_base_dn = "DC=ad,DC=inme"                      # NEW (our values)
ldap_user_filter = "(&(objectClass=user)(objectCategory=person)(sAMAccountName={username}))"  # NEW (default shown)
ldap_bind_dn = "CN=kraken,CN=Users,DC=ad,DC=inme"   # NEW (dedicated account, or glpi_tech — IT admin's call)
# ldap_bind_password: DPAPI secret file (ldap-bind.secret), never in TOML     # NEW
ldap_start_tls = true                             # NEW
ldap_allow_insecure = false                       # NEW (escape hatch, closed segment)
ldap_ca_file = ""                                 # NEW (optional: internal CA / DC cert)
ldap_timeout = 5.0                                # NEW
lan_cidrs = ["10.0.0.0/8", "192.168.0.0/16"]      # NEW (empty = gate off; adjust to the real office subnets)
```

All mapped to `KRAKEN_*` env vars in `ServerConfig.apply_to_environment` and
consumed in `kraken_server/composition.py`.

### 8. Operational prerequisites — checklist for the IT administrator

The owner is not the systems administrator; this section is written to be
forwarded to the company IT admin as-is.

**Resolved** (from the GLPI "LDAP-каталоги" settings, 2026-11-10):

- Domain `ad.inme`, DC `DC.ad.inme:389`; base DN `DC=ad,DC=inme`.
- Login attribute `sAMAccountName`; user filter
  `(&(objectClass=user)(objectCategory=person)(sAMAccountName={username}))`.
- The Kraken server sits in the **same network segment** as the DC → no
  firewall rule needed for port 389.

**Remaining for the IT admin:**

1. **Bind account:** a read-only AD account for Kraken — recommended: a
   dedicated `kraken` account (independent lifecycle/rotation from GLPI's
   `glpi_tech`); reusing the existing `glpi_tech` is acceptable (one-line
   config difference). Needs only read access to user objects under
   `DC=ad,DC=inme`. Hand over the DN + password (goes into the DPAPI secret
   file, never into the TOML).
2. **TLS:** confirm whether the DC offers STARTTLS on 389 and which
   certificate it presents (internal CA or AD self-signed). If the Kraken
   server cannot trust that certificate: either hand over the certificate
   (→ `ldap_ca_file`) or explicitly approve `ldap_allow_insecure = true` for
   the closed segment.
3. **Office subnets:** provide the LAN CIDR ranges for the registration gate
   (`lan_cidrs`) — e.g. the office subnets where employee workstations sit.

## Anti-goals (failure if any appears)

- Kerberos/SPNEGO or a browser/OAuth flow sneaking back into the design.
- A domain user locked out or renamed because of a login collision.
- Any remaining path that silently auto-creates a local account.
- An LDAP-backed account falling back to a local password — or a local account
  verified against AD. Providers are strictly separate.
- AD consulted on authenticated requests (login time only).
- The LDAP bind password in a TOML file, environment dump, log, or DB (DPAPI
  secret file only).
- A half-finished profile UI (presence indicators, uploads) masquerading as the
  future feature.

## Testing

- **Unit (LdapClient, fake transport):** successful verify (service bind →
  search → user bind); wrong password → `None`; missing user → `None`;
  ambiguous search → `LdapUnavailable`; service bind rejected →
  `LdapAuthenticationError`; timeout → `LdapUnavailable`; TLS policy selection
  (`ldaps://` vs STARTTLS vs plain refused); attribute extraction including
  absent `thumbnailPhoto`; username escaping in the filter.
- **Unit (login routing, broker matrix):** unknown user + LDAP configured →
  JIT (account row `auth_provider='ldap'` with structurally unusable hash,
  principal `provider=ldap`, session created); unknown user + LDAP unconfigured
  → plain 401 (today's behavior); local account → hash path only (LDAP client
  never invoked); LDAP account → LDAP path only (hash never consulted);
  `LdapUnavailable` → 503; `LdapAuthenticationError` → 503.
- **Unit (gate):** `is_lan_client` matrix — inside/outside/boundary CIDR,
  empty list, IPv4 (and IPv6-mapped) cases.
- **Integration (PG + TestClient, existing `database` fixture):**
  - registration from in-LAN IP (injected IP provider) → 403; from outside →
    201; gate off (empty `lan_cidrs`) → 201 everywhere;
  - local registration while an LDAP principal holds the login → rejected;
    `kraken-admin account-create` with the same login → command fails, no
    account created;
  - LDAP JIT while a local account holds the login → local disabled +
    sessions revoked + audit row + LDAP principal provisioned;
  - reverse interleave: local account created (principal not yet provisioned) →
    domain user claims the login → local user's first login still succeeds,
    its principal gets `login_key = NULL` + `login_claim_conflict_local_skipped`
    audit row (no IntegrityError, no lockout);
  - `kraken-admin reset-password` on an LDAP account → refused; `disable` on an
    LDAP account → subsequent domain logins rejected;
  - profile PATCH: self ok, foreign 403; `avatar_source=manual` stops AD sync
    overwrite.
- **Hub:** flow tests for the explicit "create account" mode and the removed
  auto-registration (stubbed server endpoints); no other hub auth changes.

## Effort estimate

- LDAP client + login broker + config + admin-CLI adjustments: ~2–3 days
- Explicit registration + LAN gate: ~1–2 days
- Global login uniqueness (+ migration/backfill): ~1–2 days
- Profile groundwork (columns, sync, API): ~1–2 days

Total: roughly 1.5–2 weeks of focused work.

## Left to the builder

- Exact Qt widgets/placement for the "Создать аккаунт" mode (follow existing
  dialog conventions and RU tone).
- LdapClient transport-seam details (how the `ldap3` `Connection` is wrapped
  for fake-transport tests).
- Test seam for the client-IP provider.
- Broker wiring in `composition.py` and the `--development` mode path.
- Exact RU error-message wording (consistent with existing strings).
