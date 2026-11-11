"""LAN client detection for the local-registration gate."""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def is_lan_client(ip: str, cidrs: Sequence[str]) -> bool:
    """Return True when ``ip`` falls inside one of the ``cidrs`` networks.

    Fails open for registration: an unparseable client IP or a CIDR entry
    that cannot be parsed yields False (allow the request) and is logged,
    never raised. IPv4 and IPv6 are both supported; an address only ever
    matches a network of the same version.
    """
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        logger.warning("LAN gate: cannot parse client IP %r; treating it as external", ip)
        return False
    for cidr in cidrs:
        try:
            network = ipaddress.ip_network(cidr, strict=False)
        except ValueError:
            logger.warning("LAN gate: cannot parse CIDR %r; ignoring it", cidr)
            continue
        if address in network:
            return True
    return False
