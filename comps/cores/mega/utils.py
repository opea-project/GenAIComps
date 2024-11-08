# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ipaddress
import multiprocessing
import os
import random
from socket import AF_INET, SOCK_STREAM, socket
from typing import List, Optional, Union

import jwt
import requests
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .keycloak import Keycloak
from .logger import CustomLogger


def is_port_free(host: str, port: int) -> bool:
    """Check if a given port on a host is free.

    :param host: The host to check.
    :param port: The port to check.
    :return: True if the port is free, False otherwise.
    """
    with socket(AF_INET, SOCK_STREAM) as session:
        return session.connect_ex((host, port)) != 0


def check_ports_availability(host: Union[str, List[str]], port: Union[int, List[int]]) -> bool:
    """Check if one or more ports on one or more hosts are free.

    :param host: The host(s) to check.
    :param port: The port(s) to check.
    :return: True if all ports on all hosts are free, False otherwise.
    """
    hosts = [host] if isinstance(host, str) else host
    ports = [port] if isinstance(port, int) else port

    return all(is_port_free(h, p) for h in hosts for p in ports)


def get_internal_ip():
    """Return the private IP address of the gateway in the same network.

    :return: Private IP address.
    """
    import socket

    ip = "127.0.0.1"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
    except Exception:
        pass
    return ip


def get_public_ip(timeout: float = 0.3):
    """Return the public IP address of the gateway in the public network."""
    import urllib.request

    def _get_public_ip(url):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as fp:
                _ip = fp.read().decode().strip()
                return _ip

        except:
            pass

    ip_lookup_services = [
        "https://api.ipify.org",
        "https://ident.me",
        "https://checkip.amazonaws.com/",
    ]

    for _, url in enumerate(ip_lookup_services):
        ip = _get_public_ip(url)
        if ip:
            return ip


def typename(obj):
    """Get the typename of object."""
    if not isinstance(obj, type):
        obj = obj.__class__
    try:
        return f"{obj.__module__}.{obj.__name__}"
    except AttributeError:
        return str(obj)


def get_event(obj) -> multiprocessing.Event:
    if isinstance(obj, multiprocessing.Process) or isinstance(obj, multiprocessing.context.ForkProcess):
        return multiprocessing.Event()
    elif isinstance(obj, multiprocessing.context.SpawnProcess):
        return multiprocessing.get_context("spawn").Event()
    else:
        raise TypeError(f'{obj} is not an instance of "multiprocessing.Process"')


def in_docker():
    """Checks if the current process is running inside Docker."""
    path = "/proc/self/cgroup"
    if os.path.exists("/.dockerenv"):
        return True
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as file:
            return any("docker" in line for line in file)
    return False


def host_is_local(hostname):
    """Check if hostname is point to localhost."""
    import socket

    fqn = socket.getfqdn(hostname)
    if fqn in ("localhost", "0.0.0.0") or hostname == "0.0.0.0":
        return True

    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


assigned_ports = set()
unassigned_ports = []
DEFAULT_MIN_PORT = 49153
MAX_PORT = 65535


def reset_ports():
    def _get_unassigned_ports():
        # if we are running out of ports, lower default minimum port
        if MAX_PORT - DEFAULT_MIN_PORT - len(assigned_ports) < 100:
            min_port = int(os.environ.get("JINA_RANDOM_PORT_MIN", "16384"))
        else:
            min_port = int(os.environ.get("JINA_RANDOM_PORT_MIN", str(DEFAULT_MIN_PORT)))
        max_port = int(os.environ.get("JINA_RANDOM_PORT_MAX", str(MAX_PORT)))
        return set(range(min_port, max_port + 1)) - set(assigned_ports)

    unassigned_ports.clear()
    assigned_ports.clear()
    unassigned_ports.extend(_get_unassigned_ports())
    random.shuffle(unassigned_ports)


def random_port() -> Optional[int]:
    """Get a random available port number.

    :return: A random port.
    """

    def _random_port():
        import socket

        def _check_bind(port):
            with socket.socket() as s:
                try:
                    s.bind(("", port))
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    return port
                except OSError:
                    return None

        _port = None
        if len(unassigned_ports) == 0:
            reset_ports()
        for idx, _port in enumerate(unassigned_ports):
            if _check_bind(_port) is not None:
                break
        else:
            raise OSError(
                f"can not find an available port in {len(unassigned_ports)} unassigned ports, assigned already {len(assigned_ports)} ports"
            )
        int_port = int(_port)
        unassigned_ports.pop(idx)
        assigned_ports.add(int_port)
        return int_port

    try:
        return _random_port()
    except OSError:
        assigned_ports.clear()
        unassigned_ports.clear()
        return _random_port()


def get_access_token(token_url: str, client_id: str, client_secret: str) -> str:
    """Get access token using OAuth client credentials flow."""
    logger = CustomLogger("tgi_or_tei_service_auth")
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(token_url, data=data, headers=headers)
    if response.status_code == 200:
        token_info = response.json()
        return token_info.get("access_token", "")
    else:
        logger.error(f"Failed to retrieve access token: {response.status_code}, {response.text}")
        return ""


bearer_scheme = HTTPBearer(auto_error=False)


def token_validator(allowed_roles: Optional[List[str]] = None):
    async def validate_token(
        request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
    ):
        """Validates the token, checks for allowed roles, and sets user details in request.state.user if valid.

        Raises HTTPException with appropriate status code and message if validation fails.
        """
        # If token is not provided, skip validation
        JWT_AUTH = os.getenv("JWT_AUTH", False)
        if not JWT_AUTH:
            request.state.user = None
            return
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme or Missing Token",
            )
        if credentials.scheme != "Bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme",
            )
        try:
            token = credentials.credentials
            identity_provider = Keycloak()
            decoded_token = identity_provider.decode_token(token)

            if not decoded_token:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token.")

            # Extract roles from the token
            user_roles = decoded_token.get("realm_access", {}).get("roles", [])

            # Check if user has any of the allowed roles
            if allowed_roles and not any(role in user_roles for role in allowed_roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail="User does not have required permissions."
                )

            # Set user details in request.state.user
            request.state.user = {
                "username": decoded_token.get("preferred_username"),
                "email": decoded_token.get("email"),
                "roles": user_roles,
            }

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired.")

        except jwt.InvalidTokenError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signature.")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Token validation error: {str(e)}"
            )

    return validate_token


class SafeContextManager:
    """This context manager ensures that the `__exit__` method of the
    sub context is called, even when there is an Exception in the
    `__init__` method."""

    def __init__(self, context_to_manage):
        self.context_to_manage = context_to_manage

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.context_to_manage.__exit__(exc_type, exc_val, exc_tb)
