# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Dict, Optional

import jwt
import requests
from jwt import ExpiredSignatureError, InvalidTokenError


class Keycloak:
    def __init__(self, realm_url: str = os.getenv("REALM_URL"), algorithm: str = "RS256"):
        """Initializes the Keycloak JWT Interface with the realm URL and algorithm.

        :param realm_url: Keycloak realm URL to fetch public keys for token verification.
        :param algorithm: Algorithm used for the token, usually 'RS256' for Keycloak.
        """
        self.realm_url = realm_url
        self.algorithm = algorithm
        self.public_keys = self.fetch_public_keys()

    def fetch_public_keys(self) -> Dict[str, str]:
        """Fetches and returns Keycloak public keys for token verification.

        :return: Dictionary mapping key IDs to their corresponding public keys.
        """
        try:
            response = requests.get(f"{self.realm_url}/protocol/openid-connect/certs")
            response.raise_for_status()
            certs = response.json()
            return {key["kid"]: jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key)) for key in certs["keys"]}
        except requests.RequestException as e:
            return {}

    def decode_token(self, token: str) -> Optional[Dict]:
        """Decodes a Keycloak JWT token and verifies its signature and expiration.

        :param token: JWT token as a string.
        :return: Decoded payload as a dictionary if valid, None otherwise.
        """
        try:
            unverified_header = jwt.get_unverified_header(token)
            key = self.public_keys.get(unverified_header.get("kid"))
            if not key:
                print("Invalid token header: key ID not found.")
                return None
            decoded = jwt.decode(token, key=key, algorithms=[self.algorithm])
            return decoded
        except ExpiredSignatureError:
            raise ExpiredSignatureError
        except InvalidTokenError:
            raise InvalidTokenError

    def verify_token(self, token: str) -> bool:
        """Verifies if the token is valid and not expired.

        :param token: JWT token as a string.
        :return: True if valid, False otherwise.
        """
        decoded = self.decode_token(token)
        return decoded is not None

    def get_user_info(self, token: str) -> Optional[Dict]:
        """Extracts user information from the JWT token payload.

        :param token: JWT token as a string.
        :return: Dictionary of user information if available, None otherwise.
        """
        decoded = self.decode_token(token)
        if decoded:
            return {
                "username": decoded.get("preferred_username"),
                "email": decoded.get("email"),
                "roles": decoded.get("realm_access", {}).get("roles", []),
            }
        return None
