"""
Simple authentication service for prototype.
Uses SHA256 hashing for passwords - easily replaceable with bcrypt later.
"""
import hashlib
import os

class SimpleAuthService:
    """Simple auth service for prototype - easily replaceable with bcrypt"""

    def __init__(self):
        # Simple salt for prototype (change this in production)
        self.salt = os.getenv("PASSWORD_SALT", "simple-salt-change-in-production")

    def hash_password(self, password: str) -> str:
        """Simple SHA256 hash for prototype"""
        salted_password = f"{self.salt}{password}"
        return hashlib.sha256(salted_password.encode()).hexdigest()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        salted_password = f"{self.salt}{plain_password}"
        return hashlib.sha256(salted_password.encode()).hexdigest() == hashed_password
