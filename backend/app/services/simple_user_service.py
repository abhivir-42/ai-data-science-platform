"""
Simple user management service for prototype.
Handles user creation, authentication, and basic user operations.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User
from app.services.simple_auth_service import SimpleAuthService
import uuid
from datetime import datetime

class SimpleUserService:
    """Simple user management for prototype"""

    def __init__(self):
        self.auth_service = SimpleAuthService()

    async def create_user(self, db: AsyncSession, username: str, email: str, password: str, full_name: str = None) -> User:
        """Create a new user"""
        # Check if user already exists
        existing_user = await self.get_user_by_username(db, username)
        if existing_user:
            raise ValueError(f"User with username '{username}' already exists")

        existing_email = await self.get_user_by_email(db, email)
        if existing_email:
            raise ValueError(f"User with email '{email}' already exists")

        # Create new user
        user = User(
            user_id=str(uuid.uuid4()),
            username=username,
            email=email,
            password=self.auth_service.hash_password(password),
            full_name=full_name
        )

        db.add(user)
        await db.commit()
        await db.refresh(user)

        return user

    async def authenticate_user(self, db: AsyncSession, username: str, password: str) -> User:
        """Authenticate user with username and password"""
        user = await self.get_user_by_username(db, username)
        if not user:
            return None

        if not self.auth_service.verify_password(password, user.password):
            return None

        # Update last login
        user.last_login = datetime.utcnow()
        await db.commit()

        return user

    async def get_user_by_username(self, db: AsyncSession, username: str) -> User:
        """Get user by username"""
        result = await db.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def get_user_by_email(self, db: AsyncSession, email: str) -> User:
        """Get user by email"""
        result = await db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def get_user_by_id(self, db: AsyncSession, user_id: str) -> User:
        """Get user by user_id"""
        result = await db.execute(
            select(User).where(User.user_id == user_id)
        )
        return result.scalar_one_or_none()
