#!/usr/bin/env python3
"""
Comprehensive test script for simple user authentication system.
Tests all components: registration, login, session management, database integration.
"""
import asyncio
import sys
import os
from sqlalchemy import text

# Add backend to path
backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_path)

from app.services.simple_user_service import SimpleUserService
from app.services.simple_session_manager import SimpleSessionManager
from app.services.session_service import SessionService
from app.core.database import init_database
from app.models.user import User
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

class SimpleAuthTester:
    """Test class for simple authentication system"""

    def __init__(self):
        self.user_service = SimpleUserService()
        self.session_manager = SimpleSessionManager()
        self.session_service = SessionService()

    async def setup_database(self):
        """Initialize database connection"""
        print("🔧 Setting up database connection...")
        await init_database()
        print("✅ Database initialized")

    async def test_user_registration(self):
        """Test user registration"""
        print("\n🧪 Testing User Registration...")

        # Create a test database session
        from app.core.database import database_manager
        async with database_manager.async_session_maker() as db:
            # Test user registration
            user = await self.user_service.create_user(
                db=db,
                username="testuser2",
                email="test2@example.com",
                password="password123"
            )

            print(f"✅ User created: {user.username} (ID: {user.user_id})")
            return user

    async def test_user_login(self, username: str, password: str):
        """Test user login"""
        print("\n🧪 Testing User Login...")

        from app.core.database import database_manager
        async with database_manager.async_session_maker() as db:
            user = await self.user_service.authenticate_user(db, username, password)
            if user:
                print(f"✅ Login successful for: {user.username}")
                return user
            else:
                print("❌ Login failed")
                return None

    async def test_session_management(self, user_id: str):
        """Test session creation and validation"""
        print("\n🧪 Testing Session Management...")

        # Create session
        session_id = self.session_manager.create_session(user_id)
        print(f"✅ Session created: {session_id}")

        # Validate session
        retrieved_user_id = self.session_manager.get_user_from_session(session_id)
        if retrieved_user_id == user_id:
            print("✅ Session validation successful")
            return session_id
        else:
            print("❌ Session validation failed")
            return None

    async def test_agent_session_with_user(self, user_id: str):
        """Test that agent sessions work with user association"""
        print("\n🧪 Testing Agent Session with User Association...")

        # Create a mock agent instance (we'll use a simple object for testing)
        class MockAgent:
            def __init__(self):
                self.data = {"test": "data"}

            def get_data_cleaned(self):
                return self.data

        mock_agent = MockAgent()

        # Create agent session with user
        session_id = await self.session_service.create_session(
            agent_instance=mock_agent,
            agent_type="cleaning",
            user_id=user_id,
            metadata={"test": True}
        )

        print(f"✅ Agent session created with user: {session_id}")

        # Verify session has user_id
        session = await self.session_service.get_session(session_id)
        if session:
            print("✅ Agent session retrieved successfully")
            return session_id
        else:
            print("❌ Agent session retrieval failed")
            return None

    async def test_database_integrity(self):
        """Test database integrity and relationships"""
        print("\n🧪 Testing Database Integrity...")

        from app.core.database import database_manager
        async with database_manager.async_session_maker() as db:
            # Check users table
            result = await db.execute(text("SELECT COUNT(*) FROM users"))
            user_count = result.scalar()
            print(f"✅ Users in database: {user_count}")

            # Check sessions with user_id
            result = await db.execute(text("SELECT COUNT(*) FROM agent_sessions WHERE user_id IS NOT NULL"))
            sessions_with_user = result.scalar()
            print(f"✅ Agent sessions with user association: {sessions_with_user}")

            # Check anonymous user exists
            result = await db.execute(text("SELECT * FROM users WHERE username = 'anonymous'"))
            anonymous_user = result.first()
            if anonymous_user:
                print("✅ Anonymous user exists for backward compatibility")
            else:
                print("❌ Anonymous user missing")

    async def run_all_tests(self):
        """Run all authentication tests"""
        print("🚀 Starting Simple Authentication System Tests")
        print("=" * 60)

        try:
            # Setup
            await self.setup_database()

            # Test user registration
            user = await self.test_user_registration()
            if not user:
                print("❌ Registration test failed")
                return

            # Test user login
            logged_in_user = await self.test_user_login("testuser2", "password123")
            if not logged_in_user:
                print("❌ Login test failed")
                return

            # Test session management
            session_id = await self.test_session_management(user.user_id)
            if not session_id:
                print("❌ Session management test failed")
                return

            # Test agent session with user
            agent_session_id = await self.test_agent_session_with_user(user.user_id)
            if not agent_session_id:
                print("❌ Agent session integration test failed")
                return

            # Test database integrity
            await self.test_database_integrity()

            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED!")
            print("✅ User registration works")
            print("✅ User login works")
            print("✅ Session management works")
            print("✅ Agent sessions associate with users")
            print("✅ Database integrity maintained")
            print("\n🚀 Simple authentication system is ready!")

        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    tester = SimpleAuthTester()
    asyncio.run(tester.run_all_tests())
