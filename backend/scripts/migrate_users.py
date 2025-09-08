#!/usr/bin/env python3
"""
Database migration script for simple user authentication.
Adds user table and user_id columns to existing tables.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from app.core.database import init_database

async def migrate_database():
    """Run database migrations for user authentication"""
    print("🚀 Starting database migration for user authentication...")

    # Initialize database connection
    await init_database()

    # Import required modules after database is initialized
    from sqlalchemy import text
    from app.core.database import database_manager

    async with database_manager.async_session_maker() as db_session:
        try:
            print("📋 Step 1: Creating users table...")

            # Create users table
            await db_session.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR(36) PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    full_name VARCHAR(100),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login DATETIME,
                    is_active BOOLEAN DEFAULT TRUE
                );
            """))

            print("📋 Step 2: Adding user_id column to agent_sessions...")
            # Add user_id column to agent_sessions (if it doesn't exist)
            try:
                await db_session.execute(text("""
                    ALTER TABLE agent_sessions ADD COLUMN user_id VARCHAR(36);
                """))
                print("✅ Added user_id column to agent_sessions")
            except Exception as e:
                if "duplicate column name" in str(e).lower():
                    print("ℹ️  user_id column already exists in agent_sessions")
                else:
                    print(f"⚠️  Error adding user_id to agent_sessions: {e}")

            print("📋 Step 3: Adding user_id column to workflow_executions...")
            # Add user_id column to workflow_executions (if it doesn't exist)
            try:
                await db_session.execute(text("""
                    ALTER TABLE workflow_executions ADD COLUMN user_id VARCHAR(36);
                """))
                print("✅ Added user_id column to workflow_executions")
            except Exception as e:
                if "duplicate column name" in str(e).lower():
                    print("ℹ️  user_id column already exists in workflow_executions")
                else:
                    print(f"⚠️  Error adding user_id to workflow_executions: {e}")

            print("📋 Step 4: Creating default anonymous user...")
            # Create a default anonymous user for existing sessions
            try:
                await db_session.execute(text("""
                    INSERT OR IGNORE INTO users
                    (user_id, username, email, password, full_name, is_active)
                    VALUES
                    ('anonymous-user', 'anonymous', 'anonymous@example.com', 'anonymous', 'Anonymous User', TRUE);
                """))
                print("✅ Created default anonymous user")
            except Exception as e:
                print(f"⚠️  Error creating anonymous user: {e}")

            print("📋 Step 5: Migrating existing sessions to anonymous user...")
            # Update existing sessions to use anonymous user
            try:
                await db_session.execute(text("""
                    UPDATE agent_sessions
                    SET user_id = 'anonymous-user'
                    WHERE user_id IS NULL;
                """))
                print("✅ Migrated agent_sessions to anonymous user")

                await db_session.execute(text("""
                    UPDATE workflow_executions
                    SET user_id = 'anonymous-user'
                    WHERE user_id IS NULL;
                """))
                print("✅ Migrated workflow_executions to anonymous user")
            except Exception as e:
                print(f"⚠️  Error migrating sessions: {e}")

            # Commit all changes
            await db_session.commit()
            print("✅ All database migrations completed successfully!")

        except Exception as e:
            print(f"❌ Migration failed: {e}")
            await db_session.rollback()
            raise

if __name__ == "__main__":
    asyncio.run(migrate_database())
