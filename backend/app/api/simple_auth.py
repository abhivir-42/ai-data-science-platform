"""
Simple authentication API endpoints for prototype.
Handles user registration, login, and logout.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Response, Request, Cookie, Header
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from app.services.simple_user_service import SimpleUserService
from app.services.simple_session_manager import SimpleSessionManager
from app.core.database import database_manager

# Create router
router = APIRouter()

# Initialize services
user_service = SimpleUserService()

# Import shared session manager to ensure consistency with auth middleware
from app.core.auth_middleware import _shared_session_manager
session_manager = _shared_session_manager

# Pydantic models for request/response
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: str = None

class LoginRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    success: bool
    message: str
    user_id: str = None
    session_id: str = None

# Dependency to get database session
async def get_db():
    async with database_manager.async_session_maker() as session:
        yield session

@router.post("/register", response_model=AuthResponse)
async def register_user(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user"""
    try:
        user = await user_service.create_user(
            db=db,
            username=request.username,
            email=request.email,
            password=request.password,
            full_name=request.full_name
        )

        return AuthResponse(
            success=True,
            message="User registered successfully",
            user_id=user.user_id
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.post("/login", response_model=AuthResponse)
async def login_user(request: LoginRequest, response: Response, db: AsyncSession = Depends(get_db)):
    """Authenticate user and create session"""
    try:
        user = await user_service.authenticate_user(
            db=db,
            username=request.username,
            password=request.password
        )

        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        # Create session
        session_id = session_manager.create_session(user.user_id)

        # Set session cookie (optional, can also return session_id)
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=86400,  # 24 hours
            samesite="lax"
        )

        return AuthResponse(
            success=True,
            message="Login successful",
            user_id=user.user_id,
            session_id=session_id
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@router.post("/logout", response_model=dict)
async def logout_user(session_id: str = None, response: Response = None):
    """Logout user and destroy session"""
    try:
        if session_id:
            session_manager.destroy_session(session_id)

        # Clear session cookie
        if response:
            response.delete_cookie("session_id")

        return {"success": True, "message": "Logged out successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logout failed: {str(e)}")

@router.get("/me")
async def get_current_user(
    request: Request, 
    session_id: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None)
):
    """Get current user info - supports multiple authentication methods"""
    from app.core.auth_middleware import auth_middleware
    
    # Try to get user_id from various methods
    user_id = auth_middleware.get_user_from_request(request, session_id, authorization)
    
    if not user_id:
        raise HTTPException(
            status_code=401, 
            detail="Authentication required. Please login first."
        )

    return {
        "success": True,
        "user_id": user_id,
        "session_valid": True,
        "authentication_method": "session_verified"
    }
