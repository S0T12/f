"""
User Service
============
User management and authentication service.
"""

from typing import Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.models import User
from app.core.security import hash_password, verify_password


class UserService:
    """User management service."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        result = await self.db.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()
    
    async def create(
        self,
        email: str,
        username: str,
        password: str,
        full_name: Optional[str] = None,
    ) -> User:
        """Create new user."""
        user = User(
            email=email,
            username=username,
            hashed_password=hash_password(password),
            full_name=full_name,
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user
    
    async def authenticate(
        self,
        email: str,
        password: str,
    ) -> Optional[User]:
        """Authenticate user with email and password."""
        user = await self.get_by_email(email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        await self.db.commit()
        
        return user
    
    async def update_password(
        self,
        user: User,
        new_password: str,
    ) -> None:
        """Update user password."""
        user.hashed_password = hash_password(new_password)
        await self.db.commit()
    
    async def deactivate(self, user: User) -> None:
        """Deactivate user account."""
        user.is_active = False
        await self.db.commit()
