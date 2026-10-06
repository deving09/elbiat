from functools import wraps
from fastapi import HTTPException, status, Depends
from .models import User, UserRole
from .routes.users import get_current_user  # adjust import to match your actual location


def require_role(*allowed_roles: UserRole):
    """Dependency that checks if user has one of the allowed roles."""
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role.value}' not authorized"
            )
        return current_user
    return role_checker


# Convenience dependencies
require_admin = require_role(UserRole.ADMIN)
require_enterprise = require_role(UserRole.ADMIN, UserRole.ENTERPRISE)
require_user = require_role(UserRole.ADMIN, UserRole.ENTERPRISE, UserRole.USER)


# Helper functions
def is_admin(user: User) -> bool:
    return user.role == UserRole.ADMIN


def is_enterprise_or_above(user: User) -> bool:
    return user.role in (UserRole.ADMIN, UserRole.ENTERPRISE)