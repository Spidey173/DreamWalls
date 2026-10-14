from fastapi import Header, HTTPException, status
from typing import Optional
from app.core.config import settings

def verify_admin_password(
    x_admin_password: Optional[str] = Header(None, alias="X-Admin-Password"),
    authorization: Optional[str] = Header(None)
):
    provided_password = None
    if x_admin_password:
        provided_password = x_admin_password.strip()
    elif authorization and authorization.lower().startswith("bearer "):
        provided_password = authorization[7:].strip()

    if not provided_password or provided_password != settings.ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin credentials."
        )
    return True
