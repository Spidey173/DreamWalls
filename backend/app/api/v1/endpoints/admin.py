import io
import csv
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.core.database import get_db
from app.core.security import verify_admin_password
from app.models.paint import Paint
from app.schemas.paint import PaintRead, PaintCreate, PaintUpdate

router = APIRouter()

class AdminVerifyRequest(BaseModel):
    password: str

@router.post("/verify")
def verify_password(payload: AdminVerifyRequest):
    from app.core.config import settings
    if payload.password.strip() == settings.ADMIN_PASSWORD:
        return {"valid": True, "token": settings.ADMIN_PASSWORD}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password.")

@router.get("/paints", response_model=List[PaintRead], dependencies=[Depends(verify_admin_password)])
def list_paints(
    search: Optional[str] = Query(None, description="Search by name or shade_code"),
    color_family: Optional[str] = Query(None, description="Filter by color family"),
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    query = db.query(Paint)
    if search:
        search_fmt = f"%{search.strip()}%"
        query = query.filter((Paint.name.ilike(search_fmt)) | (Paint.shade_code.ilike(search_fmt)))
    if color_family:
        query = query.filter(Paint.color_family == color_family)
    return query.offset(skip).limit(limit).all()

@router.post("/paints", response_model=PaintRead, dependencies=[Depends(verify_admin_password)])
def create_paint(payload: PaintCreate, db: Session = Depends(get_db)):
    existing = db.query(Paint).filter(Paint.shade_code == payload.shade_code).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Paint with shade code '{payload.shade_code}' already exists."
        )
    paint = Paint(**payload.model_dump())
    db.add(paint)
    db.commit()
    db.refresh(paint)
    return paint

@router.put("/paints/{paint_id}", response_model=PaintRead, dependencies=[Depends(verify_admin_password)])
def update_paint(paint_id: int, payload: PaintUpdate, db: Session = Depends(get_db)):
    paint = db.query(Paint).filter(Paint.id == paint_id).first()
    if not paint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paint not found.")
    
    update_data = payload.model_dump(exclude_unset=True)
    for field, val in update_data.items():
        setattr(paint, field, val)

    db.commit()
    db.refresh(paint)
    return paint

@router.delete("/paints/{paint_id}", dependencies=[Depends(verify_admin_password)])
def delete_paint(paint_id: int, db: Session = Depends(get_db)):
    paint = db.query(Paint).filter(Paint.id == paint_id).first()
    if not paint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paint not found.")
    db.delete(paint)
    db.commit()
    return {"message": f"Paint ID {paint_id} deleted successfully."}

@router.post("/paints/import-csv", dependencies=[Depends(verify_admin_password)])
async def import_csv_catalog(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Must be a CSV file.")
    
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    
    headers = None
    imported_count = 0
    updated_count = 0

    for row in reader:
        if not row or row[0].strip().startswith("#"):
            continue
        if headers is None:
            headers = [h.strip().lower() for h in row]
            continue

        row_dict = dict(zip(headers, [item.strip() for item in row]))
        shade_code = row_dict.get("shade_code")
        if not shade_code:
            continue

        price_val = None
        if row_dict.get("price"):
            try:
                price_val = float(row_dict["price"])
            except ValueError:
                price_val = None

        existing = db.query(Paint).filter(Paint.shade_code == shade_code).first()
        if existing:
            existing.brand = row_dict.get("brand", "Asian Paints")
            existing.name = row_dict.get("name", existing.name)
            existing.hex = row_dict.get("hex", existing.hex)
            existing.finish = row_dict.get("finish", existing.finish)
            existing.color_family = row_dict.get("color_family", existing.color_family)
            existing.price = price_val if price_val is not None else existing.price
            updated_count += 1
        else:
            new_paint = Paint(
                brand=row_dict.get("brand", "Asian Paints"),
                name=row_dict.get("name", "Unknown"),
                shade_code=shade_code,
                hex=row_dict.get("hex", "#FFFFFF"),
                finish=row_dict.get("finish", "Matte"),
                color_family=row_dict.get("color_family", "Neutral"),
                price=price_val
            )
            db.add(new_paint)
            imported_count += 1

    db.commit()
    total_paints = db.query(Paint).count()
    return {
        "message": f"Import complete: {imported_count} added, {updated_count} updated.",
        "added": imported_count,
        "updated": updated_count,
        "total_catalog": total_paints
    }
