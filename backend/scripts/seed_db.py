import csv
import sys
from pathlib import Path

# Add backend to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.database import Base, engine, SessionLocal
from app.models.paint import Paint
from app.core.config import settings

def seed_database(csv_path: Path = None):
    csv_file = csv_path or (settings.DATA_DIR / "paints.csv")
    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
        sys.exit(1)

    print(f"Initializing database tables at: {settings.DATABASE_URL}...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        count = 0
        updated = 0
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = None
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
                    existing.name = row_dict.get("name", "")
                    existing.hex = row_dict.get("hex", "#FFFFFF")
                    existing.finish = row_dict.get("finish", "Matte")
                    existing.color_family = row_dict.get("color_family", "Neutral")
                    existing.price = price_val
                    updated += 1
                else:
                    new_paint = Paint(
                        brand=row_dict.get("brand", "Asian Paints"),
                        name=row_dict.get("name", ""),
                        shade_code=shade_code,
                        hex=row_dict.get("hex", "#FFFFFF"),
                        finish=row_dict.get("finish", "Matte"),
                        color_family=row_dict.get("color_family", "Neutral"),
                        price=price_val,
                    )
                    db.add(new_paint)
                    count += 1

        db.commit()
        total = db.query(Paint).count()
        print(f"Successfully seeded database! Added {count} new shades, updated {updated}. Total paints: {total}")
    except Exception as e:
        db.rollback()
        print(f"Error while seeding paints: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
