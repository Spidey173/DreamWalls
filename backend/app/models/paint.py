from sqlalchemy import Column, Integer, String, Float
from app.core.database import Base

class Paint(Base):
    __tablename__ = "paints"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    brand = Column(String(100), nullable=False, default="Asian Paints")
    name = Column(String(150), nullable=False, index=True)
    shade_code = Column(String(50), nullable=False, unique=True, index=True)
    hex = Column(String(10), nullable=False)
    finish = Column(String(50), nullable=False, default="Matte")
    color_family = Column(String(100), nullable=False, index=True)
    price = Column(Float, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "brand": self.brand,
            "name": self.name,
            "shade_code": self.shade_code,
            "hex": self.hex,
            "finish": self.finish,
            "color_family": self.color_family,
            "price": self.price,
        }
