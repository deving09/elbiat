from app.db import SessionLocal  # adjust import
from app import models           # adjust import

def main():
    db = SessionLocal()
    try:
        rows = db.query(models.Models).all()
        #rows = db.query(models.Evals).all()
        #rows = db.query(models.Task).all()

        for i, row in enumerate(rows):
            print("=" * 50)
            for column in row.__table__.columns:
                print(f"{column.name}: {getattr(row, column.name)}\n")

            if i > 15:
                break
    finally:
        db.close()

if __name__ == "__main__":
    main()
