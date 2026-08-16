"""
scripts/seed_db.py - populates the fleet-ops tables from the hand-authored, deterministic
rows in src/db/seed_data.py. Safe to re-run: clears the 4 tables (children before parents,
respecting foreign keys) and reinserts from scratch every time, so the DB always matches
whatever's currently in seed_data.py.

Run from the project root (after `python -m scripts.setup_db`):
    python -m scripts.seed_db
"""

from src.db.database import SessionLocal
from src.db.models import MaintenanceEvent, Mission, Suit, Technician
from src.db.seed_data import MAINTENANCE_EVENTS, MISSIONS, SUITS, TECHNICIANS


def main():
    session = SessionLocal()
    try:
        # Children first - both have FKs into suits/technicians.
        session.query(MaintenanceEvent).delete()
        session.query(Mission).delete()
        session.query(Suit).delete()
        session.query(Technician).delete()
        session.flush()

        suits_by_name = {}
        for row in SUITS:
            suit = Suit(**row)
            session.add(suit)
            suits_by_name[row["mark_name"]] = suit

        technicians_by_name = {}
        for row in TECHNICIANS:
            tech = Technician(**row)
            session.add(tech)
            technicians_by_name[row["name"]] = tech

        session.flush()  # assigns .id to every suit/technician before we reference them

        for row in MAINTENANCE_EVENTS:
            session.add(MaintenanceEvent(
                suit_id=suits_by_name[row["suit"]].id,
                technician_id=technicians_by_name[row["technician"]].id,
                event_date=row["event_date"],
                component=row["component"],
                issue=row["issue"],
                resolution=row["resolution"],
                resolution_hours=row["resolution_hours"],
                cost_usd=row["cost_usd"],
            ))

        for row in MISSIONS:
            session.add(Mission(
                suit_id=suits_by_name[row["suit"]].id,
                mission_date=row["mission_date"],
                location=row["location"],
                threat_level=row["threat_level"],
                duration_min=row["duration_min"],
                outcome=row["outcome"],
            ))

        session.commit()
        print(f"Seeded {len(SUITS)} suits, {len(TECHNICIANS)} technicians, "
              f"{len(MAINTENANCE_EVENTS)} maintenance events, {len(MISSIONS)} missions.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
