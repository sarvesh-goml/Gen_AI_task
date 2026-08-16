"""
src/db/models.py - the structured database schema: suits, technicians, maintenance
events, and missions. This is the "structured data" counterpart to data/documents/.
"""

from sqlalchemy import Column, Integer, String, Text, Numeric, Date, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Suit(Base):
    __tablename__ = "suits"

    id = Column(Integer, primary_key=True)
    mark_name = Column(String, unique=True, nullable=False)
    status = Column(String, nullable=False)  # combat_ready | needs_maintenance | in_storage | decommissioned
    power_core_pct = Column(Numeric, nullable=False)
    last_diagnostic_date = Column(Date, nullable=False)

    maintenance_events = relationship("MaintenanceEvent", back_populates="suit")
    missions = relationship("Mission", back_populates="suit")


class Technician(Base):
    __tablename__ = "technicians"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    specialty = Column(String, nullable=False)
    years_experience = Column(Integer, nullable=False)

    maintenance_events = relationship("MaintenanceEvent", back_populates="technician")


class MaintenanceEvent(Base):
    __tablename__ = "maintenance_events"

    id = Column(Integer, primary_key=True)
    suit_id = Column(Integer, ForeignKey("suits.id"), nullable=False)
    technician_id = Column(Integer, ForeignKey("technicians.id"), nullable=False)
    event_date = Column(Date, nullable=False)
    component = Column(String, nullable=False)
    issue = Column(Text, nullable=False)
    resolution = Column(Text, nullable=False)
    resolution_hours = Column(Numeric, nullable=False)
    cost_usd = Column(Numeric, nullable=False)

    suit = relationship("Suit", back_populates="maintenance_events")
    technician = relationship("Technician", back_populates="maintenance_events")


class Mission(Base):
    __tablename__ = "missions"

    id = Column(Integer, primary_key=True)
    suit_id = Column(Integer, ForeignKey("suits.id"), nullable=False)
    mission_date = Column(Date, nullable=False)
    location = Column(String, nullable=False)
    threat_level = Column(Integer, nullable=False)  # 1 (routine) - 5 (extinction-level)
    duration_min = Column(Integer, nullable=False)
    outcome = Column(String, nullable=False)  # success | partial | aborted

    suit = relationship("Suit", back_populates="missions")
