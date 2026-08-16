"""
src/db/seed_data.py - deterministic, hand-authored seed rows for the Spider-Man database.
"""

SUITS = [
    {"mark_name": "Stark Suit", "status": "needs_maintenance", "power_core_pct": 84, "last_diagnostic_date": "2024-03-11"},
    {"mark_name": "Iron Spider", "status": "combat_ready", "power_core_pct": 97, "last_diagnostic_date": "2024-03-19"},
    {"mark_name": "Advanced Suit", "status": "combat_ready", "power_core_pct": 99, "last_diagnostic_date": "2024-03-20"},
    {"mark_name": "Stealth Suit", "status": "combat_ready", "power_core_pct": 91, "last_diagnostic_date": "2024-03-18"},
    {"mark_name": "Homemade Suit", "status": "in_storage", "power_core_pct": 88, "last_diagnostic_date": "2024-02-01"},
    {"mark_name": "Classic Suit", "status": "decommissioned", "power_core_pct": 0, "last_diagnostic_date": "2012-05-04"},
]

TECHNICIANS = [
    {"name": "Happy Hogan", "specialty": "Structural", "years_experience": 8},
    {"name": "Peter Parker", "specialty": "Web Systems", "years_experience": 20},
    {"name": "Ned Leeds", "specialty": "Avionics", "years_experience": 6},
    {"name": "Otto Octavius", "specialty": "Power Systems", "years_experience": 11},
    {"name": "Gwen Stacy", "specialty": "Structural", "years_experience": 4},
    {"name": "Spidey Automated Diagnostics", "specialty": "Software", "years_experience": 0},
]

MAINTENANCE_EVENTS = [
    {"suit": "Stark Suit", "technician": "Happy Hogan", "event_date": "2023-12-01", "component": "Left web shooter", "issue": "Intermittent fault under cold conditions", "resolution": "Replaced web nozzle coil and resealed housing", "resolution_hours": 4.5, "cost_usd": 2200},
    {"suit": "Stark Suit", "technician": "Happy Hogan", "event_date": "2024-01-14", "component": "Left web shooter", "issue": "Repeat intermittent fault after nozzle coil replacement", "resolution": "Escalated to full web shooter housing replacement", "resolution_hours": 9, "cost_usd": 6800},
    {"suit": "Stark Suit", "technician": "Happy Hogan", "event_date": "2024-03-02", "component": "Left web shooter", "issue": "Fault flagged a third time under sustained cold exposure", "resolution": "Replaced with redesigned cold-rated nozzle assembly", "resolution_hours": 6, "cost_usd": 4100},
    {"suit": "Stark Suit", "technician": "Ned Leeds", "event_date": "2024-02-10", "component": "HUD display", "issue": "Minor glare artifact in direct sunlight", "resolution": "Recalibrated display polarization filter", "resolution_hours": 1.5, "cost_usd": 300},
    {"suit": "Stark Suit", "technician": "Otto Octavius", "event_date": "2024-01-22", "component": "Power core regulator", "issue": "Output dipped 4% below nominal for under a minute", "resolution": "Replaced regulator fuse, retested to spec", "resolution_hours": 2, "cost_usd": 900},
    {"suit": "Stark Suit", "technician": "Happy Hogan", "event_date": "2024-03-11", "component": "Right web shooter", "issue": "New fault reported on right side for the first time", "resolution": "Replaced right web nozzle preemptively", "resolution_hours": 4, "cost_usd": 2100},
    {"suit": "Stark Suit", "technician": "Happy Hogan", "event_date": "2024-02-07", "component": "Left gauntlet plating", "issue": "Minor wear from training exercise", "resolution": "Buffed and resealed plating", "resolution_hours": 1, "cost_usd": 150},
    {"suit": "Stark Suit", "technician": "Peter Parker", "event_date": "2024-01-20", "component": "Web nozzle", "issue": "Output 3% below spec on diagnostic sweep", "resolution": "Recalibrated web nozzle alignment", "resolution_hours": 2, "cost_usd": 500},
    {"suit": "Iron Spider", "technician": "Peter Parker", "event_date": "2024-02-03", "component": "Chestplate servo", "issue": "Minor calibration drift after a high-G maneuver", "resolution": "Recalibrated via the diagnostic dock", "resolution_hours": 1, "cost_usd": 150},
    {"suit": "Iron Spider", "technician": "Peter Parker", "event_date": "2024-03-19", "component": "Power regulation circuit", "issue": "Output fluctuation of plus or minus 2 percent under sustained load", "resolution": "Replaced the regulation circuit board", "resolution_hours": 3, "cost_usd": 2400},
    {"suit": "Iron Spider", "technician": "Gwen Stacy", "event_date": "2023-12-15", "component": "Left gauntlet plating", "issue": "Hairline stress fracture after impact", "resolution": "Replaced plating section", "resolution_hours": 2.5, "cost_usd": 700},
    {"suit": "Iron Spider", "technician": "Happy Hogan", "event_date": "2024-03-05", "component": "Chestplate servo", "issue": "Servo grinding noise reported by pilot", "resolution": "Lubricated and retested servo assembly", "resolution_hours": 1, "cost_usd": 180},
    {"suit": "Iron Spider", "technician": "Gwen Stacy", "event_date": "2024-01-16", "component": "Chestplate integrity", "issue": "Minor scoring from debris impact", "resolution": "Buffed and resealed chestplate coating", "resolution_hours": 1.5, "cost_usd": 300},
    {"suit": "Iron Spider", "technician": "Otto Octavius", "event_date": "2024-03-15", "component": "Power core regulator", "issue": "Preventive inspection ahead of scheduled mission", "resolution": "No repair needed - logged as passed diagnostic", "resolution_hours": 1, "cost_usd": 0},
    {"suit": "Advanced Suit", "technician": "Spidey Automated Diagnostics", "event_date": "2024-02-20", "component": "Neural sensor array", "issue": "Reassembly lag of 0.4 seconds above spec", "resolution": "Applied firmware patch; lag reduced to 0.1 seconds", "resolution_hours": 0.5, "cost_usd": 0},
    {"suit": "Advanced Suit", "technician": "Otto Octavius", "event_date": "2024-01-05", "component": "Web nozzle", "issue": "Thermal throttling triggered below spec threshold", "resolution": "Replaced coolant line, retested under load", "resolution_hours": 3, "cost_usd": 1800},
    {"suit": "Advanced Suit", "technician": "Ned Leeds", "event_date": "2023-12-28", "component": "Targeting HUD", "issue": "Lock time drift of 0.1 seconds above spec", "resolution": "Recalibrated sensor array", "resolution_hours": 1, "cost_usd": 200},
    {"suit": "Advanced Suit", "technician": "Gwen Stacy", "event_date": "2024-02-14", "component": "Left web shooter", "issue": "Minor efficiency loss reported", "resolution": "Cleaned shooter intake, retested to spec", "resolution_hours": 2, "cost_usd": 400},
    {"suit": "Advanced Suit", "technician": "Peter Parker", "event_date": "2024-02-28", "component": "Neural sensor array", "issue": "Routine firmware audit", "resolution": "Updated firmware to latest validated build", "resolution_hours": 1, "cost_usd": 0},
    {"suit": "Advanced Suit", "technician": "Ned Leeds", "event_date": "2024-03-20", "component": "Power core regulator", "issue": "Routine post-mission inspection", "resolution": "No repair needed - logged as passed diagnostic", "resolution_hours": 0.5, "cost_usd": 0},
    {"suit": "Stealth Suit", "technician": "Happy Hogan", "event_date": "2024-01-30", "component": "Web wing mount", "issue": "Mount vibration exceeding tolerance during sustained use", "resolution": "Reinforced mount bracket", "resolution_hours": 5, "cost_usd": 3100},
    {"suit": "Stealth Suit", "technician": "Gwen Stacy", "event_date": "2023-11-18", "component": "Left leg servo", "issue": "Actuator response delay under heavy load", "resolution": "Replaced actuator servo", "resolution_hours": 4, "cost_usd": 2600},
    {"suit": "Stealth Suit", "technician": "Peter Parker", "event_date": "2024-02-25", "component": "Power core regulator", "issue": "Output spike during web discharge", "resolution": "Installed surge dampener", "resolution_hours": 3.5, "cost_usd": 2900},
    {"suit": "Stealth Suit", "technician": "Ned Leeds", "event_date": "2024-01-27", "component": "Comms array", "issue": "Static interference on priority channel", "resolution": "Replaced comms antenna array", "resolution_hours": 2.5, "cost_usd": 1200},
    {"suit": "Stealth Suit", "technician": "Ned Leeds", "event_date": "2023-11-30", "component": "HUD display", "issue": "Refresh rate below spec under G-load", "resolution": "Replaced HUD driver board", "resolution_hours": 2, "cost_usd": 950},
    {"suit": "Stealth Suit", "technician": "Gwen Stacy", "event_date": "2023-12-08", "component": "Web line stabilizer", "issue": "Drift during high-speed maneuvering", "resolution": "Recalibrated stabilizer gyroscope", "resolution_hours": 1.5, "cost_usd": 300},
    {"suit": "Homemade Suit", "technician": "Ned Leeds", "event_date": "2023-11-05", "component": "Web line stabilizer", "issue": "Minor drift during hover mode", "resolution": "Recalibrated stabilizer gyroscope", "resolution_hours": 1, "cost_usd": 250},
    {"suit": "Homemade Suit", "technician": "Otto Octavius", "event_date": "2024-01-08", "component": "Power core", "issue": "Routine capacity check, no fault found", "resolution": "No repair needed - logged as passed diagnostic", "resolution_hours": 0.5, "cost_usd": 0},
    {"suit": "Homemade Suit", "technician": "Otto Octavius", "event_date": "2023-12-20", "component": "Web-fluid regulation valve", "issue": "Output ceiling test", "resolution": "No repair needed - logged as passed diagnostic", "resolution_hours": 0.5, "cost_usd": 0},
    {"suit": "Classic Suit", "technician": "Peter Parker", "event_date": "2012-05-04", "component": "Full frame", "issue": "Total structural failure during the Battle of New York", "resolution": "Suit decommissioned, not repaired", "resolution_hours": 0, "cost_usd": 0},
]

MISSIONS = [
    {"suit": "Stark Suit", "mission_date": "2024-01-05", "location": "Coney Island", "threat_level": 4, "duration_min": 38, "outcome": "success"},
    {"suit": "Stark Suit", "mission_date": "2024-02-18", "location": "Times Square", "threat_level": 3, "duration_min": 22, "outcome": "success"},
    {"suit": "Stark Suit", "mission_date": "2024-03-01", "location": "Manhattan Bridge", "threat_level": 2, "duration_min": 15, "outcome": "success"},
    {"suit": "Stark Suit", "mission_date": "2023-12-12", "location": "Queens Warehouse", "threat_level": 2, "duration_min": 14, "outcome": "success"},
    {"suit": "Iron Spider", "mission_date": "2024-01-20", "location": "New York City", "threat_level": 5, "duration_min": 54, "outcome": "success"},
    {"suit": "Iron Spider", "mission_date": "2024-02-05", "location": "Oscorp Lab", "threat_level": 5, "duration_min": 61, "outcome": "partial"},
    {"suit": "Iron Spider", "mission_date": "2024-03-10", "location": "Midtown High", "threat_level": 2, "duration_min": 12, "outcome": "success"},
    {"suit": "Iron Spider", "mission_date": "2024-02-27", "location": "Statue of Liberty", "threat_level": 3, "duration_min": 19, "outcome": "success"},
    {"suit": "Advanced Suit", "mission_date": "2024-01-12", "location": "Washington Monument", "threat_level": 5, "duration_min": 47, "outcome": "success"},
    {"suit": "Advanced Suit", "mission_date": "2024-02-22", "location": "London Bridge", "threat_level": 5, "duration_min": 58, "outcome": "success"},
    {"suit": "Advanced Suit", "mission_date": "2024-03-05", "location": "Stark Cargo Plane", "threat_level": 4, "duration_min": 33, "outcome": "success"},
    {"suit": "Advanced Suit", "mission_date": "2023-12-30", "location": "Peter's Neighborhood", "threat_level": 1, "duration_min": 8, "outcome": "success"},
    {"suit": "Advanced Suit", "mission_date": "2024-01-15", "location": "Oscorp Vault", "threat_level": 4, "duration_min": 36, "outcome": "aborted"},
    {"suit": "Stealth Suit", "mission_date": "2024-01-08", "location": "Washington Monument", "threat_level": 5, "duration_min": 49, "outcome": "success"},
    {"suit": "Stealth Suit", "mission_date": "2024-02-14", "location": "Grand Central", "threat_level": 4, "duration_min": 30, "outcome": "partial"},
    {"suit": "Stealth Suit", "mission_date": "2024-03-18", "location": "Daily Bugle Rooftop", "threat_level": 2, "duration_min": 20, "outcome": "success"},
    {"suit": "Stealth Suit", "mission_date": "2024-03-22", "location": "Central Park", "threat_level": 3, "duration_min": 28, "outcome": "success"},
    {"suit": "Homemade Suit", "mission_date": "2023-11-10", "location": "Ferry Split Site", "threat_level": 3, "duration_min": 25, "outcome": "success"},
    {"suit": "Homemade Suit", "mission_date": "2024-01-25", "location": "Stark Jet Crash Site", "threat_level": 1, "duration_min": 10, "outcome": "success"},
    {"suit": "Homemade Suit", "mission_date": "2024-02-01", "location": "Aunt May's Neighborhood", "threat_level": 1, "duration_min": 9, "outcome": "success"},
]
