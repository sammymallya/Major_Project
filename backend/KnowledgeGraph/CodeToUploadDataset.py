import json
from neo4j import GraphDatabase

# ====== YOUR AURA DETAILS ======
NEO4J_URI = "neo4j+s://0e1913a5.databases.neo4j.io"
NEO4J_USERNAME = "0e1913a5"
NEO4J_PASSWORD = "_wD_m3FFsY4CfPWI3Rv7qMGT--bjoPjmSLr-77itHA0"
# =================================

JSON_FILE = "Tourist_updated_dataset.json"


def insert_place(tx, record):

    tx.run("""
    MERGE (p:Place {id: $id})
    SET p.name = $name,
        p.type = $type,
        p.description = $description,
        p.best_time = $best_time,
        p.entry_fee = $entry_fee

    MERGE (c:City {name: $city})
    MERGE (d:District {name: $district})
    MERGE (s:State {name: $state})

    MERGE (p)-[:LOCATED_IN]->(c)
    MERGE (c)-[:IN_DISTRICT]->(d)
    MERGE (d)-[:IN_STATE]->(s)
    """,
    id=record.get("id"),
    name=record.get("name"),
    type=record.get("type"),
    description=record.get("description"),
    best_time=record.get("best_time_to_visit"),
    entry_fee=record.get("entry_fee"),
    city=record.get("city"),
    district=record.get("district"),
    state=record.get("state")
    )

    # Insert activities
    for activity in record.get("activities", []):
        tx.run("""
        MATCH (p:Place {id: $id})
        MERGE (a:Activity {name: $activity})
        MERGE (p)-[:HAS_ACTIVITY]->(a)
        """,
        id=record.get("id"),
        activity=activity
        )

    # Insert nearby places (will connect if they exist later)
    for nearby in record.get("nearby_places", []):
        tx.run("""
        MATCH (p1:Place {id: $id})
        MERGE (p2:Place {name: $nearby})
        MERGE (p1)-[:NEARBY]->(p2)
        """,
        id=record.get("id"),
        nearby=nearby
        )


def push_dataset():

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    driver.verify_connectivity()
    print("✅ Connected to Neo4j Aura")

    with driver.session() as session:
        for record in data:
            session.execute_write(insert_place, record)

    driver.close()
    print("✅ Full dataset uploaded successfully!")


if __name__ == "__main__":
    push_dataset()
