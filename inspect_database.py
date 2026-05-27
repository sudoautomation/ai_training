"""Inspect raw documents stored in the database."""

import os
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row

load_dotenv()

PG_CONN = os.getenv("PG_CONNECTION_STRING", "").replace("postgresql+psycopg2", "postgresql")
COLLECTION_NAME = "hr_support_desk"

def inspect_database():
    """Show raw documents from database."""
    print(f"\n{'='*80}")
    print(f"DATABASE CONTENTS INSPECTION")
    print(f"{'='*80}")
    print(f"Collection: {COLLECTION_NAME}\n")
    
    sql_sample = """
        SELECT 
            e.id,
            e.document AS content,
            e.cmetadata AS metadata,
            char_length(e.document) as length
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON c.uuid = e.collection_id
        WHERE c.name = %(collection)s
        ORDER BY e.id
        LIMIT 10;
    """
    
    try:
        with psycopg.connect(PG_CONN, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_sample, {"collection": COLLECTION_NAME})
                docs = cur.fetchall()
                
                print(f"Sample of {len(docs)} documents:\n")
                
                for i, doc in enumerate(docs, 1):
                    content = doc["content"].replace("\n", " ")[:150]
                    meta = doc["metadata"]
                    
                    print(f"[{i}] ID: {doc['id']}")
                    print(f"    Length: {doc['length']} chars")
                    print(f"    Content: {content}...")
                    print(f"    Metadata: {meta}")
                    print()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    inspect_database()
