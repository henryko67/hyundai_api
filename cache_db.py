import sqlite3
import os
from threading import Lock

# Cache size limit
MAX_CACHE_SIZE = 500  # Adjust as needed
lock = Lock()

def get_cache_db_path(ships_idx):
    """
    Generate a cache database file path based on ships_idx.
    """
    cache_dir = "cache_files"  # Ensure this directory exists
    os.makedirs(cache_dir, exist_ok=True)  # Create directory if it doesn't exist
    return os.path.join(cache_dir, f"cache_ship_{ships_idx}.db")

def normalize_key(tag_description, unit):
    """
    Normalize keys for caching and retrieval.
    """
    normalized_description = tag_description.strip().upper().replace(" ", "_")
    normalized_unit = unit.strip().upper()
    return normalized_description, normalized_unit

def print_cache_contents(ships_idx):
    """
    Print the contents of the cache database for a specific ship.
    """
    cache_db_path = get_cache_db_path(ships_idx)

    if not os.path.exists(cache_db_path):
        print(f"No cache database found for ship {ships_idx} at {cache_db_path}")
        return

    print(f"Contents of cache database for ship {ships_idx} ({cache_db_path}):")
    try:
        with sqlite3.connect(cache_db_path) as conn:
            cursor = conn.execute('SELECT tag_description, unit, thing, property, frequency FROM cache')
            rows = cursor.fetchall()

            if rows:
                print(f"{'Tag Description':<30} {'Unit':<10} {'Thing':<20} {'Property':<20} {'Frequency':<10}")
                print("-" * 90)
                for row in rows:
                    tag_description = row[0] if row[0] else "NULL"
                    unit = row[1] if row[1] else "NULL"
                    thing = row[2] if row[2] else "NULL"
                    property_ = row[3] if row[3] else "NULL"
                    frequency = row[4] if row[4] else 0

                    print(f"{tag_description:<30} {unit:<10} {thing:<20} {property_:<20} {frequency:<10}")
            else:
                print("Cache is empty.")
    except sqlite3.DatabaseError as e:
        print(f"Error reading database for ship {ships_idx}: {e}")

def init_cache(ships_idx):
    """
    Initialize the SQLite database for caching for a specific ship.
    """
    cache_db_path = get_cache_db_path(ships_idx)
    with lock:
        with sqlite3.connect(cache_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    tag_description TEXT,
                    unit TEXT,
                    thing TEXT,
                    property TEXT,
                    frequency INTEGER DEFAULT 1,
                    PRIMARY KEY (tag_description, unit)
                )
            ''')
            conn.commit()
            print(f"Initialized cache for ship {ships_idx} at {cache_db_path}.")

def has_cache_key(ships_idx, key):
    """
    Check if a composite key (tag_description, unit) exists in the SQLite cache for a specific ship.
    """
    tag_description, unit = key
    cache_db_path = get_cache_db_path(ships_idx)
    with lock:
        with sqlite3.connect(cache_db_path) as conn:
            cursor = conn.execute('''
                SELECT 1 FROM cache WHERE tag_description = ? AND unit = ? LIMIT 1
            ''', (tag_description, unit))
            exists = cursor.fetchone() is not None
            print(f"Cache key {'exists' if exists else 'does not exist'} for {key} in ship {ships_idx}.")
            return exists

def get_cache(ships_idx, key):
    """
    Retrieve cache entry for a composite key (tag_description, unit) for a specific ship.
    """
    tag_description, unit = key
    cache_db_path = get_cache_db_path(ships_idx)
    with lock:
        with sqlite3.connect(cache_db_path) as conn:
            cursor = conn.execute('''
                SELECT thing, property, frequency FROM cache
                WHERE tag_description = ? AND unit = ?
            ''', (tag_description, unit))
            row = cursor.fetchone()
            if row:
                print(f"Cache hit for {key} in ship {ships_idx}: thing={row[0]}, property={row[1]}")
                conn.execute('''
                    UPDATE cache SET frequency = frequency + 1
                    WHERE tag_description = ? AND unit = ?
                ''', (tag_description, unit))
                conn.commit()
                return row[0], row[1]
            print(f"Cache miss for {key} in ship {ships_idx}.")
            return None, None

def update_cache(ships_idx, key, thing, property_):
    """
    Insert or update a cache entry with the given key, thing, and property.
    If the cache size exceeds MAX_CACHE_SIZE, evict the least frequently used entry.
    """
    tag_description, unit = key
    cache_db_path = get_cache_db_path(ships_idx)
    with lock:
        with sqlite3.connect(cache_db_path) as conn:
            # Check the current size of the cache
            cursor = conn.execute('SELECT COUNT(*) FROM cache')
            current_size = cursor.fetchone()[0]

            if current_size >= MAX_CACHE_SIZE:
                # Evict the least frequently used entry
                conn.execute('''
                    DELETE FROM cache
                    WHERE rowid IN (
                        SELECT rowid
                        FROM cache
                        ORDER BY frequency ASC, rowid ASC
                        LIMIT 1
                    )
                ''')
                print(f"Evicted least frequently used entry from ship {ships_idx} cache.")

            # Insert or update the cache entry
            try:
                conn.execute('''
                    INSERT INTO cache (tag_description, unit, thing, property, frequency)
                    VALUES (?, ?, ?, ?, 1)
                    ON CONFLICT(tag_description, unit) DO UPDATE SET
                        thing = excluded.thing,
                        property = excluded.property,
                        frequency = cache.frequency + 1
                ''', (tag_description, unit, thing, property_))
                conn.commit()
                print(f"[SUCCESS] Updated cache for ship {ships_idx}, key {key}: thing={thing}, property={property_}.")
            except sqlite3.DatabaseError as e:
                print(f"[ERROR] Failed to update cache for ship {ships_idx}, key {key}: {e}")
