import sqlite3
from threading import Lock

# Database file path
CACHE_DB = "cache.db"

# Cache size limit
MAX_CACHE_SIZE = 180  # Adjust as needed

# Lock for thread-safe operations
lock = Lock()

def init_cache():
    """
    Initialize the SQLite database for caching.
    Creates the cache table if it doesn't exist.
    """
    with lock:
        with sqlite3.connect(CACHE_DB) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    tag_description TEXT,
                    unit TEXT,
                    thing TEXT,
                    property TEXT,
                    frequency INTEGER,
                    PRIMARY KEY (tag_description, unit)
                )
            ''')
            conn.commit()

def get_cache(key):
    """
    Retrieve cache entry for a composite key (tag_description, unit).

    Args:
        key (tuple): The composite key (tag_description, unit).

    Returns:
        tuple: (thing, property) if found, (None, None) otherwise.
    """
    tag_description, unit = key
    with lock:
        with sqlite3.connect(CACHE_DB) as conn:
            cursor = conn.execute('''
                SELECT thing, property, frequency FROM cache
                WHERE tag_description = ? AND unit = ?
            ''', (tag_description, unit))
            row = cursor.fetchone()
            if row:
                print(f"Cache hit for {key}: {row}")
                conn.execute('''
                    UPDATE cache SET frequency = frequency + 1
                    WHERE tag_description = ? AND unit = ?
                ''', (tag_description, unit))
                conn.commit()
                return row[0], row[1]
            print(f"Cache miss for {key}")
            return None, None

def update_cache(key, thing, property_):
    """
    Update the cache with a composite key (tag_description, unit).

    Args:
        key (tuple): The composite key (tag_description, unit).
        thing (str): The thing to cache.
        property_ (str): The property to cache.
    """
    tag_description, unit = key
    with lock:
        with sqlite3.connect(CACHE_DB) as conn:
            # Perform eviction and update within the same transaction
            cursor = conn.execute('SELECT COUNT(*) FROM cache')
            count = cursor.fetchone()[0]
            if count >= MAX_CACHE_SIZE:
                conn.execute('''
                    DELETE FROM cache
                    WHERE (tag_description, unit) = (
                        SELECT tag_description, unit
                        FROM cache
                        ORDER BY frequency ASC
                        LIMIT 1
                    )
                ''')

            # Insert or update the cache entry
            conn.execute('''
                INSERT INTO cache (tag_description, unit, thing, property, frequency)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(tag_description, unit) DO UPDATE SET
                thing = excluded.thing,
                property = excluded.property,
                frequency = cache.frequency + 1
            ''', (tag_description, unit, thing, property_))
            conn.commit()

def has_cache_key(key):
    """
    Check if a composite key (tag_description, unit) exists in the SQLite cache.

    Args:
        key (tuple): The composite key (tag_description, unit).

    Returns:
        bool: True if the key exists in the cache, False otherwise.
    """
    tag_description, unit = key
    with lock:
        with sqlite3.connect(CACHE_DB) as conn:
            cursor = conn.execute('''
                SELECT 1 FROM cache WHERE tag_description = ? AND unit = ? LIMIT 1
            ''', (tag_description, unit))
            return cursor.fetchone() is not None
