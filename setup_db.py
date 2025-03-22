import sqlite3

# Connect to (or create) the database
conn = sqlite3.connect("books.db")
cursor = conn.cursor()

# Create table for storing book metadata
cursor.execute("""
    CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        author TEXT,
        year INTEGER,
        isbn TEXT UNIQUE,
        olid TEXT UNIQUE,
        source TEXT CHECK(source IN ('Open Library', 'Google Books')),
        license TEXT CHECK(license IN ('Public Domain', 'Creative Commons')),
        full_text_available BOOLEAN DEFAULT 0
    )
""")

# Create table for storing full text
cursor.execute("""
    CREATE TABLE IF NOT EXISTS full_texts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id INTEGER,
        text_content TEXT NOT NULL,
        FOREIGN KEY (book_id) REFERENCES books (id) ON DELETE CASCADE
    )
""")

# Create table for storing images
cursor.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id INTEGER,
        image_type TEXT CHECK(image_type IN ('cover', 'illustration')),
        image_data BLOB,
        FOREIGN KEY (book_id) REFERENCES books (id) ON DELETE CASCADE
    )
""")

# Save changes and close connection
conn.commit()
conn.close()

print("📚 Database setup complete! Tables created.")
