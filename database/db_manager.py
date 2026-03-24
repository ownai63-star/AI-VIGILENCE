import sqlite3
import os

class DatabaseManager:
    def __init__(self, db_path='database/system.db'):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Table for registered persons
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS registered_persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    image_path TEXT,
                    encoding BLOB
                )
            ''')
            # Table for detections
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    camera_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT,
                    FOREIGN KEY (person_id) REFERENCES registered_persons (id)
                )
            ''')
            conn.commit()

    def register_person(self, name, image_path, encoding):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO registered_persons (name, image_path, encoding) VALUES (?, ?, ?)',
                (name, image_path, encoding)
            )
            conn.commit()
            return cursor.lastrowid

    def log_detection(self, person_id, camera_id, image_path):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO detections (person_id, camera_id, image_path) VALUES (?, ?, ?)',
                (person_id, camera_id, image_path)
            )
            conn.commit()

    def update_detection_person(self, camera_id, image_path, person_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE detections SET person_id = ? WHERE camera_id = ? AND image_path = ?',
                (person_id, camera_id, image_path)
            )
            conn.commit()

    def get_registered_persons(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM registered_persons')
            return cursor.fetchall()

    def search_detections(self, name=None, start_time=None, end_time=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT d.*, rp.name 
                FROM detections d 
                LEFT JOIN registered_persons rp ON d.person_id = rp.id
                WHERE 1=1
            '''
            params = []
            if name:
                query += " AND rp.name LIKE ?"
                params.append(f"%{name}%")
            if start_time:
                query += " AND d.timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND d.timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY d.timestamp DESC"
            cursor.execute(query, params)
            return cursor.fetchall()
