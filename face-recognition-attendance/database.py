import sqlite3
import os
import numpy as np
from datetime import datetime
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from werkzeug.security import generate_password_hash, check_password_hash

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "database.db")
EXCEL_DIR = os.path.join(DB_DIR, "excel")

def get_db_connection():
    """Establish connection to SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database and directories, and create default admin if not exists"""
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(EXCEL_DIR, exist_ok=True)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # 1. Create Users table (Admins/Teachers)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'teacher',
        date_created TEXT NOT NULL
    );
    """)
    
    # 2. Create Classes table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS classes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    );
    """)
    
    # 3. Create Students table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        roll_number TEXT UNIQUE NOT NULL,
        class_id INTEGER NOT NULL,
        face_embedding BLOB NOT NULL,
        date_enrolled TEXT NOT NULL,
        FOREIGN KEY (class_id) REFERENCES classes(id) ON DELETE CASCADE
    );
    """)
    
    # 4. Create Attendance Sessions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        class_id INTEGER NOT NULL,
        subject TEXT NOT NULL,
        teacher_name TEXT NOT NULL,
        date_time TEXT NOT NULL,
        FOREIGN KEY (class_id) REFERENCES classes(id) ON DELETE CASCADE
    );
    """)
    
    # 5. Create Attendance Records table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        student_id INTEGER NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('P', 'A')),
        time_marked TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES attendance_sessions(id) ON DELETE CASCADE,
        FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
        UNIQUE(session_id, student_id)
    );
    """)
    
    # Add default admin if users table is empty
    cursor.execute("SELECT COUNT(*) FROM users;")
    if cursor.fetchone()[0] == 0:
        admin_hash = generate_password_hash("admin123")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, date_created) VALUES (?, ?, ?, ?);",
            ("admin", admin_hash, "admin", now_str)
        )
        print("Default admin created (username: admin, password: admin123)")
        
    conn.commit()
    conn.close()

# --- User Management ---

def create_user(username, password, role="teacher"):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        password_hash = generate_password_hash(password)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, date_created) VALUES (?, ?, ?, ?);",
            (username, password_hash, role, now_str)
        )
        conn.commit()
        return True, "User created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?;", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user["password_hash"], password):
        return dict(user)
    return None

# --- Class Operations ---

def create_class(name):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO classes (name) VALUES (?);", (name,))
        conn.commit()
        return True, cursor.lastrowid
    except sqlite3.IntegrityError:
        return False, "Class name already exists."
    finally:
        conn.close()

def get_classes():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM classes ORDER BY name ASC;")
    classes = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return classes

def get_class_by_id(class_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM classes WHERE id = ?;", (class_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# --- Student Operations ---

def add_student(name, roll_number, class_id, face_embedding_array):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Convert numpy float32 embedding vector to raw bytes for BLOB storage
        embedding_blob = face_embedding_array.astype(np.float32).tobytes()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO students (name, roll_number, class_id, face_embedding, date_enrolled) VALUES (?, ?, ?, ?, ?);",
            (name, roll_number, class_id, embedding_blob, now_str)
        )
        conn.commit()
        # Trigger Excel sync since new student is added
        sync_class_attendance_to_excel(class_id)
        return True, "Student registered successfully."
    except sqlite3.IntegrityError:
        return False, "Roll number already registered."
    finally:
        conn.close()

def get_students_by_class(class_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE class_id = ? ORDER BY roll_number ASC;", (class_id,))
    rows = cursor.fetchall()
    conn.close()
    
    students = []
    for row in rows:
        d = dict(row)
        # Reconstruct numpy array from BLOB
        d["face_embedding"] = np.frombuffer(d["face_embedding"], dtype=np.float32)
        students.append(d)
    return students

def delete_student(student_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT class_id FROM students WHERE id = ?;", (student_id,))
    row = cursor.fetchone()
    class_id = row["class_id"] if row else None
    
    cursor.execute("DELETE FROM students WHERE id = ?;", (student_id,))
    conn.commit()
    conn.close()
    
    if class_id:
        sync_class_attendance_to_excel(class_id)
    return True

# --- Attendance Operations ---

def create_attendance_session(class_id, subject, teacher_name, date_time=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if not date_time:
        date_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        
    cursor.execute(
        "INSERT INTO attendance_sessions (class_id, subject, teacher_name, date_time) VALUES (?, ?, ?, ?);",
        (class_id, subject, teacher_name, date_time)
    )
    session_id = cursor.lastrowid
    
    # For every student in this class, initially mark them as Absent 'A'
    cursor.execute("SELECT id FROM students WHERE class_id = ?;", (class_id,))
    students = cursor.fetchall()
    
    for student in students:
        cursor.execute(
            "INSERT OR IGNORE INTO attendance_records (session_id, student_id, status, time_marked) VALUES (?, ?, 'A', ?);",
            (session_id, student["id"], "N/A")
        )
        
    conn.commit()
    conn.close()
    sync_class_attendance_to_excel(class_id)
    return session_id

def mark_student_present(session_id, student_id, time_marked=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if not time_marked:
        time_marked = datetime.now().strftime("%H:%M:%S")
        
    # Get student's class_id to trigger excel sync
    cursor.execute("SELECT class_id FROM students WHERE id = ?;", (student_id,))
    student_row = cursor.fetchone()
    class_id = student_row["class_id"] if student_row else None
    
    cursor.execute(
        """
        INSERT INTO attendance_records (session_id, student_id, status, time_marked) 
        VALUES (?, ?, 'P', ?)
        ON CONFLICT(session_id, student_id) DO UPDATE SET status='P', time_marked=excluded.time_marked;
        """,
        (session_id, student_id, time_marked)
    )
    conn.commit()
    conn.close()
    
    if class_id:
        sync_class_attendance_to_excel(class_id)
    return True

def get_attendance_session_details(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, c.name as class_name 
        FROM attendance_sessions s 
        JOIN classes c ON s.class_id = c.id 
        WHERE s.id = ?;
    """, (session_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_attendance_session_records(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT r.status, r.time_marked, s.name, s.roll_number, s.id as student_id
        FROM students s
        LEFT JOIN attendance_records r ON s.id = r.student_id AND r.session_id = ?
        WHERE s.class_id = (SELECT class_id FROM attendance_sessions WHERE id = ?)
        ORDER BY s.roll_number ASC;
    """, (session_id, session_id))
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records

def get_all_sessions_by_class(class_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, 
               (SELECT COUNT(*) FROM attendance_records r WHERE r.session_id = s.id AND r.status = 'P') as present_count,
               (SELECT COUNT(*) FROM attendance_records r WHERE r.session_id = s.id) as total_count
        FROM attendance_sessions s
        WHERE s.class_id = ?
        ORDER BY s.date_time DESC;
    """, (class_id,))
    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return sessions

# --- Excel Sync Engine ---

def sync_class_attendance_to_excel(class_id):
    """Generates and updates a matrix-styled Excel grid for a class."""
    cls = get_class_by_id(class_id)
    if not cls:
        return
        
    class_name = cls["name"]
    # Clean filename of unsafe characters
    safe_name = "".join([c if c.isalnum() or c in " _-" else "_" for c in class_name])
    filepath = os.path.join(EXCEL_DIR, f"{safe_name}_attendance.xlsx")
    
    # 1. Fetch Students
    students = get_students_by_class(class_id)
    if not students:
        # If no students, remove or don't create Excel sheet
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
        return
        
    # 2. Fetch Sessions
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, subject, date_time 
        FROM attendance_sessions 
        WHERE class_id = ? 
        ORDER BY date_time ASC;
    """, (class_id,))
    sessions = cursor.fetchall()
    
    # 3. Build Attendance Matrix
    data = []
    for s in students:
        row_data = {
            "Roll Number": s["roll_number"],
            "Student Name": s["name"]
        }
        for sess in sessions:
            # Query attendance record for this student & session
            cursor.execute(
                "SELECT status FROM attendance_records WHERE session_id = ? AND student_id = ?;",
                (sess["id"], s["id"])
            )
            rec = cursor.fetchone()
            # If session exists but student has no record, default to Absent 'A'
            status = rec["status"] if rec else "A"
            # Format column header: e.g. "2026-08-09 (Database)"
            dt_formatted = sess["date_time"].split()[0] # get YYYY-MM-DD
            col_name = f"{dt_formatted}\n({sess['subject']})"
            row_data[col_name] = status
            
        data.append(row_data)
        
    conn.close()
    
    df = pd.DataFrame(data)
    
    # 4. Write to Excel & Apply Premium Styling
    df.to_excel(filepath, index=False)
    
    # Apply styling using openpyxl
    wb = load_workbook(filepath)
    ws = wb.active
    ws.title = "Attendance Matrix"
    
    # Stylings
    navy_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    white_font = Font(name="Times New Roman", size=11, bold=True, color="FFFFFF")
    regular_font = Font(name="Times New Roman", size=10)
    bold_font = Font(name="Times New Roman", size=10, bold=True)
    
    # Success/Failure highlighting
    present_fill = PatternFill(start_color="D9EAD3", end_color="D9EAD3", fill_type="solid") # Light soft green
    present_font = Font(name="Times New Roman", size=10, bold=True, color="274E13") # Dark green
    
    absent_fill = PatternFill(start_color="F4CCCC", end_color="F4CCCC", fill_type="solid") # Light soft red
    absent_font = Font(name="Times New Roman", size=10, bold=True, color="660000") # Dark red
    
    thin_side = Side(border_style="thin", color="D3D3D3")
    grid_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
    
    # Center alignment for grid cells, left for name
    center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left_align = Alignment(horizontal="left", vertical="center")
    
    # Set Row Heights
    ws.row_dimensions[1].height = 40 # Header row
    for row in range(2, len(students) + 2):
        ws.row_dimensions[row].height = 24
        
    # Style Header Row
    for col in range(1, len(df.columns) + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = navy_fill
        cell.font = white_font
        cell.alignment = center_align
        cell.border = grid_border
        
    # Style Data Cells
    for row in range(2, len(students) + 2):
        # Zebra striping for backgrounds
        bg_color = "F9FBFD" if row % 2 == 0 else "FFFFFF"
        row_fill = PatternFill(start_color=bg_color, end_color=bg_color, fill_type="solid")
        
        for col in range(1, len(df.columns) + 1):
            cell = ws.cell(row=row, column=col)
            val = cell.value
            cell.font = regular_font
            cell.border = grid_border
            cell.fill = row_fill
            
            # Align columns
            if col == 1: # Roll number
                cell.alignment = center_align
                cell.font = bold_font
            elif col == 2: # Name
                cell.alignment = left_align
            else: # Date status cells (P / A)
                cell.alignment = center_align
                if val == 'P':
                    cell.fill = present_fill
                    cell.font = present_font
                elif val == 'A':
                    cell.fill = absent_fill
                    cell.font = absent_font
                    
    # Auto-adjust column widths
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        
        # Check lengths of cell strings
        for cell in col:
            val_str = str(cell.value or '')
            # If cell has a newline (like column headers), split and get max line length
            lines = val_str.split('\n')
            max_len = max(max_len, max(len(l) for l in lines))
            
        # Give some padding
        ws.column_dimensions[col_letter].width = max(max_len + 4, 12)
        
    # Set Roll Number and Name columns to standard widths if small
    ws.column_dimensions['A'].width = 15 # Roll number
    ws.column_dimensions['B'].width = 25 # Name
    
    # Save the styled Excel
    wb.save(filepath)

if __name__ == "__main__":
    # Test initialization
    init_db()
    print("Database initialised successfully.")
