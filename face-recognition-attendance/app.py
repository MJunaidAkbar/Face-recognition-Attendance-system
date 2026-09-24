import os
import cv2
import io
import time
import threading
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify, send_file
from werkzeug.security import generate_password_hash

import database
from face_engine import FaceEngine

app = Flask(__name__)
app.secret_key = "auragate_secure_local_secret_key_63"

# Initialize Face Engine and Database
try:
    face_engine = FaceEngine()
except Exception as e:
    print(f"Error loading Face Engine: {e}. Running download script first...")
    # Attempt to download models inline if missing
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
    import download_models
    download_models.download_models()
    face_engine = FaceEngine()

database.init_db()

# Create data directories
PDF_DIR = os.path.join(database.DB_DIR, "pdf")
os.makedirs(PDF_DIR, exist_ok=True)


# --- Thread-safe Camera Manager ---
class CameraManager:
    def __init__(self):
        self.video = None
        self.latest_frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.clients_count = 0
        self.clients_lock = threading.Lock()
        self.camera_index = 0

    def start(self):
        with self.lock:
            if not self.running:
                # 1. Attempt to open selected camera index
                self.video = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(self.camera_index)
                if not self.video.isOpened():
                    self.video = cv2.VideoCapture(self.camera_index)
                    
                # 2. Dynamic Fallback Port Scanner:
                # If selected index fails, scan indices 0 to 4 for any active camera handle
                if not self.video.isOpened():
                    print(f"Camera index {self.camera_index} failed to open. Scanning fallback ports...")
                    fallback_index = None
                    for idx in range(5):
                        if idx == self.camera_index:
                            continue
                        temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(idx)
                        if not temp_cap.isOpened():
                            temp_cap = cv2.VideoCapture(idx)
                            
                        if temp_cap.isOpened():
                            temp_cap.release()
                            fallback_index = idx
                            print(f"Fallback operational camera detected at port {idx}!")
                            break
                            
                    if fallback_index is not None:
                        self.camera_index = fallback_index
                        self.video = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(self.camera_index)
                        if not self.video.isOpened():
                            self.video = cv2.VideoCapture(self.camera_index)
                    else:
                        print("Error: No active camera indices found on this system.")
                        self.running = False
                        return False
                
                self.running = True
                self.thread = threading.Thread(target=self._update, name="WebcamThread", daemon=True)
                self.thread.start()
                print(f"Webcam reader thread started for index {self.camera_index}.")
        return True

    def stop(self):
        with self.lock:
            if self.running:
                self.running = False
                if self.thread:
                    self.thread.join(timeout=1.0)
                if self.video:
                    self.video.release()
                    self.video = None
                self.latest_frame = None
                print("Webcam reader thread stopped and camera released.")

    def set_camera_index(self, index):
        with self.lock:
            if self.camera_index == index:
                return True
            self.camera_index = index
            print(f"Camera index updated to {index}.")
            
        if self.running:
            self.stop()
            return self.start()
        return True

    def _update(self):
        while self.running:
            if self.video:
                success, frame = self.video.read()
                if success:
                    # Mirror horizontally for natural view
                    frame = cv2.flip(frame, 1)
                    with self.lock:
                        self.latest_frame = frame
            time.sleep(0.03) # Limit reading to ~30 FPS

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def add_client(self):
        with self.clients_lock:
            self.clients_count += 1
            if self.clients_count == 1:
                self.start()

    def remove_client(self):
        with self.clients_lock:
            self.clients_count = max(0, self.clients_count - 1)
            if self.clients_count == 0:
                # Give a 3 second grace period before closing in case of page refreshes
                threading.Thread(target=self._delayed_stop, daemon=True).start()

    def _delayed_stop(self):
        time.sleep(3.0)
        with self.clients_lock:
            if self.clients_count == 0:
                self.stop()

# Instantiate global camera manager
camera_manager = CameraManager()

# Global Cache to store face vectors during student enrollment before saving
# Key: roll_number -> Value: {direction: embedding_array}
enrollment_embeddings_cache = {}


# --- Login Protection Middleware ---
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash("You must be logged in to view that page.", "danger")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# --- Flask Web Application Routes ---

@app.route('/')
def attendance_hud():
    """Main Public View: Attendance Scanner HUD"""
    active_session_id = session.get('active_session_id')
    session_details = None
    if active_session_id:
        session_details = database.get_attendance_session_details(active_session_id)
        # If session detail doesn't exist anymore, clear it
        if not session_details:
            session.pop('active_session_id', None)
            active_session_id = None
            
    classes = database.get_classes()
    return render_template(
        'attendance.html', 
        session_id=active_session_id, 
        session_details=session_details,
        classes=classes
    )

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Sign-in route for authorized personnel"""
    if 'user' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        user = database.verify_user(username, password)
        if user:
            session['user'] = {
                'id': user['id'],
                'username': user['username'],
                'role': user['role']
            }
            flash("Successfully logged in!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password.", "danger")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout of active session"""
    session.pop('user', None)
    flash("Successfully logged out.", "success")
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Main Admin panel"""
    classes = database.get_classes()
    
    # Retrieve stats
    conn = database.get_db_connection()
    cursor = conn.cursor()
    students_count = cursor.execute("SELECT COUNT(*) FROM students;").fetchone()[0]
    sessions_count = cursor.execute("SELECT COUNT(*) FROM attendance_sessions;").fetchone()[0]
    conn.close()
    
    return render_template(
        'dashboard.html', 
        classes=classes, 
        students_count=students_count,
        sessions_count=sessions_count
    )


# --- Class Actions ---

@app.route('/dashboard/add_class', methods=['POST'])
@login_required
def add_class_route():
    class_name = request.form.get('class_name', '').strip()
    if class_name:
        success, msg = database.create_class(class_name)
        if success:
            flash(f"Class '{class_name}' successfully created!", "success")
        else:
            flash(f"Error: {msg}", "danger")
    return redirect(url_for('dashboard'))


# --- Roster APIs ---

@app.route('/api/students')
@login_required
def get_students_api():
    class_id = request.args.get('class_id')
    if not class_id:
        return jsonify([])
    students = database.get_students_by_class(class_id)
    # Serialize numpy arrays for JSON output
    serialized = []
    for s in students:
        serialized.append({
            "id": s["id"],
            "name": s["name"],
            "roll_number": s["roll_number"],
            "date_enrolled": s["date_enrolled"]
        })
    return jsonify(serialized)

@app.route('/api/students/delete', methods=['POST'])
@login_required
def delete_student_api():
    data = request.get_json() or {}
    student_id = data.get('student_id')
    if not student_id:
        return jsonify({"success": False, "message": "Missing student ID"})
        
    database.delete_student(student_id)
    return jsonify({"success": True, "message": "Student roster profile deleted"})

@app.route('/api/sessions')
@login_required
def get_sessions_api():
    class_id = request.args.get('class_id')
    if not class_id:
        return jsonify([])
    sessions = database.get_all_sessions_by_class(class_id)
    return jsonify(sessions)


# --- Face Enrollment Endpoints ---

@app.route('/enroll/capture_angle', methods=['POST'])
@login_required
def capture_angle():
    data = request.get_json() or {}
    roll_number = data.get('roll_number', '').strip()
    direction = data.get('direction', '').strip()
    
    if not roll_number or not direction:
        return jsonify({"success": False, "message": "Missing roll number or direction"})
        
    # Grab the current frame from camera manager
    frame = camera_manager.get_frame()
    if frame is None:
        return jsonify({"success": False, "message": "Camera stream is initializing. Please wait."})
        
    # Detect face
    faces = face_engine.detect_faces(frame)
    if len(faces) == 0:
        return jsonify({"success": False, "message": "No face detected in camera viewport. Adjust lighting/position."})
    elif len(faces) > 1:
        return jsonify({"success": False, "message": "Multiple faces detected. Only the enrolling student should be in frame."})
        
    # Exactly one face detected
    face_meta = faces[0]
    embedding = face_engine.extract_embedding(frame, face_meta["landmarks"])
    
    if embedding is None:
        return jsonify({"success": False, "message": "Failed to align face crop or map features. Try again."})
        
    # Save feature embedding to local server memory cache
    if roll_number not in enrollment_embeddings_cache:
        enrollment_embeddings_cache[roll_number] = {}
        
    enrollment_embeddings_cache[roll_number][direction] = embedding
    return jsonify({"success": True, "message": f"Successfully mapped head angle '{direction}'!"})

@app.route('/enroll/save_student', methods=['POST'])
@login_required
def save_student():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    roll_number = data.get('roll', '').strip()
    class_id = data.get('classId')
    
    if not name or not roll_number or not class_id:
        return jsonify({"success": False, "message": "Missing name, roll number, or class ID"})
        
    # Verify cache has all 5 angles
    student_cache = enrollment_embeddings_cache.get(roll_number)
    required_angles = ["straight", "left", "right", "up", "down"]
    
    if not student_cache or any(ang not in student_cache for ang in required_angles):
        return jsonify({"success": False, "message": "Missing facial mappings. Map all 5 head angles first."})
        
    # Compile the 5 embeddings and compute the average
    embeddings = [student_cache[ang] for ang in required_angles]
    # Average the vectors along axis 0
    avg_embedding = np.mean(embeddings, axis=0)
    
    # L2-normalize the resulting average embedding vector to keep similarity mapping accurate
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm
        
    # Add to SQLite database
    success, msg = database.add_student(name, roll_number, class_id, avg_embedding)
    
    # Clear cache
    if roll_number in enrollment_embeddings_cache:
        del enrollment_embeddings_cache[roll_number]
        
    if success:
        return jsonify({"success": True, "message": f"Student '{name}' registered successfully."})
    else:
        return jsonify({"success": False, "message": msg})

@app.route('/enroll/clear_session', methods=['POST'])
@login_required
def clear_enrollment_session():
    # Helper to clean up memory cache if admin cancels halfway
    enrollment_embeddings_cache.clear()
    return jsonify({"success": True})


# --- Camera Attendance Session Control ---

@app.route('/attendance/start', methods=['POST'])
def start_session():
    class_id = request.form.get('class_id')
    subject = request.form.get('subject', '').strip()
    teacher_name = request.form.get('teacher_name', '').strip()
    
    if not class_id or not subject or not teacher_name:
        flash("Please fill in all session settings.", "danger")
        return redirect(url_for('attendance_hud'))
        
    session_id = database.create_attendance_session(class_id, subject, teacher_name)
    session['active_session_id'] = session_id
    flash(f"Attendance session initialized for '{subject}'!", "success")
    return redirect(url_for('attendance_hud'))

@app.route('/attendance/stop', methods=['POST'])
def stop_session():
    session.pop('active_session_id', None)
    flash("Attendance camera stopped. Session finalized and synced to local files.", "success")
    return redirect(url_for('attendance_hud'))

@app.route('/api/session_records')
def get_session_records_api():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify([])
    records = database.get_attendance_session_records(session_id)
    return jsonify(records)


# --- Video Streaming Source Engine ---

def generate_video_stream(mode, session_id=None):
    """
    Video streaming generator.
    Streams frames from global CameraManager and overlays real-time detection/recognition.
    """
    camera_manager.add_client()
    
    # Fetch enrolled student references once at session start to speed up comparison loops
    enrolled_students = []
    if mode == 'attendance' and session_id:
        sess_details = database.get_attendance_session_details(session_id)
        if sess_details:
            enrolled_students = database.get_students_by_class(sess_details["class_id"])
            
    try:
        last_processed_time = 0
        presents_cache = set() # Avoid hitting DB continuously for already marked students
        
        while True:
            frame = camera_manager.get_frame()
            if frame is None:
                # If camera is not running or failed to start, yield a clean placeholder frame and exit
                if not camera_manager.running:
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "CAMERA PORT OFFLINE", (140, 220), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(placeholder, f"Port {camera_manager.camera_index} is unavailable.", (160, 260), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(placeholder, "Please select another source above.", (140, 295), cv2.FONT_HERSHEY_DUPLEX, 0.5, (168, 85, 247), 1)
                    ret, jpeg = cv2.imencode('.jpg', placeholder)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    break
                time.sleep(0.05)
                continue
                
            current_time = time.time()
            # Throttle heavy detection/matching to ~15 FPS to prevent CPU thrashing
            if current_time - last_processed_time > 0.06:
                last_processed_time = current_time
                
                # Detect faces
                faces = face_engine.detect_faces(frame)
                
                for face in faces:
                    x, y, w, h = face["box"]
                    landmarks = face["landmarks"]
                    
                    if mode == 'attendance' and enrolled_students:
                        # Extract embedding and match
                        emb = face_engine.extract_embedding(frame, landmarks)
                        matched_student, score = face_engine.match_face(emb, enrolled_students)
                        
                        if matched_student:
                            student_name = matched_student["name"]
                            roll = matched_student["roll_number"]
                            student_id = matched_student["id"]
                            
                            # Mark present in database asynchronously
                            if student_id not in presents_cache:
                                database.mark_student_present(session_id, student_id)
                                presents_cache.add(student_id)
                                
                            # HUD Graphic - Green box for recognized students
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                            # Overlay name banner
                            cv2.rectangle(frame, (x, y - 30), (x + w, y), (0, 200, 0), -1)
                            cv2.putText(
                                frame, f"{student_name} ({int(score*100)}%)", 
                                (x + 5, y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1
                            )
                        else:
                            # HUD Graphic - Red/Yellow box for Unknown faces
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                            cv2.rectangle(frame, (x, y - 24), (x + w, y), (0, 165, 255), -1)
                            cv2.putText(
                                frame, "Unknown Face", 
                                (x + 5, y - 7), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1
                            )
                    else:
                        # Enrollment/Idle Mode - Just draw simple blue bounding boxes
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 128, 0), 2)
                        
            # Encode frame to JPEG format
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    finally:
        camera_manager.remove_client()

@app.route('/video_feed')
def video_feed():
    """Video feed route streaming MJPEG response"""
    mode = request.args.get('mode', 'idle')
    session_id = request.args.get('session_id')
    return Response(
        generate_video_stream(mode, session_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# --- File Exports (PDF & Excel) ---

@app.route('/export/pdf/<int:session_id>')
def download_pdf(session_id):
    """Compile and download ReportLab PDF attendance sheet"""
    sess = database.get_attendance_session_details(session_id)
    if not sess:
        return "Session not found", 404
        
    records = database.get_attendance_session_records(session_id)
    
    # 1. Prepare PDF Document
    pdf_filename = f"attendance_session_{session_id}.pdf"
    pdf_path = os.path.join(PDF_DIR, pdf_filename)
    
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    
    # Create Doc
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    elements = []
    
    # Stylings
    styles = getSampleStyleSheet()
    
    # Academic Header Fonts
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Times-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1F4E79'),
        alignment=1, # Center
        spaceAfter=5
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#555555'),
        alignment=1, # Center
        spaceAfter=15
    )
    
    meta_label_style = ParagraphStyle(
        'MetaLabel',
        fontName='Times-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#333333')
    )
    
    meta_val_style = ParagraphStyle(
        'MetaVal',
        fontName='Times-Roman',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#555555')
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        fontName='Times-Roman',
        fontSize=10,
        leading=12,
        textColor=colors.HexColor('#333333')
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        fontName='Times-Bold',
        fontSize=10,
        leading=12,
        textColor=colors.HexColor('#1F4E79')
    )
    
    table_cell_present = ParagraphStyle(
        'TableCellPresent',
        fontName='Times-Bold',
        fontSize=10,
        leading=12,
        textColor=colors.HexColor('#274E13') # Dark green
    )
    
    table_cell_absent = ParagraphStyle(
        'TableCellAbsent',
        fontName='Times-Bold',
        fontSize=10,
        leading=12,
        textColor=colors.HexColor('#660000') # Dark red
    )
    
    # 2. Add Header Titles
    elements.append(Paragraph("AuraGate Academic Portal", title_style))
    elements.append(Paragraph("OFFLINE OFFICIAL ATTENDANCE RECORD SHEET", subtitle_style))
    elements.append(Spacer(1, 10))
    
    # 3. Add Session Metadata Table
    presents_count = sum(1 for r in records if r["status"] == 'P')
    pct = round((presents_count / len(records) * 100) if len(records) > 0 else 0)
    
    meta_data = [
        [
            Paragraph("Class Name:", meta_label_style), Paragraph(sess["class_name"], meta_val_style),
            Paragraph("Subject Name:", meta_label_style), Paragraph(sess["subject"], meta_val_style)
        ],
        [
            Paragraph("Lecturer:", meta_label_style), Paragraph(sess["teacher_name"], meta_val_style),
            Paragraph("Session Date/Time:", meta_label_style), Paragraph(sess["date_time"], meta_val_style)
        ],
        [
            Paragraph("Roster Strength:", meta_label_style), Paragraph(f"{len(records)} Students", meta_val_style),
            Paragraph("Attendance Ratio:", meta_label_style), Paragraph(f"{presents_count} Present ({pct}%)", meta_val_style)
        ]
    ]
    
    meta_table = Table(meta_data, colWidths=[100, 160, 110, 160])
    meta_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    
    elements.append(meta_table)
    elements.append(Spacer(1, 20))
    
    # 4. Add Roster Table
    # Table headers
    roster_data = [[
        Paragraph("Roll Number", table_cell_bold),
        Paragraph("Student Name", table_cell_bold),
        Paragraph("Attendance Status", table_cell_bold),
        Paragraph("Checked-in Time", table_cell_bold)
    ]]
    
    # Table rows
    for r in records:
        status_lbl = "Present" if r["status"] == 'P' else "Absent"
        status_style = table_cell_present if r["status"] == 'P' else table_cell_absent
        time_lbl = r["time_marked"] if r["status"] == 'P' else "N/A"
        
        roster_data.append([
            Paragraph(r["roll_number"], table_cell_bold),
            Paragraph(r["name"], table_cell_style),
            Paragraph(status_lbl, status_style),
            Paragraph(time_lbl, table_cell_style)
        ])
        
    roster_table = Table(roster_data, colWidths=[110, 200, 110, 110])
    roster_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1F4E79')), # Dark blue header
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D3D3D3')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    
    # Clean header text color inside ReportLab paragraphs
    # Quick fix: style overrides paragraph color so we update the headers in code:
    header_bold_style = ParagraphStyle(
        'HeaderBold',
        parent=table_cell_bold,
        textColor=colors.white
    )
    for col in range(4):
        roster_data[0][col].style = header_bold_style
        
    elements.append(roster_table)
    elements.append(Spacer(1, 40))
    
    # 5. Add Signature Block (CR and Teacher)
    # We create a 2-column signature table with empty lines above labels
    sig_data = [
        ["", ""], # Spacer row
        ["____________________________________", "____________________________________"],
        ["Class Representative Signature", "Class Teacher / Lecturer Signature"],
        ["CR, " + sess["class_name"], "Department Faculty"]
    ]
    
    sig_table = Table(sig_data, colWidths=[260, 270])
    
    sig_label_style = ParagraphStyle(
        'SigLabel',
        fontName='Times-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#333333'),
        alignment=1 # Center
    )
    
    sig_sub_style = ParagraphStyle(
        'SigSub',
        fontName='Times-Italic',
        fontSize=8,
        leading=12,
        textColor=colors.HexColor('#666666'),
        alignment=1 # Center
    )
    
    sig_line_style = ParagraphStyle(
        'SigLine',
        fontName='Times-Roman',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#999999'),
        alignment=1 # Center
    )
    
    sig_table_data = [
        [
            Paragraph(sig_data[1][0], sig_line_style),
            Paragraph(sig_data[1][1], sig_line_style)
        ],
        [
            Paragraph(sig_data[2][0], sig_label_style),
            Paragraph(sig_data[2][1], sig_label_style)
        ],
        [
            Paragraph(sig_data[3][0], sig_sub_style),
            Paragraph(sig_data[3][1], sig_sub_style)
        ]
    ]
    
    sig_table = Table(sig_table_data, colWidths=[265, 265])
    sig_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    
    elements.append(sig_table)
    
    # Build Document
    doc.build(elements)
    
    return send_file(pdf_path, as_attachment=True)

@app.route('/export/excel/<int:class_id>')
def download_excel(class_id):
    """Serve the auto-generated Excel matrix sheet for a class"""
    cls = database.get_class_by_id(class_id)
    if not cls:
        return "Class not found", 404
        
    class_name = cls["name"]
    safe_name = "".join([c if c.isalnum() or c in " _-" else "_" for c in class_name])
    filepath = os.path.join(database.EXCEL_DIR, f"{safe_name}_attendance.xlsx")
    
    if not os.path.exists(filepath):
        # Trigger creation in case Excel doesn't exist yet
        database.sync_class_attendance_to_excel(class_id)
        
    if not os.path.exists(filepath):
        return "No student rosters found in this class to export.", 400
        
    return send_file(filepath, as_attachment=True)

# --- Camera & Offline Photo Training APIs ---

@app.route('/api/available_cameras')
def available_cameras():
    """Query and return active system camera indexes"""
    active_ports = []
    # Scan indices 0 to 4
    for index in range(5):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(index)
        if cap.isOpened():
            active_ports.append(index)
            cap.release()
    # Fallback to at least index 0 if none open
    if not active_ports:
        active_ports = [0]
    return jsonify(active_ports)

@app.route('/api/set_camera', methods=['POST'])
def set_camera():
    """Switch current camera stream to target index"""
    data = request.get_json() or {}
    index = data.get('camera_index')
    if index is None:
        return jsonify({"success": False, "message": "Missing camera index"}), 400
        
    try:
        index = int(index)
        success = camera_manager.set_camera_index(index)
        if success:
            return jsonify({"success": True, "message": f"Successfully switched camera to port {index}"})
        else:
            return jsonify({"success": False, "message": f"Failed to open camera port {index}"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/enroll/upload_photos', methods=['POST'])
@login_required
def upload_photos():
    """Train student profile using uploaded images (offline enrollment)"""
    name = request.form.get('name', '').strip()
    roll_number = request.form.get('roll', '').strip()
    class_id = request.form.get('classId')
    
    if not name or not roll_number or not class_id:
        return jsonify({"success": False, "message": "Missing name, roll number, or class ID"}), 400
        
    uploaded_files = request.files.getlist('photos')
    if not uploaded_files or len(uploaded_files) == 0 or uploaded_files[0].filename == '':
        return jsonify({"success": False, "message": "Please upload at least 1 image file."}), 400
        
    embeddings = []
    try:
        for file in uploaded_files:
            if file.filename == '':
                continue
            
            # Read file bytes and decode with OpenCV
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({"success": False, "message": f"Error decoding image file: {file.filename}"}), 400
                
            # Detect faces
            faces = face_engine.detect_faces(image)
            if len(faces) == 0:
                return jsonify({"success": False, "message": f"No face detected in '{file.filename}'. Make sure the student face is clearly visible."}), 400
            elif len(faces) > 1:
                return jsonify({"success": False, "message": f"Multiple faces detected in '{file.filename}'. Please only upload photos with 1 person."}), 400
                
            face_meta = faces[0]
            embedding = face_engine.extract_embedding(image, face_meta["landmarks"])
            
            if embedding is None:
                return jsonify({"success": False, "message": f"Failed to map face features in '{file.filename}'. Try a different angle or lighting."}), 400
                
            embeddings.append(embedding)
            
        if not embeddings:
            return jsonify({"success": False, "message": "No valid faces processed from the uploaded files."}), 400
            
        # Average and normalize embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
            
        # Write student to database
        success, msg = database.add_student(name, roll_number, class_id, avg_embedding)
        if success:
            return jsonify({"success": True, "message": f"Student '{name}' successfully registered via photo upload!"})
        else:
            return jsonify({"success": False, "message": msg})
            
    except Exception as e:
        return jsonify({"success": False, "message": f"Server training error: {str(e)}"}), 500

# Release camera on exit
import atexit
atexit.register(lambda: camera_manager.stop())

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
