import streamlit as st
import cv2
import dlib
import imutils
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from keras.models import load_model
import os
import pandas as pd
import time
from datetime import datetime
import sqlite3
import plotly.express as px

from assistant import StressAssistant
import random

# Import authentication modules
from auth import init_auth_db
from login_page import show_login_page
from signup_page import show_signup_page
from profile_page import show_profile_page

# --------------------------
# UI Theme and Configuration
# --------------------------
def set_ui_theme():
    st.set_page_config(
        page_title="Employee Stress Detection",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
        /* Main sidebar */
        .sidebar .sidebar-content {
            background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
            color: white;
            padding-top: 2rem;
        }
        
        /* Video container */
        .video-container {
            border: 2px solid #4e4376;
            border-radius: 10px;
            padding: 10px;
            background: #f0f2f6;
            margin-bottom: 20px;
        }
        
        /* Button container */
        .button-container {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .button-container button {
            flex: 1;
        }
        
        /* Metric cards */
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-title {
            font-size: 1rem;
            color: #6c757d;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #4e4376;
        }
    </style>
    """, unsafe_allow_html=True)

# --------------------------
# Database Functions
# --------------------------
def init_db():
    try:
        conn = sqlite3.connect('stress_data.db')
        c = conn.cursor()
        
        # Check if table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stress_results'")
        table_exists = c.fetchone()
        
        if not table_exists:
            # Create new table with all columns
            c.execute('''CREATE TABLE stress_results
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          timestamp TEXT,
                          end_time TEXT,
                          duration REAL,
                          stress_value REAL,
                          stress_level TEXT,
                          emotion TEXT,
                          blink_count INTEGER,
                          ear REAL,
                          employee_id TEXT)''')
        else:
            # Check for missing columns and add them
            c.execute("PRAGMA table_info(stress_results)")
            columns = [column[1] for column in c.fetchall()]
            
            if 'end_time' not in columns:
                c.execute("ALTER TABLE stress_results ADD COLUMN end_time TEXT")
            if 'duration' not in columns:
                c.execute("ALTER TABLE stress_results ADD COLUMN duration REAL")
        
        conn.commit()
        conn.close()
        st.session_state.db_initialized = True
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")

def save_to_db(timestamp, end_time, duration, stress_value, stress_level, emotion, blink_count, ear, employee_id="default"):
    try:
        conn = sqlite3.connect('stress_data.db')
        c = conn.cursor()
        
        # First ensure all columns exist
        c.execute("PRAGMA table_info(stress_results)")
        columns = [column[1] for column in c.fetchall()]
        
        missing_columns = []
        if 'end_time' not in columns:
            missing_columns.append("end_time TEXT")
        if 'duration' not in columns:
            missing_columns.append("duration REAL")
        
        if missing_columns:
            for col in missing_columns:
                try:
                    c.execute(f"ALTER TABLE stress_results ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass  # Column might already exist
        
        # Now perform the insert
        c.execute("""INSERT INTO stress_results 
                     (timestamp, end_time, duration, stress_value, stress_level, emotion, blink_count, ear, employee_id) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (timestamp, end_time, duration, stress_value, stress_level, emotion, blink_count, ear, employee_id))
        
        conn.commit()
        conn.close()
        st.session_state.last_save_time = datetime.now()
    except Exception as e:
        st.error(f"Error saving to database: {str(e)}")

def get_all_results():
    try:
        conn = sqlite3.connect('stress_data.db')
        c = conn.cursor()
        c.execute("SELECT * FROM stress_results ORDER BY timestamp DESC")
        results = c.fetchall()
        conn.close()
        return results
    except Exception as e:
        st.error(f"Error fetching data from database: {str(e)}")
        return []

def get_results_by_employee(employee_id):
    try:
        conn = sqlite3.connect('stress_data.db')
        c = conn.cursor()
        c.execute("SELECT * FROM stress_results WHERE employee_id=? ORDER BY timestamp DESC", (employee_id,))
        results = c.fetchall()
        conn.close()
        return results
    except Exception as e:
        st.error(f"Error fetching employee data: {str(e)}")
        return []

# --------------------------
# Model Loading (Cached)
# --------------------------
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_models():
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        emotion_classifier = load_model("_mini_XCEPTION.102-0.66.hdf5", compile=False)
        return detector, predictor, emotion_classifier
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# --------------------------
# Stress Detection Functions
# --------------------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def eye_brow_distance(leye, reye):
    return dist.euclidean(leye, reye)

# In your emotion_finder function, update the preprocessing:
# Modify your emotion_finder function to:
def emotion_finder(faces, frame, emotion_classifier):
    EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
    x, y, w, h = face_utils.rect_to_bb(faces)
    
    # Expand ROI by 20% for better context
    padding = int(w * 0.2)
    x, y = max(0, x-padding), max(0, y-padding)
    w, h = min(frame.shape[1]-x, w+2*padding), min(frame.shape[0]-y, h+2*padding)
    
    roi = frame[y:y+h, x:x+w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_CUBIC)
    
    # Apply histogram equalization
    roi = cv2.equalizeHist(roi)
    
    roi = roi.astype("float") / 255.0
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)
    
    preds = emotion_classifier.predict(roi)[0]
    return EMOTIONS[preds.argmax()]

def normalize_values(points, disp):
    if len(points) < 2:
        return 0, "Low Stress"
    min_p, max_p = min(points), max(points)
    if max_p == min_p:
        return 0, "Low Stress"
    normalized = abs(disp - min_p) / abs(max_p - min_p)
    stress_value = np.exp(-normalized)
    return stress_value, "High Stress" if stress_value >= 0.75 else "Low Stress"

def text_background(img, text, pos, text_color=(255,255,255), bg_color=(0,0,0)):
    x, y = pos
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (x, y-h-5), (x+w+5, y+5), bg_color, -1)
    cv2.putText(img, text, (x+3, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

# --------------------------
# Detection Page
# --------------------------
def detect_stress():
    st.title("🧠 Real-time Stress Detection")
    
    # Initialize stress assistant
    if 'stress_assistant' not in st.session_state:
        st.session_state.stress_assistant = StressAssistant()
    
    # Check if user is logged in
    if not st.session_state.get("logged_in", False):
        st.warning("Please log in to use the stress detection feature")
        return
    
    # Initialize session variables if they don't exist
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'last_save_time' not in st.session_state:
        st.session_state.last_save_time = None
    
    # Create layout
    col_video, col_controls = st.columns([7, 3])
    
    with col_video:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        video_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_controls:
        st.info(f"Logged in as: {st.session_state.user_name} (ID: {st.session_state.employee_id})")
        
        # Buttons using HTML/CSS instead of nested columns
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        start_btn = st.button("🚀 Start", key="start")
        stop_btn = st.button("⏹️ Stop", key="stop")
        st.markdown('</div>', unsafe_allow_html=True)

        # Session timer display
        timer_placeholder = st.empty()
        timer_placeholder.markdown('<div class="metric-card"><div class="metric-title">⏱️ Session Duration</div><div class="metric-value">00:00:00</div></div>', 
                                 unsafe_allow_html=True)

        # Assistant advice placeholder
        advice_placeholder = st.empty()

    # Handle button actions
    if start_btn:
        st.session_state.running = True
        st.session_state.start_time = datetime.now()
        st.session_state.detection_data = {
            'stress_value': 0,
            'stress_level': "Low Stress",
            'emotion': "neutral",
            'blink_count': 0,
            'ear': 0
        }

    if stop_btn:
        if st.session_state.get('running', False):
            # Save results before stopping
            end_time = datetime.now()
            duration = (end_time - st.session_state.start_time).total_seconds()
            
            save_to_db(
                st.session_state.start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                end_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                duration,
                st.session_state.detection_data['stress_value'],
                st.session_state.detection_data['stress_level'],
                st.session_state.detection_data['emotion'],
                st.session_state.detection_data['blink_count'],
                st.session_state.detection_data['ear'],
                st.session_state.employee_id
            )
        
        st.session_state.running = False
        advice_placeholder.empty()

    if st.session_state.get('running', False):
        detector, predictor, emotion_classifier = load_models()
        if None in (detector, predictor, emotion_classifier):
            st.error("Models failed to load")
            st.session_state.running = False
        else:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Initialize variables
            ar_thresh = 0.3
            counter = total = 0
            points = []
            last_detection = None
            
            while st.session_state.get('running', False):
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)
                frame = imutils.resize(frame, width=800)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                clahe_image = clahe.apply(gray)
                
                # Face detection with stabilization
                detections = detector(clahe_image, 1)
                if not detections and last_detection:
                    detections = [last_detection]
                elif detections:
                    last_detection = max(detections, key=lambda r: r.width() * r.height())
                
                if detections:
                    try:
                        shape = predictor(gray, last_detection)
                        shape = face_utils.shape_to_np(shape)
                        
                        # Eye metrics
                        left_eye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
                        right_eye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]
                        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
                        
                        # Blink detection
                        if ear < ar_thresh:
                            counter += 1
                        elif counter >= 5:
                            total += 1
                            counter = 0
                        
                        # Emotion detection
                 
                        emotion = emotion_finder(last_detection, frame, emotion_classifier) 
                        emotion = emotion.lower().strip()
                        
                        # Stress detection
                        leyebrow = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"][1]]
                        reyebrow = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"][1]]
                        current_dist = eye_brow_distance(leyebrow[-1], reyebrow[0])
                        points.append(current_dist)
                        points = points[-50:]  # Keep last 50 points
                        
                        stress_value, stress_level = normalize_values(points, current_dist)
                        
                        # Update session state
                        st.session_state.detection_data = {
                            'stress_value': stress_value,
                            'stress_level': stress_level,
                            'emotion': emotion,
                            'blink_count': total,
                            'ear': ear
                        }

                        # Get advice from assistant (only if session has been running for at least 10 seconds)
                        if (datetime.now() - st.session_state.start_time).total_seconds() > 10:
                            blink_rate = total / ((datetime.now() - st.session_state.start_time).total_seconds() / 60)  # blinks per minute
                            advice = st.session_state.stress_assistant.analyze_stress(
                                stress_value,
                                emotion,
                                blink_rate
                            )
                            if advice:
                                st.session_state.stress_assistant.display_advice(advice, advice_placeholder)
                        
                        # Draw all annotations on video frame
                        x, y, w, h = face_utils.rect_to_bb(last_detection)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        
                        # Draw facial landmarks
                        for (x, y) in shape:
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                        
                        # Draw eye contours
                        leftEyeHull = cv2.convexHull(left_eye)
                        rightEyeHull = cv2.convexHull(right_eye)
                        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                        
                        # Draw eyebrow contours
                        leyebrowhull = cv2.convexHull(leyebrow)
                        reyebrowhull = cv2.convexHull(reyebrow)
                        cv2.drawContours(frame, [leyebrowhull], -1, (255, 0, 0), 1)
                        cv2.drawContours(frame, [reyebrowhull], -1, (255, 0, 0), 1)
                        
                        # Display metrics on video frame
                        text_background(frame, f"EAR: {ear:.2f}", (10, 30))
                        text_background(frame, f"Emotion: {emotion}", (10, 60))
                        text_background(frame, f"Stress: {stress_level}", (10, 90))
                        text_background(frame, f"Value: {int(stress_value*100)}%", (10, 120))
                        text_background(frame, f"Blinks: {total}", (10, 150))
                    
                    except Exception as e:
                        st.error(f"Detection error: {e}")
                
                # Update session timer
                elapsed_time = datetime.now() - st.session_state.start_time
                timer_placeholder.markdown(
                    f'<div class="metric-card"><div class="metric-title">⏱️ Session Duration</div>'
                    f'<div class="metric-value">{str(elapsed_time).split(".")[0]}</div></div>', 
                    unsafe_allow_html=True
                )
                
                video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                time.sleep(0.03)
            
            # Clean up when stopped
            cap.release()
            cv2.destroyAllWindows()
            advice_placeholder.empty()
# --------------------------
# Dashboard Page
# --------------------------
def dashboard():
    st.title("Stress Detection Dashboard")
    
    # Check if user is logged in
    if not st.session_state.get("logged_in", False):
        st.warning("Please log in to view the dashboard")
        return
    
    # Add a refresh button
    if st.button("🔄 Refresh Data"):
        st.experimental_rerun()
    
    # Check if database exists
    if not os.path.exists('stress_data.db'):
        st.warning("Database not found. Please run a detection session first.")
        return
    
    # Get results based on user role
    employee_id = st.session_state.employee_id
    results = get_results_by_employee(employee_id)
    
    if not results:
        st.warning("No data available. Run detection first.")
        return
    
    # Create DataFrame and process data
    df = pd.DataFrame(results, columns=["id", "timestamp", "end_time", "duration", 
                                      "stress_value", "stress_level", "emotion", 
                                      "blink_count", "ear", "employee_id"])
    
    # Convert datetime columns
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    
    # Normalize emotion strings (FIX FOR MISSING EMOTIONS)
    df['emotion'] = df['emotion'].str.lower().str.strip()
    
    # Map similar emotions to consistent labels
    emotion_map = {
        'happiness': 'happy',
        'anger': 'angry',
        'fear': 'scared',
        'surprise': 'surprised',
        'sadness': 'sad'
    }
    df['emotion'] = df['emotion'].map(emotion_map).fillna(df['emotion'])
    
    # Add date and time calculations
    df['date'] = df['timestamp'].dt.date
    df['time_mid'] = df['timestamp'] + (df['end_time'] - df['timestamp'])/2
    
    # Debug: Show unique emotions found
    st.write("Detected emotions:", df['emotion'].unique())
    
    # Filters
    with st.expander("🔍 Filter Data"):
        date_range = st.date_input("Date range", [df['timestamp'].dt.date.min(), df['timestamp'].dt.date.max()])
    
    if len(date_range) == 2:
        df = df[(df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])]
    
    # Session Metrics
    cols = st.columns(4)
    cols[0].metric("Total Sessions", len(df))
    cols[1].metric("Avg Duration", f"{df['duration'].mean():.1f} sec")
    cols[2].metric("Avg Stress", f"{df['stress_value'].mean()*100:.1f}%")
    
    # Get top emotion (capitalized for display)
    top_emotion = df['emotion'].mode()[0].capitalize() if not df.empty else "N/A"
    cols[3].metric("Top Emotion", top_emotion)
    
    # Main Visualization Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Timeline", "Stress Analysis", "Emotion Analysis", "Stress Assistant"])
    
    with tab1:
        st.subheader("Session Timeline")
        timeline_df = df.copy()
        timeline_df['duration_min'] = timeline_df['duration'] / 60
        timeline_df['stress_percent'] = timeline_df['stress_value'] * 100
        
        fig = px.scatter(timeline_df, 
                        x='time_mid',
                        size='duration_min',
                        color='stress_level',
                        hover_data=['duration', 'stress_percent', 'emotion', 'blink_count'],
                        labels={'time_mid': 'Session Time', 'duration_min': 'Duration (min)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Stress Level Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(df.set_index('time_mid')['stress_value'], height=300)
        with col2:
            st.bar_chart(df.groupby('stress_level').size(), height=300)
        st.metric("Average Blink Rate", 
                 f"{df['blink_count'].sum() / (df['duration'].sum()/60):.1f} blinks/min")
    
    with tab3:
        st.subheader("Emotion Analysis")
        
        # Ensure we have consistent emotion capitalization for display
        display_df = df.copy()
        display_df['emotion'] = display_df['emotion'].str.capitalize()
        
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(display_df['emotion'].value_counts(), height=300)
        with col2:
            emotion_time = display_df.groupby('emotion')['duration'].sum().sort_values(ascending=False)
            st.bar_chart(emotion_time, height=300)
        
        st.write("Emotion Distribution Over Time")
        emotion_over_time = display_df.groupby([pd.Grouper(key='time_mid', freq='D'), 'emotion']).size().unstack()
        st.area_chart(emotion_over_time.fillna(0))
    
    with tab4:
        st.markdown('<h3 class="custom-subtitle">🧠 Personalized Stress Assistant</h3>', unsafe_allow_html=True)
        
        if len(df) == 0:
            st.info("Complete at least one stress detection session to get personalized recommendations")
        else:
            # Overall stress assessment
            avg_stress = df['stress_value'].mean()
            if avg_stress > 0.7:
                stress_assessment = "consistently high"
                recommendation = "Consider regular stress management practices throughout your day"
                bg_color = "#ffebee"  # Light red
                text_color = "#b71c1c"  # Dark red
            elif avg_stress > 0.4:
                stress_assessment = "moderate"
                recommendation = "Targeted stress reduction during peak times would be beneficial"
                bg_color = "#fff8e1"  # Light amber
                text_color = "#ff8f00"  # Dark amber
            else:
                stress_assessment = "generally low"
                recommendation = "Maintain your current healthy habits"
                bg_color = "#e8f5e9"  # Light green
                text_color = "#2e7d32"  # Dark green
            
            st.markdown(f"""
            <div style="
                background-color: {bg_color};
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                border-left: 4px solid {text_color};
            ">
                <h4 style="color: {text_color}; margin-top: 0;">📊 Your Stress Assessment</h4>
                <p style="color: #333;">Your stress levels are <strong>{stress_assessment}</strong>. {recommendation}.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Time-based patterns
            st.markdown("### ⏰ Your Stress Patterns")
            
            # Time of day analysis
            df['hour'] = df['timestamp'].dt.hour
            hourly_stress = df.groupby('hour')['stress_value'].mean()
            
            if not hourly_stress.empty:
                peak_hour = hourly_stress.idxmax()
                st.markdown(f"""
                <div style="
                    background-color: #fff3e0;
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    border-left: 4px solid #fb8c00;
                ">
                    <h4 style="color: #e65100; margin-top: 0;">🕒 Peak Stress Time</h4>
                    <p style="color: #333;">Your stress tends to peak around <strong>{peak_hour}:00 - {peak_hour+1}:00</strong>.</p>
                    <p style="color: #333;">Consider scheduling a 5-minute break during this time for deep breathing or stretching.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Emotion-based recommendations
            dominant_emotion = df['emotion'].mode()[0]
            emotion_tips = {
                "angry": "Try the 5-4-3-2-1 grounding technique when you notice frustration building",
                "sad": "Connect with a colleague or friend when you're feeling down",
                "anxious": "Practice box breathing (4-4-4-4) when you feel anxious",
                "neutral": "Your neutral expressions suggest good emotional balance",
                "happy": "Your positive expressions are great - share that energy with others!"
            }
            
            emotion_colors = {
                "angry": ("#ffebee", "#c62828"),
                "sad": ("#e3f2fd", "#1565c0"),
                "anxious": ("#fff8e1", "#ff8f00"),
                "neutral": ("#f5f5f5", "#424242"),
                "happy": ("#e8f5e9", "#2e7d32")
            }
            
            bg_color, text_color = emotion_colors.get(dominant_emotion.lower(), ("#f5f5f5", "#333"))
            
            st.markdown(f"""
            <div style="
                background-color: {bg_color};
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                border-left: 4px solid {text_color};
            ">
                <h4 style="color: {text_color}; margin-top: 0;">😊 Your Dominant Emotion: {dominant_emotion.capitalize()}</h4>
                <p style="color: #333;">{emotion_tips.get(dominant_emotion.lower(), "Notice your emotional patterns throughout the day")}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Blink rate analysis
            blink_rate = df['blink_count'].sum() / (df['duration'].sum()/60)  # blinks per minute
            if blink_rate > 20:
                st.markdown("""
                <div style="
                    background-color: #e3f2fd;
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    border-left: 4px solid #1976d2;
                ">
                    <h4 style="color: #0d47a1; margin-top: 0;">👀 Eye Strain Alert</h4>
                    <p style="color: #333;">Your blink rate suggests possible eye strain. Try these techniques:</p>
                    <ul style="color: #333;">
                        <li>Follow the 20-20-20 rule (every 20 minutes, look 20 feet away for 20 seconds)</li>
                        <li>Adjust your screen brightness to match your environment</li>
                        <li>Practice conscious blinking for 30 seconds every hour</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Quick stress relief button - Added unique key here
            st.markdown("---")
            st.markdown("### 🚀 Quick Stress Relief")
            if st.button("Get Instant Stress Relief Tip", key="stress_relief_tip_btn"):
                advice = st.session_state.stress_assistant._general_stress_reduction_advice()
                st.markdown(f"""
                <div style="
                    background-color: #e8f5e9;
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    border-left: 4px solid #2e7d32;
                ">
                    <h4 style="color: #2e7d32; margin-top: 0;">{advice['title']}</h4>
                    <p style="color: #333;">{advice['message']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.subheader("Session Details")
    st.dataframe(df.sort_values('timestamp', ascending=False).drop(columns=['time_mid']))
# --------------------------
# Main App
# --------------------------
def main():
    set_ui_theme()
    init_db()
    init_auth_db()
    
    # Initialize session state variables
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    
    with st.sidebar:
        st.title("🧠 Employee Stress Detection")
        st.markdown("---")
        
        # Show different navigation based on login status
        if st.session_state.get('logged_in', False):
            st.success(f"Welcome, {st.session_state.user_name}!")
            page = st.radio("Navigation", ["Detect", "Dashboard", "Profile"], label_visibility="collapsed")
        else:
            if st.session_state.get('show_signup', False):
                page = "Signup"
            else:
                page = "Login"
                
        st.markdown("---")
        with st.expander("ℹ️ How to use"):
            st.write("1. Login or register\n2. Go to Detect page\n3. Start detection\n4. View dashboard for results")
    
    # Display the selected page
    if page == "Login":
        show_login_page()
    elif page == "Signup":
        show_signup_page()
    elif page == "Detect":
        detect_stress()
    elif page == "Dashboard":
        dashboard()
    elif page == "Profile":
        show_profile_page()

if __name__ == "__main__":
    main()

