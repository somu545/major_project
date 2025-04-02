import streamlit as st
import sqlite3
import hashlib
import re
from datetime import datetime

# --------------------------
# Authentication Functions
# --------------------------
def init_auth_db():
    conn = sqlite3.connect('auth.db')
    c = conn.cursor()
    
    # Create users table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  employee_id TEXT UNIQUE,
                  name TEXT,
                  email TEXT UNIQUE,
                  password TEXT,
                  department TEXT,
                  created_at TEXT)''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    return stored_password == hash_password(provided_password)

def validate_email(email):
    """Validate email format"""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email) is not None

def register_user(employee_id, name, email, password, department):
    """Register a new user"""
    try:
        conn = sqlite3.connect('auth.db')
        c = conn.cursor()
        
        # Check if employee_id or email already exists
        c.execute("SELECT * FROM users WHERE employee_id = ? OR email = ?", (employee_id, email))
        if c.fetchone():
            conn.close()
            return False, "Employee ID or Email already registered"
        
        # Insert new user
        hashed_password = hash_password(password)
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO users (employee_id, name, email, password, department, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                 (employee_id, name, email, hashed_password, department, created_at))
        
        conn.commit()
        conn.close()
        return True, "Registration successful"
    except Exception as e:
        return False, f"Registration error: {str(e)}"

def login_user(employee_id, password):
    """Authenticate a user"""
    try:
        conn = sqlite3.connect('auth.db')
        c = conn.cursor()
        
        c.execute("SELECT * FROM users WHERE employee_id = ?", (employee_id,))
        user = c.fetchone()
        conn.close()
        
        if user and verify_password(user[4], password):  # Index 4 is password
            return True, user
        else:
            return False, "Invalid employee ID or password"
    except Exception as e:
        return False, f"Login error: {str(e)}"

def get_user_profile(employee_id):
    """Get user profile data"""
    try:
        conn = sqlite3.connect('auth.db')
        c = conn.cursor()
        
        c.execute("SELECT id, employee_id, name, email, department, created_at FROM users WHERE employee_id = ?", (employee_id,))
        user = c.fetchone()
        conn.close()
        
        if user:
            return {
                "id": user[0],
                "employee_id": user[1],
                "name": user[2],
                "email": user[3],
                "department": user[4],
                "created_at": user[5]
            }
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching user profile: {str(e)}")
        return None

