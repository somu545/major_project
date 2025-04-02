import streamlit as st
from auth import get_user_profile
from datetime import datetime
import sqlite3
import pandas as pd

def show_profile_page():
    st.title("👤 Employee Profile")
    
    if not st.session_state.get("logged_in", False):
        st.warning("Please log in to view your profile")
        return
    
    employee_id = st.session_state.get("employee_id")
    user_data = get_user_profile(employee_id)
    
    if not user_data:
        st.error("Could not retrieve user profile")
        return
    
    # Profile header
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://ui-avatars.com/api/?name=" + user_data["name"].replace(" ", "+") + 
                "&background=random&size=150", width=150)
    with col2:
        st.markdown(f"## {user_data['name']}")
        st.markdown(f"**Employee ID:** {user_data['employee_id']}")
        st.markdown(f"**Department:** {user_data['department']}")
    
    st.markdown("---")
    
    # Profile details
    st.subheader("Account Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Email Address**")
        st.markdown(f"<div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; color: #333;'>{user_data['email']}</div>", 
                   unsafe_allow_html=True)
    with col2:
        st.markdown("**Account Created**")
        created_date = datetime.strptime(user_data['created_at'], "%Y-%m-%d %H:%M:%S").strftime("%B %d, %Y")
        st.markdown(f"<div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; color: #333;'>{created_date}</div>", 
                   unsafe_allow_html=True)
    
    # Stress detection history
    st.markdown("---")
    st.subheader("Stress Detection History")
    
    try:
        conn = sqlite3.connect('stress_data.db')
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM stress_results WHERE employee_id=?", (employee_id,))
        session_count = c.fetchone()[0]
        
        if session_count > 0:
            c.execute("""
                SELECT timestamp, stress_level, emotion, duration 
                FROM stress_results 
                WHERE employee_id=? 
                ORDER BY timestamp DESC LIMIT 5
            """, (employee_id,))
            recent_sessions = c.fetchall()
            
            st.markdown(f"You have completed **{session_count}** stress detection sessions.")
            
            st.markdown("**Recent Sessions:**")
            for session in recent_sessions:
                session_time = datetime.strptime(session[0], "%Y-%m-%d %H:%M:%S.%f").strftime("%b %d, %Y at %I:%M %p")
                duration = f"{session[3]:.1f} seconds"
                st.markdown(f"""
                <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: #333;'>
                    <strong>{session_time}</strong><br/>
                    Stress Level: {session[1]} | Emotion: {session[2]} | Duration: {duration}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("You haven't completed any stress detection sessions yet.")
        
        conn.close()
    except Exception as e:
        st.error(f"Error retrieving session history: {str(e)}")
    
    # Logout button
    st.markdown("---")
    if st.button("🚪 Logout", key="logout_button", type="primary"):
        st.session_state.logged_in = False
        st.session_state.employee_id = None
        st.session_state.user_name = None
        st.success("You have been logged out successfully")
        st.experimental_rerun()