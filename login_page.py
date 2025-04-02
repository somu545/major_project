import streamlit as st
from auth import login_user, init_auth_db
import requests
from streamlit_lottie import st_lottie

def load_lottie(url: str):
    """Load Lottie animation from URL"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def show_login_page():
    # Custom CSS with gradient container and top positioning
    st.markdown("""
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --text: #ffffff;  /* Changed to white for better contrast */
            --text-light: #e2e8f0;
            --border: black;
            --shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        }
        
        /* Gradient container positioned at top */
        .login-container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 2.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px 12px 16px 16px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            font-family: 'Inter', sans-serif;
            color: white;
        }
        
        /* Header with light text */
        .login-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .login-title {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.5rem;
        }
        
        .login-subtitle {
            font-size: 1rem;
            color: var(--text-light);
        }
        
        /* Input fields with transparent background */
        .stTextInput>div>div>input {
            padding: 12px 14px !important;
            border-radius: 8px !important;
            border: 1px solid var(--border) !important;
            font-size: 1rem !important;
            background-color: rgba(255,255,255,0.1) !important;
            color: white !important;
        }
        
        .stTextInput>div>div>input::placeholder {
            color: var(--text-light) !important;
            opacity: 0.7 !important;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: white !important;
            box-shadow: 0 0 0 3px rgba(255,255,255,0.2) !important;
        }
        
        /* Buttons with light style */
        .stButton>button {
    width: 100% !important;
    padding: 12px 16px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stButton>button:first-child {
    background-color: var(--primary) !important;
    color: white !important;
}

.stButton>button:first-child:hover {
    background-color: var(--primary-hover) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
}

.stButton>button:nth-child(2) {
    background-color: white !important;
    color: var(--primary) !important;
    border: 1px solid var(--border) !important;
}

.stButton>button:nth-child(2):hover {
    background-color: #f8fafc !important;
}
        
        /* Checkbox styling */
        .stCheckbox>label {
            color: var(--text-light) !important;
        }
        
        /* Animation container */
        .animation-container {
            margin: 1.5rem auto;
            max-width: 250px;
        }
        
        /* Responsive adjustments */
        @media (max-width: 600px) {
            .login-container {
                padding: 2rem 1.5rem;
                border-radius: 0;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # Load animation (optional)
    lottie_animation = load_lottie("https://assets5.lottiefiles.com/packages/lf20_ky24lkyz.json")
    
    # Login container positioned at top
    with st.container():
        st.markdown("""
        <div class="login-container">
            <div class="login-header">
                <div class="login-title">🔐 Employee Login</div>
                <div class="login-subtitle">Welcome to our Stress Detection System</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display animation if loaded
        if lottie_animation:
            st_lottie(lottie_animation, height=150, key="login_animation")
        
        # Form inputs with light placeholder text
        employee_id = st.text_input(
            "Employee ID", 
            key="login_employee_id",
            placeholder="Enter your employee ID"
        )
        
        password = st.text_input(
            "Password", 
            type="password", 
            key="login_password",
            placeholder="Enter your password"
        )
        
        # Remember me checkbox
        remember_me = st.checkbox("Remember me")
        
        # Buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            login_button = st.button("Sign In", key="login_btn")
        with col2:
            signup_button = st.button("Register", key="signup_btn")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Login logic (unchanged)
    if login_button:
        if not employee_id or not password:
            st.error("Please enter both employee ID and password")
        else:
            success, result = login_user(employee_id, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.employee_id = employee_id
                st.session_state.user_name = result[2]  # Name is at index 2
                st.success(f"Welcome back, {st.session_state.user_name}!")
                st.experimental_rerun()
            else:
                st.error(result)
    
    # Signup redirect logic
    if signup_button:
        st.session_state.show_signup = True
        st.experimental_rerun()