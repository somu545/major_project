import streamlit as st
from auth import register_user, validate_email
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

def show_signup_page():
    # Custom CSS matching login page style
    st.markdown("""
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --text: #ffffff;
            --text-light: #e2e8f0;
            --border: black;
            --shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            --error-bg: #fee2e2;
            --success-bg: #ecfdf5;
        }
        
        /* Animated background matching login */
        .animated-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            overflow: hidden;
        }
        
        .animated-bg::before {
            content: "";
            position: absolute;
            width: 200%;
            height: 200%;
            top: -50%;
            left: -50%;
            background: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxkZWZzPjxwYXR0ZXJuIGlkPSJwYXR0ZXJuIiB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHBhdHRlcm5Vbml0cz0idXNlclNwYWNlT25Vc2UiIHBhdHRlcm5UcmFuc2Zvcm09InJvdGF0ZSg0NSkiPjxyZWN0IHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgZmlsbD0icmdiYSgyNTUsMjU1LDI1NSwwLjA1KSIvPjwvcGF0dGVybj48L2RlZnM+PHJlY3QgZmlsbD0idXJsKCNwYXR0ZXJuKSIgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIvPjwvc3ZnPg==') repeat;
            animation: animateBg 20s linear infinite;
        }
        
        /* Floating signup container */
        .signup-container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 2.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border-radius: 12px 12px 16px 16px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            font-family: 'Inter', sans-serif;
            color: white;
            position: relative;
            top: -30px;
        }
        .stApp {
            padding-top: 0 !important;
        }
        
        .signup-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .signup-title {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.5rem;
        }
        
        .signup-subtitle {
            font-size: 1rem;
            color: var(--text-light);
        }
        
        /* Form elements */
        .stTextInput>div>div>input {
            padding: 12px 14px !important;
            border-radius: 8px !important;
            border: 1px solid var(--border) !important;
            font-size: 1rem !important;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
        }
        
        /* Buttons */
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
        
        /* Required field indicator */
        .required-field::after {
            content: " *";
            color: #ef4444;
        }
    </style>
    """, unsafe_allow_html=True)

    # Add animated background
    st.markdown('<div class="animated-bg"></div>', unsafe_allow_html=True)
    
    # Load animation (create account illustration)
    lottie_animation = load_lottie("https://assets2.lottiefiles.com/packages/lf20_ilujvy3q.json")
    
    # Floating signup container
    with st.container():
        st.markdown("""
        <div class="signup-container">
            <div class="signup-header">
                <div class="signup-title">👤 Create Account</div>
                <div class="signup-subtitle">Join our Stress Detection System</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display animation if loaded
        if lottie_animation:
            st_lottie(lottie_animation, height=150, key="signup_animation")
        
        # Form inputs with required indicators
        st.markdown('<label class="required-field">Employee ID</label>', unsafe_allow_html=True)
        employee_id = st.text_input(
            "", 
            key="signup_employee_id",
            placeholder="Enter your employee ID",
            label_visibility="collapsed"
        )
        
        st.markdown('<label class="required-field">Full Name</label>', unsafe_allow_html=True)
        name = st.text_input(
            "", 
            key="signup_name",
            placeholder="Enter your full name",
            label_visibility="collapsed"
        )
        
        st.markdown('<label class="required-field">Email Address</label>', unsafe_allow_html=True)
        email = st.text_input(
            "", 
            key="signup_email",
            placeholder="Enter your work email",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<label class="required-field">Password</label>', unsafe_allow_html=True)
            password = st.text_input(
                "", 
                type="password", 
                key="signup_password",
                placeholder="Create password",
                label_visibility="collapsed"
            )
        with col2:
            st.markdown('<label class="required-field">Confirm Password</label>', unsafe_allow_html=True)
            confirm_password = st.text_input(
                "", 
                type="password", 
                key="signup_confirm",
                placeholder="Confirm password",
                label_visibility="collapsed"
            )
        
        department = st.selectbox(
            "Department", 
            ["HR", "IT", "Finance", "Marketing", "Operations", "Sales", "Other"]
        )
        
        # Buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            register_button = st.button("Create Account", key="register_btn")
        with col2:
            login_button = st.button("Back to Login", key="login_btn")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Registration logic
    if register_button:
        if not employee_id or not name or not email or not password:
            st.error("Please fill in all required fields")
        elif password != confirm_password:
            st.error("Passwords do not match")
        elif len(password) < 6:
            st.error("Password must be at least 6 characters long")
        elif not validate_email(email):
            st.error("Please enter a valid email address")
        else:
            success, message = register_user(employee_id, name, email, password, department)
            if success:
                st.success(message)
                st.info("You can now log in with your credentials")
                st.session_state.show_signup = False
                st.experimental_rerun()
            else:
                st.error(message)
    
    if login_button:
        st.session_state.show_signup = False
        st.experimental_rerun()