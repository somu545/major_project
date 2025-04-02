import streamlit as st
import random
from datetime import datetime

class StressAssistant:
    def __init__(self):
        self.stress_history = []
        self.last_advice_time = None
        self.advice_cooldown = 300  # 5 minutes in seconds
        
    def analyze_stress(self, stress_level, emotion, blink_rate):
        """Analyze stress data and provide recommendations"""
        timestamp = datetime.now()
        self.stress_history.append({
            "timestamp": timestamp,
            "stress_level": stress_level,
            "emotion": emotion,
            "blink_rate": blink_rate
        })
        
        # Keep only last 10 readings
        if len(self.stress_history) > 10:
            self.stress_history = self.stress_history[-10:]
            
        return self._generate_advice(stress_level, emotion, blink_rate)
    
    def _generate_advice(self, stress_level, emotion, blink_rate):
        """Generate personalized stress reduction advice"""
        current_time = datetime.now()
        
        # Check if we should give advice (not too frequently)
        if (self.last_advice_time and 
            (current_time - self.last_advice_time).total_seconds() < self.advice_cooldown):
            return None
        
        # Only provide advice for high stress
        if stress_level < 0.7:
            return None
        
        self.last_advice_time = current_time
        
        # Base recommendations on stress level and emotion
        emotion_lower = emotion.lower()
        if emotion_lower in ["angry", "frustrated"]:
            return self._anger_management_advice()
        elif emotion_lower in ["anxious", "scared", "nervous"]:
            return self._anxiety_reduction_advice()
        elif emotion_lower in ["sad", "depressed"]:
            return self._mood_improvement_advice()
        elif blink_rate > 25:  # High blink rate indicates possible eye strain
            return self._eye_strain_advice()
        else:
            return self._general_stress_reduction_advice()
    
    def _anger_management_advice(self):
        techniques = [
            "Take 5 deep breaths - inhale for 4 seconds, hold for 4, exhale for 6",
            "Try the 5-4-3-2-1 grounding technique: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste",
            "Step away for a 5-minute walk to cool down",
            "Write down what's bothering you, then tear it up as symbolic release",
            "Squeeze a stress ball or press your palms together firmly for 10 seconds"
        ]
        return {
            "title": "😤 Anger Management Tip",
            "message": random.choice(techniques),
            "priority": "high"
        }
    
    def _anxiety_reduction_advice(self):
        techniques = [
            "Practice box breathing: Inhale 4s, hold 4s, exhale 4s, hold 4s - repeat 5 times",
            "Try progressive muscle relaxation: Tense and release each muscle group from toes to head",
            "Write down your worries and categorize them: 'Can control' vs 'Can't control'",
            "Listen to calming music or nature sounds for 3 minutes",
            "Name 3 things in your environment that are safe and stable"
        ]
        return {
            "title": "😰 Anxiety Reduction Tip",
            "message": random.choice(techniques),
            "priority": "high"
        }
    
    def _mood_improvement_advice(self):
        techniques = [
            "Think of 3 things you're grateful for today",
            "Call or message someone who makes you smile",
            "Look at photos of happy memories for 2 minutes",
            "Do 2 minutes of gentle stretching or yoga poses",
            "Write down one small thing you can do today that would make tomorrow better"
        ]
        return {
            "title": "😔 Mood Improvement Tip",
            "message": random.choice(techniques),
            "priority": "medium"
        }
    
    def _eye_strain_advice(self):
        techniques = [
            "Follow the 20-20-20 rule: Every 20 minutes, look at something 20 feet away for 20 seconds",
            "Close your eyes and place warm palms over them for 30 seconds",
            "Massage your temples and eyebrow area gently for 1 minute",
            "Blink rapidly 10 times, then close eyes for 20 seconds",
            "Adjust your screen brightness to match your environment"
        ]
        return {
            "title": "👀 Eye Strain Relief",
            "message": random.choice(techniques),
            "priority": "medium"
        }
    
    def _general_stress_reduction_advice(self):
        techniques = [
            "Take a 3-minute break to stand up and stretch",
            "Drink a glass of water - dehydration increases stress",
            "Hum or sing your favorite song for 1 minute",
            "Do 10 slow neck rolls in each direction",
            "List 3 things you've accomplished today, no matter how small"
        ]
        return {
            "title": "🧘 Stress Relief Tip",
            "message": random.choice(techniques),
            "priority": "low"
        }
    
    def display_advice(self, advice, placeholder=None):
        """Display advice in a nicely formatted way"""
        if not advice:
            return
        
        priority_colors = {
            "high": "#ff4b4b",
            "medium": "#ffa700",
            "low": "#0068c9"
        }
        
        color = priority_colors[advice['priority']]
        
        html = f"""
        <div style="
            background-color: {color}10;
            border-left: 4px solid {color};
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        ">
            <h4 style="margin-top: 0; color: {color}">{advice['title']}</h4>
            <p>{advice['message']}</p>
        </div>
        """
        
        if placeholder:
            placeholder.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown(html, unsafe_allow_html=True)