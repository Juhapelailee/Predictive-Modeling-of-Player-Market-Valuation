import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Page configuration
st.set_page_config(page_title="Player Value Predictor", layout="wide")

st.markdown(
    """
    <style>
    /* Page background: subtle football pitch photo with lighter overlay */
    .stApp {
        background-image: linear-gradient(rgba(255,255,255,0.15), rgba(255,255,255,0.1)), 
                          url('https://images.unsplash.com/photo-1507925921958-8a45f6b60a3b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: #1a1a1a;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header style */
    .hero {
        background: linear-gradient(135deg, rgba(255,255,255,0.85), rgba(255,255,255,0.75));
        padding: 28px 32px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        gap: 20px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.15);
        margin-bottom: 32px;
        border: 1px solid rgba(255,255,255,0.8);
        backdrop-filter: blur(10px);
    }
    .hero h1 { 
        margin: 0; 
        font-size: 36px; 
        color: #1a1a1a; 
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
    }
    .subtitle { 
        color: #4a4a4a; 
        opacity: 0.95; 
        font-size: 16px; 
        margin-top: 8px;
        font-weight: 400;
    }

    /* Form container */
    .form-container {
        background: rgba(255,255,255,0.75);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
        margin-bottom: 24px;
    }
    
    /* Section headers */
    .stSubheader {
        color: #1a1a1a !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        margin-top: 24px !important;
        margin-bottom: 16px !important;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(0,0,0,0.15);
    }
    
    /* Result card with animation */
    .result-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.85));
        padding: 32px;
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        color: #1a1a1a;
        border: 2px solid rgba(255,255,255,0.9);
        backdrop-filter: blur(15px);
        animation: slideIn 0.5s ease-out;
        margin-top: 24px;
    }
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .result-value { 
        font-size: 48px; 
        font-weight: 800; 
        margin: 12px 0; 
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .result-sub { 
        color: #4a4a4a; 
        font-size: 16px; 
        margin-top: 8px;
    }

    /* Team badge */
    .team-badge { 
        width: 64px; 
        height: 64px; 
        border-radius: 50%; 
        display:inline-block; 
        border: 4px solid rgba(255,255,255,0.95);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        flex-shrink: 0;
    }
    
    /* Enhanced input styling */
    .stSelectbox > div > div {
        background-color: rgba(255,255,255,0.9);
        border-radius: 8px;
    }
    
    .stNumberInput > div > div > input {
        background-color: rgba(255,255,255,0.9);
        color: #1a1a1a;
        border-radius: 8px;
    }
    
    /* Slider styling */
    .stSlider {
        padding: 12px 0;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background-color: rgba(255,255,255,0.7);
        padding: 12px;
        border-radius: 8px;
        border: 1px solid rgba(0,0,0,0.1);
    }
    
    /* Submit button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 18px;
        padding: 16px 32px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        margin-top: 24px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(102, 126, 234, 0.6);
    }
    
    /* Label styling */
    label {
        color: #1a1a1a !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 8px 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero header
st.markdown(
    """
    <div class="hero">
      <div>
        <h1>⚽ Football Player Market Value Predictor</h1>
        <div class="subtitle">AI-powered valuation tool • Predict player market values using advanced machine learning</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
# Load resources
@st.cache_resource
def load_resources():
    model = joblib.load('model.pkl')
    df = pd.read_pickle('df.pkl')
    with open('team_target_encoding.json', 'r') as f:
        team_encoding = json.load(f)
    return model, df, team_encoding

model, df, team_encoding = load_resources()

# Small mapping of team -> brand color for nicer badges (extend as needed)
team_colors = {
    'Manchester City': '#6CABDD',
    'Manchester United': '#DA291C',
    'Liverpool': '#C8102E',
    'Real Madrid': '#FEBE10',
    'FC Barcelona': '#A50044',
    'Paris Saint-Germain': '#004170',
    'FC Bayern München': '#DC052D',
    'Chelsea': '#034694',
    'Arsenal': '#EF0107',
    'Juventus': '#000000',
}

def get_team_color(team_name: str) -> str:
    return team_colors.get(team_name, '#6c757d')


# Input form with enhanced layout
st.markdown('<div class="form-container">', unsafe_allow_html=True)

with st.form("player_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 Basic Information")
        position_category = st.selectbox(
            "Position Category", 
            ['Forward', 'Midfielder', 'Defender', 'Goalkeeper'], 
            index=1, 
            key='position_category',
            help="Select the player's primary position"
        )
        age = st.number_input(
            "Age", 
            min_value=16, 
            max_value=45, 
            value=30,
            help="Player's current age"
        )
        team_list = list(team_encoding.keys())
        default_team_index = team_list.index('Manchester City') if 'Manchester City' in team_list else 0
        team = st.selectbox(
            "Team", 
            team_list, 
            index=default_team_index,
            help="Current team of the player"
        )
        foot = st.radio(
            "Preferred Foot", 
            ["Right", "Left"], 
            index=0,
            horizontal=True,
            help="Player's dominant foot"
        )
        wage = st.slider(
            "Wage ($ per week)", 
            min_value=500, 
            max_value=440_000, 
            value=270_000,
            step=1000,
            format="$%d",
            help="Weekly wage in US dollars"
        )
        years_left = st.number_input(
            "Years Left on Contract", 
            min_value=0, 
            max_value=10, 
            value=3,
            help="Remaining years on current contract"
        )
        
    with col2:
        st.subheader("💪 Physical Attributes")
        height = st.number_input(
            "Height (cm)", 
            min_value=150, 
            max_value=220, 
            value=181,
            help="Player height in centimeters"
        )
        weight = st.number_input(
            "Weight (kg)", 
            min_value=50, 
            max_value=120, 
            value=70,
            help="Player weight in kilograms"
        )
        acceleration = st.slider(
            "Acceleration", 
            1, 100, 78,
            help="How quickly the player can reach top speed"
        )
        sprint_speed = st.slider(
            "Sprint Speed", 
            1, 100, 76,
            help="Maximum running speed"
        )
        
    with col3:
        st.subheader("⭐ Additional Info")
        international_reputation = st.slider(
            "International Reputation", 
            1, 5, 5,
            help="Player's reputation on international stage (1-5 stars)"
        )
        on_loan = st.radio(
            "On Loan", 
            ["No", "Yes"], 
            index=0,
            horizontal=True,
            help="Whether player is currently on loan"
        )
        agility = st.slider(
            "Agility", 
            1, 100, 82,
            help="Ability to change direction quickly"
        )
        balance = st.slider(
            "Balance", 
            1, 100, 80,
            help="Ability to maintain stability"
        )
        stamina = st.slider(
            "Stamina", 
            1, 100, 90,
            help="Endurance and ability to maintain performance"
        )
        strength = st.slider(
            "Strength", 
            1, 100, 74,
            help="Physical power and ability to hold off opponents"
        )
    
    # Position-specific skills with expander
    with st.expander("🎯 Position-specific Skills", expanded=True):
        skills = {}
        position_cols = st.columns(4)
        
        if position_category == 'Forward':
            with position_cols[0]:
                skills['crossing'] = st.slider('Crossing', 1, 100, 75, help="Ability to deliver accurate crosses")
                skills['finishing'] = st.slider('Finishing', 1, 100, 82, help="Accuracy in front of goal")
            with position_cols[1]:
                skills['dribbling'] = st.slider('Dribbling', 1, 100, 88, help="Ball control while running")
                skills['ball_control'] = st.slider('Ball Control', 1, 100, 85, help="First touch and close control")
            with position_cols[2]:
                skills['volleys'] = st.slider('Volleys', 1, 100, 78, help="Ability to strike the ball in the air")
        elif position_category == 'Midfielder':
            with position_cols[0]:
                skills['vision'] = st.slider('Vision', 1, 100, 94, help="Ability to see and create opportunities")
                skills['long_passing'] = st.slider('Long Passing', 1, 100, 93, help="Accuracy of long-range passes")
            with position_cols[1]:
                skills['short_passing'] = st.slider('Short Passing', 1, 100, 92, help="Accuracy of short passes")
                skills['composure'] = st.slider('Composure', 1, 100, 88, help="Calmness under pressure")
        elif position_category == 'Defender':
            with position_cols[0]:
                skills['defensive_awareness'] = st.slider('Defensive Awareness', 1, 100, 82, help="Positioning and reading of the game")
                skills['standing_tackle'] = st.slider('Standing Tackle', 1, 100, 85, help="Tackling ability while standing")
            with position_cols[1]:
                skills['interceptions'] = st.slider('Interceptions', 1, 100, 83, help="Ability to intercept passes")
                skills['aggression'] = st.slider('Aggression', 1, 100, 75, help="Aggressiveness in challenges")
        else:  # Goalkeeper
            with position_cols[0]:
                skills['gk_diving'] = st.slider('GK Diving', 1, 100, 80, help="Ability to dive and save shots")
                skills['gk_handling'] = st.slider('GK Handling', 1, 100, 78, help="Catching and holding the ball")
            with position_cols[1]:
                skills['gk_reflexes'] = st.slider('GK Reflexes', 1, 100, 85, help="Quick reaction saves")
    
    submitted = st.form_submit_button("🚀 Predict Market Value", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Prediction and results
if submitted:
    # Calculate position score
    position_score = np.mean(list(skills.values())) if skills else 0
    
    # Create feature array
    features = {
        'Age': age,
        'foot': 1 if foot == "Right" else 0,
        'Wage': np.log1p(wage),
        'Height_cm': height,
        'Weight_kg': weight,
        'Acceleration': acceleration,
        'Sprint speed': sprint_speed,
        'Agility': agility,
        'Balance': balance,
        'Stamina': stamina,
        'Strength': strength,
        'International reputation': international_reputation,
        'On Loan': 1 if on_loan == "Yes" else 0,
        'Team_encoded': team_encoding[team],
        'Years left': years_left,
        'Forward Score': position_score if position_category == 'Forward' else 0,
        'Midfielder Score': position_score if position_category == 'Midfielder' else 0,
        'Defender Score': position_score if position_category == 'Defender' else 0,
        'Goalkeeper Score': position_score if position_category == 'Goalkeeper' else 0,
        'Position Category_Defender': 1 if position_category == 'Defender' else 0,
        'Position Category_Forward': 1 if position_category == 'Forward' else 0,
        'Position Category_Goalkeeper': 1 if position_category == 'Goalkeeper' else 0,
        'Position Category_Midfielder': 1 if position_category == 'Midfielder' else 0,
    }

    # Convert to DataFrame and predict
    input_df = pd.DataFrame([features])
    log_prediction = model.predict(input_df)[0]
    prediction = np.exp(log_prediction)  # Apply exponential to reverse log transform

    # Display results in enhanced styled card
    color = get_team_color(team)
    value_usd = prediction
    value_m = value_usd / 1_000_000
    
    # Format value with appropriate scale
    if value_m >= 1000:
        value_display = f"${value_m/1000:.2f}B"
        value_label = f"${value_m/1000:.2f} Billion"
    elif value_m >= 1:
        value_display = f"${value_m:.2f}M"
        value_label = f"${value_m:.2f} Million"
    else:
        value_display = f"${value_usd:,.0f}"
        value_label = f"${value_usd:,.0f}"
    
    formatted = f"${value_usd:,.2f}"
    
    # Additional metrics
    col_result1, col_result2, col_result3 = st.columns(3)
    
    with col_result1:
        st.metric("Market Value (USD)", formatted)
    with col_result2:
        st.metric("Market Value (Millions)", f"${value_m:.2f}M")
    with col_result3:
        st.metric("Position Score", f"{position_score:.1f}/100")
    
    st.markdown(f"""
    <div class="result-card">
        <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap">
            <div class="team-badge" style="background:{color}"></div>
            <div style="flex:1;min-width:250px">
                <div class="result-sub" style="font-size:14px;text-transform:uppercase;letter-spacing:1px;opacity:0.8">Predicted Market Value</div>
                <div class="result-value">🏆 {formatted}</div>
                <div class="result-sub" style="margin-top:12px">
                    <span style="display:inline-block;margin-right:16px">📊 {value_label}</span>
                    <span style="display:inline-block;margin-right:16px">👤 {position_category}</span>
                    <span style="display:inline-block">🏟️ <strong>{team}</strong></span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add some visual feedback
    st.success("✅ Prediction completed successfully!")
    
