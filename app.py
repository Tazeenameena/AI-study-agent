import streamlit as st
import numpy as np
import random

# -----------------------------
# Q-LEARNING BACKEND
# -----------------------------

num_states = 12
num_actions = 4

Q = np.zeros((num_states, num_actions))

alpha = 0.1
gamma = 0.9
epsilon = 0.3

def get_state(energy, days_left, score):
    return energy * 4 + days_left * 2 + score

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        return np.argmax(Q[state])

def get_reward(action, energy, days_left):
    if action == 3:          # break
        return -3
    if energy == 0:          # tired but studying
        return -5
    if action == 2 and days_left == 1:  # revise near exam
        return 5
    return 10

# Train agent once
episodes = 500
for _ in range(episodes):
    energy = random.randint(0, 2)
    days_left = random.randint(0, 1)
    score = random.randint(0, 1)

    state = get_state(energy, days_left, score)
    action = choose_action(state)
    reward = get_reward(action, energy, days_left)

    next_state = get_state(max(0, energy - 1), days_left, score)

    Q[state, action] += alpha * (
        reward + gamma * np.max(Q[next_state]) - Q[state, action]
    )

def recommend_action(energy, days_left, score):
    state = get_state(energy, days_left, score)
    action = np.argmax(Q[state])
    actions = ["Study Math", "Study English", "Revise", "Take a Break"]
    return actions[action]

# -----------------------------
# STREAMLIT INTERFACE
# -----------------------------

st.set_page_config(page_title="Smart Study Planner", page_icon="📘")

st.title("🧠 Smart Study Planner AI")
st.write("An AI agent that learns what you should do next using reinforcement learning.")

st.divider()

energy = st.selectbox(
    "How is your energy level?",
    options=[0, 1, 2],
    format_func=lambda x: ["Low", "Medium", "High"][x]
)

days_left = st.selectbox(
    "How close is your exam?",
    options=[0, 1],
    format_func=lambda x: ["Far", "Near"][x]
)

score = st.selectbox(
    "How was your last test performance?",
    options=[0, 1],
    format_func=lambda x: ["Poor", "Good"][x]
)

if st.button("Get Study Recommendation"):
    result = recommend_action(energy, days_left, score)
    st.success(f"📌 Recommended Action: **{result}**")

st.divider()
st.caption("Powered by Q-Learning Reinforcement Learning")
