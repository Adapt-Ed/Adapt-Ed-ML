import os
import json
import time
import hashlib
import pandas as pd
import streamlit as st
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# Set page configuration
st.set_page_config(
    page_title="Quiz Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for beautiful UI
st.markdown(
    """
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 30px;
        font-weight: bold;
        color: #1E88E5;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .quiz-header {
        font-size: 24px;
        font-weight: bold;
        color: #1E88E5;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .question-text {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .option-text {
        font-size: 16px;
        margin-bottom: 5px;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: #000;
        margin-bottom: 20px;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: #e3f2fd;
        margin-bottom: 20px;
    }
    .correct-answer {
        background-color: #c8e6c9;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        color: #000;
    }
    .wrong-answer {
        background-color: #ffcdd2;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        color: #000;
    }
    .login-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        background-color: white;
    }
    .quiz-settings {
        background-color: #f1f8fe;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
</style>
""",
    unsafe_allow_html=True,
)


# Quiz model definition
class QuizOutput(BaseModel):
    quiz: List[Dict] = Field(
        description="List of dictionaries containing 5 keys, 'questions', 'options', 'correct option', 'sub_topic' and 'difficulty_level'."
    )


# Quiz generation function
def generate_quiz(no_questions, topic, subtopics):
    parser = JsonOutputParser(pydantic_object=QuizOutput)

    PROMPT = """You are a quiz generator. Your job is to create {no_questions} quiz question based on the provided topic, subtopics, and the user's proficiency level.

    Inputs:
    - Topic: {topic} (The main theme of the quiz)
    - SubTopics: {subtopics} (Relevant subtopics)

    Instructions:
    1. Ensure all the generated questions are form the subtopics only.
    2. Make sure you cover all the subtopics in the quiz.
    3. Quiz should be divided into 3 levels of difficulty: Easy, Medium, and Hard.

    Output Format (JSON):
    {{
      "quiz": [
        {{
          "question": <question_text>,
          "options": ["A. <option1>", "B. <option2>", "C. <option3>", "D. <option4>"],
          "correct_option": <one_of_A_B_C_or_D>,
          "sub_topic": <subtopic>,
          "difficulty_level": <difficulty_level>
        }},
        ...
      ],
    }}

    - Provide only the answer in the specified format with no extra explanations.
    - Strictly generate the {no_questions} questions only.
    """

    llm = ChatOpenAI(
        model=st.secrets["OPENAI"]["model"],
        base_url=st.secrets["OPENAI"]["base_url"],
        api_key=st.secrets["OPENAI"]["api_key"],
    )

    prompt = PromptTemplate(
        template=PROMPT,
        input_variables=["no_questions", "topic", "subtopics"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    try:
        with st.spinner("Generating quiz questions..."):
            result = chain.invoke(
                {"no_questions": no_questions, "topic": topic, "subtopics": subtopics}
            )
        return result
    except Exception as e:
        st.error(f"Error generating quiz please try again...")
        return None


# User authentication functions
def get_users_db():
    if not os.path.exists("users.json"):
        with open("users.json", "w") as f:
            json.dump({}, f)

    with open("users.json", "r") as f:
        return json.load(f)


def save_users_db(users_db):
    with open("users.json", "w") as f:
        json.dump(users_db, f)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username, password):
    users_db = get_users_db()
    if username in users_db:
        return False

    users_db[username] = {"password_hash": hash_password(password), "quiz_history": []}
    save_users_db(users_db)
    return True


def authenticate_user(username, password):
    users_db = get_users_db()
    if username not in users_db:
        return False

    if users_db[username]["password_hash"] == hash_password(password):
        return True

    return False


def save_quiz_result(username, quiz_data, score, total):
    users_db = get_users_db()
    if username in users_db:
        quiz_history = users_db[username].get("quiz_history", [])
        quiz_history.append(
            {
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "topic": quiz_data["topic"],
                "subtopics": quiz_data["subtopics"],
                "no_questions": quiz_data["no_questions"],
                "score": score,
                "total": total,
                "percentage": round((score / total) * 100, 2),
            }
        )
        users_db[username]["quiz_history"] = quiz_history
        save_users_db(users_db)


# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "quiz_generated" not in st.session_state:
    st.session_state.quiz_generated = False
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "current_stage" not in st.session_state:
    st.session_state.current_stage = "login"

# Sidebar for navigation and user info
with st.sidebar:
    st.image(
        "https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png", width=200
    )
    st.title("Quiz Generator")

    if st.session_state.authenticated:
        st.success(f"Logged in as {st.session_state.username}")

        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.quiz_generated = False
            st.session_state.quiz_data = None
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.quiz_score = 0
            st.session_state.current_stage = "login"
            st.rerun()

        st.write("---")

        if st.session_state.current_stage != "quiz_settings":
            if st.button("Create New Quiz", key="sidebar_create_quiz"):
                st.session_state.current_stage = "quiz_settings"
                st.session_state.quiz_generated = False
                st.session_state.quiz_data = None
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()

        if st.session_state.current_stage != "history":
            if st.button("View Quiz History", key="sidebar_view_history"):
                st.session_state.current_stage = "history"
                st.rerun()

# Main application flow
if not st.session_state.authenticated:
    st.markdown(
        '<div class="main-header">Welcome to Quiz Generator</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login_form"):
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            submit_button = st.form_submit_button("Login")

            if submit_button:
                if authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.current_stage = "quiz_settings"
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with tab2:
        with st.form("register_form"):
            st.subheader("Register")
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")

            submit_button = st.form_submit_button("Register")

            if submit_button:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif not new_username or not new_password:
                    st.error("Username and password cannot be empty")
                else:
                    if create_user(new_username, new_password):
                        st.success("Registration successful! You can now login.")
                    else:
                        st.error("Username already exists")

elif st.session_state.current_stage == "quiz_settings":
    st.markdown('<div class="main-header">Quiz Generator</div>', unsafe_allow_html=True)

    with st.form("quiz_settings_form"):
        st.markdown(
            '<div class="sub-header">Quiz Settings</div>', unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            topic = st.text_input("Topic", "Data Structures and Algorithms")
            no_questions = st.slider(
                "Number of Questions", min_value=5, max_value=30, value=10, step=5
            )

        with col2:
            subtopics_input = st.text_area(
                "Subtopics (one per line)", "Arrays\nLinked Lists\nSorting Algorithms"
            )

        submit_button = st.form_submit_button("Generate Quiz")

        if submit_button:
            if not subtopics_input or not topic:
                st.error("Please provide a topic and subtopics")
            else:
                subtopics = [
                    s.strip() for s in subtopics_input.split("\n") if s.strip()
                ]

                quiz_result = generate_quiz(no_questions, topic, subtopics)

                if quiz_result:
                    st.session_state.quiz_generated = True
                    st.session_state.quiz_data = {
                        "quiz": quiz_result["quiz"],
                        "topic": topic,
                        "subtopics": subtopics,
                        "no_questions": no_questions,
                    }
                    st.session_state.current_stage = "take_quiz"
                    st.success("Quiz generated successfully!")
                    st.rerun()

# Replace the quiz form section in the take_quiz stage with this code:

elif st.session_state.current_stage == "take_quiz":
    quiz_data = st.session_state.quiz_data

    st.markdown(
        f'<div class="main-header">{quiz_data["topic"]} Quiz</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sub-header">Subtopics: {", ".join(quiz_data["subtopics"])}</div>',
        unsafe_allow_html=True,
    )

    # Initialize answers dictionary if not already done
    if "answers_initialized" not in st.session_state:
        for i in range(len(quiz_data["quiz"])):
            st.session_state.quiz_answers[i] = ""
        st.session_state.answers_initialized = True

    with st.form("quiz_form"):
        for i, question in enumerate(quiz_data["quiz"]):
            st.markdown(
                f"""
            <div class="card">
                <div class="question-text">Q{i+1}. {question["question"]}</div>
                <div class="option-text">{question["sub_topic"]} | {question["difficulty_level"]}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Add a "Select an answer" option as the first choice
            options = [opt.strip() for opt in question["options"]]

            # Add an instruction as first option that indicates no selection
            display_options = ["Select an answer"] + options
            selected_index = st.radio(
                f"Select your answer for question {i+1}:",
                options=range(len(display_options)),
                format_func=lambda x: display_options[x],
                key=f"q_{i}",
            )

            # Only update the answer if the user selected an actual option (not "Select an answer")
            if selected_index > 0:
                st.session_state.quiz_answers[i] = options[selected_index - 1]
            else:
                st.session_state.quiz_answers[i] = ""

            st.write("---")

        # Validate that all questions have been answered
        submit_quiz = st.form_submit_button("Submit Quiz")

        if submit_quiz:
            # Check if all questions are answered
            unanswered = [
                i + 1
                for i in range(len(quiz_data["quiz"]))
                if not st.session_state.quiz_answers.get(i)
            ]

            if unanswered:
                st.error(
                    f"Please answer all questions before submitting. Missing answers for questions: {', '.join(map(str, unanswered))}"
                )
            else:
                score = 0
                for i, question in enumerate(quiz_data["quiz"]):
                    correct_option = question["correct_option"]
                    if st.session_state.quiz_answers[i].startswith(correct_option):
                        score += 1

                st.session_state.quiz_submitted = True
                st.session_state.quiz_score = score
                save_quiz_result(
                    st.session_state.username, quiz_data, score, len(quiz_data["quiz"])
                )
                st.session_state.current_stage = "results"
                st.rerun()

elif st.session_state.current_stage == "results":
    quiz_data = st.session_state.quiz_data
    score = st.session_state.quiz_score
    total = len(quiz_data["quiz"])
    percentage = round((score / total) * 100, 2)

    st.markdown('<div class="main-header">Quiz Results</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", f"{score}/{total}")
    with col2:
        st.metric("Percentage", f"{percentage}%")
    with col3:
        if percentage >= 80:
            st.metric("Grade", "Excellent")
        elif percentage >= 60:
            st.metric("Grade", "Good")
        elif percentage >= 40:
            st.metric("Grade", "Average")
        else:
            st.metric("Grade", "Needs Improvement")

    st.markdown(
        '<div class="sub-header">Detailed Results</div>', unsafe_allow_html=True
    )

    for i, question in enumerate(quiz_data["quiz"]):
        user_answer = st.session_state.quiz_answers[i]
        correct_option = question["correct_option"]
        is_correct = user_answer.startswith(correct_option)

        with st.expander(
            f"Question {i+1}: {question['question']} ({question['sub_topic']} | {question['difficulty_level']})"
        ):
            st.write("**Options:**")
            for option in question["options"]:
                st.write(f"- {option}")

            st.write("**Your Answer:**")
            if is_correct:
                st.markdown(
                    f'<div class="correct-answer">{user_answer} ✓</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="wrong-answer">{user_answer} ✗</div>',
                    unsafe_allow_html=True,
                )
                st.write("**Correct Answer:**")
                for option in question["options"]:
                    if option.startswith(correct_option):
                        st.markdown(
                            f'<div class="correct-answer">{option}</div>',
                            unsafe_allow_html=True,
                        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Take Another Quiz", key="results_take_another"):
            st.session_state.current_stage = "quiz_settings"
            st.session_state.quiz_generated = False
            st.session_state.quiz_data = None
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.rerun()

    with col2:
        if st.button("View Quiz History", key="results_view_history"):
            st.session_state.current_stage = "history"
            st.rerun()

elif st.session_state.current_stage == "history":
    st.markdown('<div class="main-header">Quiz History</div>', unsafe_allow_html=True)

    users_db = get_users_db()
    if st.session_state.username in users_db:
        quiz_history = users_db[st.session_state.username].get("quiz_history", [])

        if not quiz_history:
            st.info("You haven't taken any quizzes yet!")
        else:
            df = pd.DataFrame(quiz_history)

            # Performance metrics
            st.markdown(
                '<div class="sub-header">Performance Metrics</div>',
                unsafe_allow_html=True,
            )
            col1, col2, col3 = st.columns(3)

            with col1:
                avg_score = df["percentage"].mean()
                st.metric("Average Score", f"{avg_score:.2f}%")

            with col2:
                total_quizzes = len(df)
                st.metric("Total Quizzes", total_quizzes)

            with col3:
                if not df.empty:
                    best_score = df["percentage"].max()
                    st.metric("Best Score", f"{best_score:.2f}%")

            # History table
            st.markdown(
                '<div class="sub-header">Quiz History</div>', unsafe_allow_html=True
            )

            # Format the DataFrame for display
            df_display = df.copy()
            df_display["topic_subtopics"] = df_display.apply(
                lambda row: f"{row['topic']} ({', '.join(row['subtopics'][:2]) + ('...' if len(row['subtopics']) > 2 else '')})",
                axis=1,
            )
            df_display["score_display"] = df_display.apply(
                lambda row: f"{row['score']}/{row['total']} ({row['percentage']}%)",
                axis=1,
            )

            st.dataframe(
                df_display[
                    ["date", "topic_subtopics", "no_questions", "score_display"]
                ].rename(
                    columns={
                        "date": "Date",
                        "topic_subtopics": "Topic & Subtopics",
                        "no_questions": "Questions",
                        "score_display": "Score",
                    }
                ),
                use_container_width=True,
            )

            # Performance over time
            st.markdown(
                '<div class="sub-header">Performance Over Time</div>',
                unsafe_allow_html=True,
            )

            # Convert date to datetime
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            chart_data = df[["date", "percentage"]]
            st.line_chart(chart_data.set_index("date"))

    if st.button("Take Another Quiz", key="history_take_another"):
        st.session_state.current_stage = "quiz_settings"
        st.session_state.quiz_generated = False
        st.session_state.quiz_data = None
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False
        st.rerun()
