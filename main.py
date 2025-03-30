import uuid
import logging
from typing import List, Dict, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- LangChain / LLM Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Setup basic logging configuration.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Pydantic Model for LLM Output ---
class QuizOutput(BaseModel):
    quiz: List[Dict] = Field(
        description="List of dictionaries containing a question, its options and the correct option."
    )
    user_level: Optional[str] = Field(
        description="User proficiency level based on the answers provided."
    )


# --- Pydantic Models for Requests & Responses ---
class StartQuizRequest(BaseModel):
    user_id: str  # Unique identifier for the user
    topic: str
    proficiency: str


class QuizResponse(BaseModel):
    session_id: str
    user_id: str
    quiz: List[Dict]
    current_level: str


class UserAnswer(BaseModel):
    user_selected_option: str  # Only need the option the user selected


class SubmitAnswerRequest(BaseModel):
    session_id: str
    user_id: str  # Include user_id in submissions for validation
    answers: List[UserAnswer]


class FinalQuizResponse(BaseModel):
    session_id: str
    user_id: str
    final_proficiency: str
    history: List
    performance_summary: Dict
    completed_at: datetime


# --- User Progress Tracking ---
class UserProgressRecord(BaseModel):
    user_id: str
    topic: str
    session_id: str
    start_level: str
    final_level: str
    accuracy: float
    questions_answered: int
    completed_at: datetime


# --- Setup the chain (replace API_KEY with your key or env variable) ---
parser = JsonOutputParser(pydantic_object=QuizOutput)
PROMPT = """
You are a quiz generator, your task is to generate 1 question of quiz based on the topic and user's level.

Topic: {topic} [Topic of the quiz.]

User Proficiency: {level} [User's current proficiency level.]

Previous Questions and Answers of the user: {history}

Your task is to predict the user's proficiency based on the history if history is non-empty.
If the history is empty, return user's proficiency as None.
When history is empty, you are generating the first question.

Each history item contains:
- The question that was asked
- The user's selected answer
- The correct answer
- Level of the question
- Whether the user was correct

Example History Item:
[
  {{
    "question": "What is the capital of France?",
    "user_selected_option": "A. Paris",
    "correct_option": "A",
    "is_correct": true
  }}
]

Example Output:
{{
"quiz":[
    {{
        "question": "What is ...?",
        "options": ["A. Option1", "B. Option2", "C. Option3", "D. Option4"],
        "correct_option": "A"
    }}
],
"user_level": null if history is empty, else your analysis based on user's answers
}}

- Only provide the answer without extra information or justifications.
- Always generate questions based on the user's proficiency.
- We have 4 types of proficiency on our platform [Novice, Beginner, Intermediate and Expert]. 
- When predicting user level, consider these guidelines:
  * See the history if the user is not able to give the right answer of the current level then lower one level of the user and question's difficulty
  * See the history if the user is able to give the right answer of the level then increase one levle of the user and question's difficulty
- Do not generate duplicate or same questions based on the history.
- Generate fresh and new questions based on the topic and difficulty.
"""

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key="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",  # Replace with your API key
)

prompt_template = PromptTemplate(
    template=PROMPT,
    input_variables=["topic", "level", "history"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Chain the prompt, LLM, and parser together.
chain = prompt_template | llm | parser


# --- Quiz Session Class ---
class QuizSession:
    def __init__(self, session_id: str, user_id: str, topic: str, initial_level: str):
        self.session_id = session_id
        self.user_id = user_id
        self.topic = topic
        self.current_level = initial_level
        self.initial_level = initial_level  # Store initial level for comparison
        self.history: List[Dict] = []  # Stores complete question & answer history
        self.total_questions = 0
        self.correct_answers = 0
        self.current_question = None  # Store the current question for later reference
        self.started_at = datetime.now()

    def generate_quiz(self) -> Dict:
        """Generate 1 quiz question using the LLM chain."""
        # Log the current history for debugging.
        logger.info(
            f"User {self.user_id}, Session {self.session_id} - Generating quiz with history: {self.history}"
        )
        output = chain.invoke(
            {
                "topic": self.topic,
                "level": self.current_level,
                "history": self.history,
            }
        )

        # Update current proficiency if provided by the LLM.
        if (
            output.get("user_level")
            and output.get("user_level") != "null"
            and output.get("user_level") is not None
        ):
            self.current_level = output["user_level"]
            logger.info(
                f"User {self.user_id}, Session {self.session_id} - Updated level to: {self.current_level}"
            )

        # Store the current question for later reference when processing answers
        if output.get("quiz") and len(output["quiz"]) > 0:
            self.current_question = output["quiz"][0]

        # Track the total number of questions generated.
        num_questions = len(output.get("quiz", []))
        self.total_questions += num_questions
        return output

    def add_history(self, user_answers: List[UserAnswer]):
        """Append the user's answers with question context to the session history."""
        # Make sure we have a current question
        if not self.current_question:
            logger.error(f"No current question found for session {self.session_id}")
            return

        # Process each answer
        for answer in user_answers:
            # Get the correct option from the stored question
            correct_option = self.current_question.get("correct_option")

            # Create a complete history item
            history_item = {
                "question": self.current_question.get("question"),
                "options": self.current_question.get("options"),
                "user_selected_option": answer.user_selected_option,
                "correct_option": correct_option,
                "level_of_question": self.current_level,
                "is_correct": answer.user_selected_option == correct_option,
            }

            # Add to history
            self.history.append(history_item)

            # Track correct answers
            if history_item["is_correct"]:
                self.correct_answers += 1

        # Log the updated history after adding new answers
        logger.info(
            f"User {self.user_id}, Session {self.session_id} - Updated history: {self.history}"
        )
        logger.info(
            f"User {self.user_id}, Session {self.session_id} - Performance: {self.correct_answers}/{self.total_questions}"
        )

    def get_performance_summary(self) -> Dict:
        """Generate a summary of the user's performance."""
        accuracy = 0
        if self.total_questions > 0:
            accuracy = (self.correct_answers / self.total_questions) * 100

        return {
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "accuracy": round(accuracy, 2),
            "final_level": self.current_level,
            "level_progression": f"{self.initial_level} → {self.current_level}",
            "session_duration_minutes": round(
                (datetime.now() - self.started_at).total_seconds() / 60, 2
            ),
        }

    def create_progress_record(self) -> UserProgressRecord:
        """Create a user progress record for database storage."""
        accuracy = 0
        if self.total_questions > 0:
            accuracy = (self.correct_answers / self.total_questions) * 100

        return UserProgressRecord(
            user_id=self.user_id,
            topic=self.topic,
            session_id=self.session_id,
            start_level=self.initial_level,
            final_level=self.current_level,
            accuracy=round(accuracy, 2),
            questions_answered=self.total_questions,
            completed_at=datetime.now(),
        )


# --- In-memory Session Store ---
session_store: Dict[str, QuizSession] = {}

# --- In-memory User Progress Store (would be a database in production) ---
user_progress_store: List[UserProgressRecord] = []

# --- FastAPI App Initialization ---
app = FastAPI(title="Adaptive Quiz Platform")


# --- API Endpoints ---
@app.post("/quiz/start", response_model=QuizResponse)
def start_quiz(req: StartQuizRequest):
    """
    Start a new quiz session.
    Generates the first question based on the topic and initial proficiency.
    """
    session_id = str(uuid.uuid4())
    session = QuizSession(session_id, req.user_id, req.topic, req.proficiency)
    quiz_output = session.generate_quiz()
    session_store[session_id] = session

    logger.info(f"Started new quiz session for user {req.user_id}: {session_id}")

    return QuizResponse(
        session_id=session_id,
        user_id=req.user_id,
        quiz=quiz_output.get("quiz", []),
        current_level=session.current_level,
    )


@app.post("/quiz/submit", response_model=QuizResponse)
def submit_answers(req: SubmitAnswerRequest):
    """
    Submit answers for the current iteration.
    Stores the user's answers with question context and generates the next question
    unless the quiz is over. The quiz is considered over when 20 questions have been generated.
    """
    session = session_store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Validate that the user_id matches the session
    if session.user_id != req.user_id:
        raise HTTPException(
            status_code=403, detail="User ID does not match session owner"
        )

    # Save user's answers with question context for this iteration.
    session.add_history(req.answers)

    # If total questions reached 20, do not generate more questions.
    if session.total_questions >= 20:
        return QuizResponse(
            session_id=req.session_id,
            user_id=req.user_id,
            quiz=[],
            current_level=session.current_level,
        )
    else:
        quiz_output = session.generate_quiz()
        return QuizResponse(
            session_id=req.session_id,
            user_id=req.user_id,
            quiz=quiz_output.get("quiz", []),
            current_level=session.current_level,
        )


@app.post("/quiz/finish", response_model=FinalQuizResponse)
def finish_quiz(session_id: str, user_id: str):
    """
    Finish the quiz session.
    Return the complete history, performance summary, and final proficiency level.
    Store progress record and remove the session from the in-memory store.
    """
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Validate that the user_id matches the session
    if session.user_id != user_id:
        raise HTTPException(
            status_code=403, detail="User ID does not match session owner"
        )

    # Generate performance summary
    performance_summary = session.get_performance_summary()
    completed_at = datetime.now()

    # Create and store user progress record
    progress_record = session.create_progress_record()
    user_progress_store.append(progress_record)

    logger.info(f"Completed quiz session for user {user_id}: {session_id}")
    logger.info(
        f"Final proficiency: {session.current_level}, Accuracy: {performance_summary['accuracy']}%"
    )

    # Here you would push session.history to your database.
    final_history = session.history
    final_proficiency = session.current_level

    # Remove the session from the local store.
    del session_store[session_id]

    return FinalQuizResponse(
        session_id=session_id,
        user_id=user_id,
        final_proficiency=final_proficiency,
        history=final_history,
        performance_summary=performance_summary,
        completed_at=completed_at,
    )


# --- User Progress Endpoints ---
@app.get("/users/{user_id}/progress", response_model=List[UserProgressRecord])
def get_user_progress(user_id: str):
    """
    Get all quiz sessions and progress for a specific user.
    """
    user_records = [
        record for record in user_progress_store if record.user_id == user_id
    ]
    return user_records


@app.get(
    "/users/{user_id}/topics/{topic}/progress", response_model=List[UserProgressRecord]
)
def get_user_topic_progress(user_id: str, topic: str):
    """
    Get all quiz sessions and progress for a specific user and topic.
    """
    user_topic_records = [
        record
        for record in user_progress_store
        if record.user_id == user_id and record.topic == topic
    ]
    return user_topic_records
