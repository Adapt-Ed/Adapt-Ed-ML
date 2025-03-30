from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from uuid import uuid4, UUID
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from datetime import datetime
import asyncio
from enum import Enum


# Proficiency levels enum
class ProficiencyLevel(str, Enum):
    NOVICE = "Novice"
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    EXPERT = "Expert"


# Data models
class QuizQuestion(BaseModel):
    question: str
    options: List[str] = Field(description="List of 4 options A, B, C, and D")
    correct_option: str = Field(description="Correct option from A, B, C, and D")


class QuizOutput(BaseModel):
    quiz: List[QuizQuestion] = Field(description="List of quiz questions")
    user_level: Optional[str] = Field(
        description="Level of the user based on the answers given"
    )


class UserAnswer(BaseModel):
    question: str
    user_option: str
    correct_option: str
    is_correct: bool


class QuizHistoryEntry(BaseModel):
    questions: List[QuizQuestion]
    user_answers: List[UserAnswer]
    calculated_level: Optional[str] = None


class QuizRequest(BaseModel):
    topic: str
    user_level: ProficiencyLevel


class UserAnswersRequest(BaseModel):
    session_id: UUID
    answers: List[str]  # List of selected options (A, B, C, or D) in order of questions


class QuizSessionResponse(BaseModel):
    session_id: UUID
    questions: List[QuizQuestion]
    iteration: int
    total_iterations: int = 4
    remaining_iterations: int
    current_level: Optional[str] = None


class QuizCompletion(BaseModel):
    session_id: UUID
    final_level: str
    correct_answers: int
    total_questions: int
    accuracy: float
    history: List[QuizHistoryEntry]


# Quiz Manager Class
class QuizManager:
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.sessions = {}
        self.api_key = api_key
        self.model_name = model_name
        self.questions_per_iteration = 5
        self.total_iterations = 4
        self.setup_langchain()

    def setup_langchain(self):
        # Output parser
        self.parser = JsonOutputParser(pydantic_object=QuizOutput)

        # LLM setup
        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key,
        )

        # Prompt template
        self.prompt_template = """
        You are a quiz generator. Your task is to generate 5 questions for a quiz based on the topic and user's proficiency level.

        Topic: {topic} [Topic of the quiz]
        User Proficiency: {level} [User Proficiency provided by them]
        Previous Questions and Answers of the user: {history}

        Your task is to predict the user's proficiency based on the history if history is not empty.
        If the history is empty, return user's proficiency as None.
        If the history is empty, it means you are generating the first 5 questions.

        Example Output:
        {{
        "quiz":[
            {{
                "question": "Question text here?",
                "options": ["A. Option A", "B. Option B", "C. Option C", "D. Option D"],
                "correct_option": "A"
            }},
            {{
                "question": "Another question?",
                "options": ["A. Option A", "B. Option B", "C. Option C", "D. Option D"],
                "correct_option": "B"
            }}
        ],
        "user_level": "Intermediate"
        }}

        - Only provide the answer in the specified JSON format without any extra information or justifications.
        - Always generate questions based on the user's proficiency.
        - We have 4 types of proficiency on our platform: Novice, Beginner, Intermediate, and Expert. When predicting the user level, keep these categories in mind.
        - Adjust question difficulty based on user performance in previous iterations.
        """

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["topic", "level", "history"],
        )

        self.chain = self.prompt | self.llm | self.parser

    async def create_session(self, topic: str, user_level: ProficiencyLevel) -> UUID:
        session_id = uuid4()
        self.sessions[session_id] = {
            "topic": topic,
            "user_level": user_level,
            "current_level": user_level,
            "history": [],
            "iteration": 0,
            "start_time": datetime.now(),
            "last_update": datetime.now(),
        }
        return session_id

    async def generate_questions(self, session_id: UUID) -> List[QuizQuestion]:
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = self.sessions[session_id]
        history_for_prompt = []

        # Format history for the prompt
        for entry in session["history"]:
            formatted_entry = []
            for idx, (question, answer) in enumerate(
                zip(entry["questions"], entry["user_answers"])
            ):
                formatted_entry.append(
                    {
                        "question": question["question"],
                        "user_answer": answer.user_option,
                        "correct_answer": answer.correct_option,
                        "is_correct": answer.is_correct,
                    }
                )
            history_for_prompt.append(formatted_entry)

        # Generate questions using LangChain
        try:
            result = await asyncio.to_thread(
                self.chain.invoke,
                {
                    "topic": session["topic"],
                    "level": session["current_level"],
                    "history": history_for_prompt if history_for_prompt else "[]",
                },
            )

            # Update session with new predicted level if available
            if result.get("user_level"):
                session["current_level"] = result["user_level"]

            session["iteration"] += 1
            session["last_update"] = datetime.now()

            return result["quiz"]
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating questions: {str(e)}"
            )

    async def process_answers(
        self, session_id: UUID, user_answers: List[str]
    ) -> QuizHistoryEntry:
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = self.sessions[session_id]

        # Get the most recent questions
        if not session["history"] or "current_questions" not in session:
            raise HTTPException(
                status_code=400,
                detail="No questions have been generated for this session yet",
            )

        current_questions = session["current_questions"]

        if len(user_answers) != len(current_questions):
            raise HTTPException(
                status_code=400,
                detail="Number of answers does not match number of questions",
            )

        # Process user answers
        processed_answers = []
        for i, (question, answer) in enumerate(zip(current_questions, user_answers)):
            is_correct = answer == question["correct_option"]
            processed_answers.append(
                UserAnswer(
                    question=question["question"],
                    user_option=answer,
                    correct_option=question["correct_option"],
                    is_correct=is_correct,
                )
            )

        # Create history entry
        history_entry = QuizHistoryEntry(
            questions=current_questions,
            user_answers=processed_answers,
            calculated_level=session["current_level"],
        )

        # Update session history
        session["history"].append(
            {
                "questions": current_questions,
                "user_answers": processed_answers,
                "calculated_level": session["current_level"],
            }
        )

        return history_entry

    async def get_quiz_completion(self, session_id: UUID) -> QuizCompletion:
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = self.sessions[session_id]

        # Calculate statistics
        total_questions = 0
        correct_answers = 0

        for entry in session["history"]:
            total_questions += len(entry["user_answers"])
            correct_answers += sum(
                1 for answer in entry["user_answers"] if answer.is_correct
            )

        accuracy = correct_answers / total_questions if total_questions > 0 else 0

        completion = QuizCompletion(
            session_id=session_id,
            final_level=session["current_level"],
            correct_answers=correct_answers,
            total_questions=total_questions,
            accuracy=round(accuracy * 100, 2),
            history=session["history"],
        )

        return completion

    def get_session(self, session_id: UUID):
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return self.sessions[session_id]

    def clean_old_sessions(self, hours: int = 24):
        current_time = datetime.now()
        sessions_to_remove = []

        for session_id, session in self.sessions.items():
            time_diff = (current_time - session["last_update"]).total_seconds() / 3600
            if time_diff > hours:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.sessions[session_id]


# Create FastAPI app
app = FastAPI(title="Adaptive Quiz Platform API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get the Quiz Manager
async def get_quiz_manager():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")

    # Singleton pattern
    if not hasattr(app, "quiz_manager"):
        app.quiz_manager = QuizManager(api_key=api_key)

    # Clean old sessions periodically
    app.quiz_manager.clean_old_sessions()

    return app.quiz_manager


# Routes
@app.post("/quiz/start", response_model=QuizSessionResponse)
async def start_quiz(
    request: QuizRequest, quiz_manager: QuizManager = Depends(get_quiz_manager)
):
    session_id = await quiz_manager.create_session(request.topic, request.user_level)
    questions = await quiz_manager.generate_questions(session_id)

    # Store current questions in session
    session = quiz_manager.get_session(session_id)
    session["current_questions"] = questions

    return QuizSessionResponse(
        session_id=session_id,
        questions=questions,
        iteration=1,
        remaining_iterations=quiz_manager.total_iterations - 1,
        current_level=request.user_level,
    )


@app.post("/quiz/answer", response_model=QuizSessionResponse)
async def submit_answers(
    request: UserAnswersRequest, quiz_manager: QuizManager = Depends(get_quiz_manager)
):
    # Process user answers
    await quiz_manager.process_answers(request.session_id, request.answers)
    session = quiz_manager.get_session(request.session_id)

    # Check if quiz is complete
    if session["iteration"] >= quiz_manager.total_iterations:
        raise HTTPException(status_code=400, detail="Quiz already completed")

    # Generate next questions
    questions = await quiz_manager.generate_questions(request.session_id)
    session["current_questions"] = questions

    return QuizSessionResponse(
        session_id=request.session_id,
        questions=questions,
        iteration=session["iteration"],
        remaining_iterations=quiz_manager.total_iterations - session["iteration"],
        current_level=session["current_level"],
    )


@app.get("/quiz/{session_id}/complete", response_model=QuizCompletion)
async def complete_quiz(
    session_id: UUID, quiz_manager: QuizManager = Depends(get_quiz_manager)
):
    return await quiz_manager.get_quiz_completion(session_id)


@app.get("/quiz/{session_id}/status", response_model=dict)
async def quiz_status(
    session_id: UUID, quiz_manager: QuizManager = Depends(get_quiz_manager)
):
    session = quiz_manager.get_session(session_id)
    return {
        "session_id": session_id,
        "topic": session["topic"],
        "current_level": session["current_level"],
        "iteration": session["iteration"],
        "total_iterations": quiz_manager.total_iterations,
        "remaining_iterations": quiz_manager.total_iterations - session["iteration"],
        "start_time": session["start_time"],
        "last_update": session["last_update"],
    }


# Main entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
