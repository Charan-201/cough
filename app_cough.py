from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, validator, constr, StringConstraints, field_validator
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from sqlalchemy import create_engine, text
import logging
import random
import string
import json
from email_validator import validate_email, EmailNotValidError
from typing import Annotated
from typing_extensions import Annotated
from contextlib import contextmanager
import yagmail
import pytz
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security
import pandas as pd

# Add the security scheme
security = HTTPBearer()

# Configuration
class Settings:
    PROJECT_NAME = "Cough_API"
    VERSION = "1.0.0"
    API_PREFIX = "/api"
    
    # Database
    DATABASE_URL = "postgresql://postgres:1234567890@localhost:5434/cauvery"
    
    # Security
    SECRET_KEY = "bWq8yF2sVAiklcKD0QaGgfdf77eXw5gW"  # Move to environment variable
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_DAYS = 30
    OTP_EXPIRE_MINUTES = 5
    
    # CORS
    CORS_ORIGINS = ["*"]  # Update with specific origins in production
    
    # Rate limiting
    RATE_LIMIT_WINDOW = 3600  # 1 hour
    MAX_REQUESTS_PER_WINDOW = 100

settings = Settings()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database
engine = create_engine(settings.DATABASE_URL)

 #Load dataset
dataset_path = "C:/Users/saich/Downloads/cough_dataset 1.csv"
data = pd.read_csv(dataset_path)

# Normalize the dataset (convert to lowercase)
data = data.apply(lambda x: x.str.strip().str.lower() if x.dtype == "object" else x)

# Initialize FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class UserRegistration(BaseModel):
    # Using Annotated for string constraints
    mobile_number: Annotated[str, StringConstraints(pattern=r'^\d{10}$')]
    name: Annotated[str, StringConstraints(min_length=2, max_length=50)]
    dob: str
    gender: Annotated[str, StringConstraints(pattern='^(male|female|other)$')]
    pin_code: Annotated[str, StringConstraints(pattern=r'^\d{6}$')]
    email: Optional[EmailStr] = None

    @field_validator('dob')
    @classmethod
    def validate_dob(cls, v):
        try:
            dob = datetime.strptime(v, '%Y-%m-%d')
            if dob > datetime.now():
                raise ValueError("Date of birth cannot be in the future")
            if dob.year < 1900:
                raise ValueError("Invalid year of birth")
            return v
        except ValueError as e:
            raise ValueError("Invalid date format. Use YYYY-MM-DD")

class OTPRequest(BaseModel):
    mobile_number: Annotated[str, StringConstraints(pattern=r'^\d{10}$')]

class OTPValidation(BaseModel):
    mobile_number: Annotated[str, StringConstraints(pattern=r'^\d{10}$')]
    otp: Annotated[str, StringConstraints(pattern=r'^\d{6}$')]
    
class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_at: str
    user_details: dict

class PredictResponse(BaseModel):
    """Union of possible responses"""
    is_final: bool
    qid: Optional[str] = None
    q_title: Optional[str] = None
    q_subtitle: Optional[str] = None
    possible_answers: Optional[list] = None
    diagnosis: Optional[str] = None
    medication: Optional[str] = ""
    advice: Optional[str] = None
    red_flags: Optional[str] = None
    precautions: Optional[str] = None
    
# Utility functions
def generate_otp() -> str:
    """Generate a 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=6))

def get_current_ist_time() -> datetime:
    """Get current time in IST"""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)

def get_expiry_time(minutes: int) -> datetime:
    """
    Calculate expiry time in IST and convert to UTC for database storage
    """
    # Get current time in IST
    current_ist = get_current_ist_time()
    logger.info(f"Current IST time: {current_ist}")
    
    # Add expiration minutes
    expiry_ist = current_ist + timedelta(minutes=minutes)
    logger.info(f"Expiry IST time: {expiry_ist}")
    
    # Convert to UTC for database storage
    expiry_utc = expiry_ist.astimezone(pytz.UTC)
    logger.info(f"Expiry UTC time: {expiry_utc}")
    
    # Remove timezone info for database storage
    return expiry_utc.replace(tzinfo=None)

async def store_otp(mobile_number: str, otp: str, conn) -> None:
    """Store OTP in database"""
    expires_at = get_expiry_time(settings.OTP_EXPIRE_MINUTES)
    
    # Invalidate existing OTPs
    await invalidate_existing_otps(mobile_number, conn)
    
    # Store new OTP
    query = text("""
        INSERT INTO otp_store (mobile_number, otp, expires_at, is_valid)
        VALUES (:mobile_number, :otp, :expires_at, TRUE)
    """)
    
    conn.execute(query, {
        "mobile_number": mobile_number,
        "otp": otp,
        "expires_at": expires_at
    })
    conn.commit()  # Explicitly commit the insert
    logger.info(f"Stored new OTP for {mobile_number}")

async def invalidate_existing_otps(mobile_number: str, conn) -> None:
    """Invalidate all existing OTPs for a mobile number"""
    query = text("""
        UPDATE otp_store 
        SET is_valid = FALSE 
        WHERE mobile_number = :mobile_number
    """)
    conn.execute(query, {"mobile_number": mobile_number})

def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

async def verify_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """Get current user from token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

# API Endpoints
@contextmanager
def get_db_transaction():
    """Context manager to handle database transactions"""
    with engine.connect() as connection:
        try:
            yield connection
            connection.commit()
            logger.info("Transaction committed successfully")
        except Exception as e:
            connection.rollback()
            logger.error(f"Transaction rolled back due to error: {str(e)}")
            raise

@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user: UserRegistration):
    logger.info(f"Starting registration for user: {user.mobile_number}")
    
    try:
        with get_db_transaction() as conn:
            # First check if user exists
            check_query = text("""
                SELECT mobile_number 
                FROM users 
                WHERE mobile_number = :mobile_number
            """)
            
            logger.debug(f"Checking for existing user: {user.mobile_number}")
            result = conn.execute(
                check_query,
                {"mobile_number": user.mobile_number}
            ).fetchone()

            if result:
                logger.warning(f"User already exists: {user.mobile_number}")
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="User already exists"
                )

            # Insert new user
            insert_query = text("""
                INSERT INTO users (
                    mobile_number,name,dob,gender,pin_code,email,created_at,is_active
                ) VALUES (
                    :mobile_number,
                    :name,
                    :dob,
                    :gender,
                    :pin_code,
                    :email,
                    NOW(),
                    TRUE
                ) RETURNING id
            """)

            logger.debug("Executing insert query with parameters:")
            logger.debug(f"Query: {insert_query}")
            logger.debug(f"Parameters: {user.model_dump()}")

            try:
                result = conn.execute(
                    insert_query,
                    {
                        "mobile_number": user.mobile_number,
                        "name": user.name,
                        "dob": user.dob,
                        "gender": user.gender,
                        "pin_code": user.pin_code,
                        "email": user.email
                    }
                )
                
                # Get the inserted ID
                user_id = result.fetchone()[0]
                logger.info(f"User inserted successfully with ID: {user_id}")

                # Verify the insertion
                verify_query = text("""
                    SELECT id, mobile_number 
                    FROM users 
                    WHERE id = :user_id
                """)
                verification = conn.execute(
                    verify_query,
                    {"user_id": user_id}
                ).fetchone()

                if not verification:
                    raise Exception("Failed to verify user insertion")

                return {
                    "message": "User registered successfully",
                    "user_id": user_id
                }

            except SQLAlchemyError as e:
                logger.error(f"Database error during insertion: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Database error: {str(e)}"
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in registration: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user"
        )

# Add a test endpoint to directly insert a user
@app.post("/test-insert")
async def test_insert():
    """Test endpoint to verify database insertion"""
    try:
        with get_db_transaction() as conn:
            test_user = {
                "mobile_number": f"9999{random.randint(10000, 99999)}",
                "name": "Test User",
                "dob": "1990-01-01",
                "gender": "male",
                "pin_code": "123456",
                "email": "test@example.com"
            }

            insert_query = text("""
                INSERT INTO users (
                    mobile_number, name, dob, gender, 
                    pin_code, email, created_at, is_active
                ) VALUES (
                    :mobile_number, :name, :dob, :gender,
                    :pin_code, :email, NOW(), TRUE
                ) RETURNING id
            """)

            result = conn.execute(insert_query, test_user)
            user_id = result.fetchone()[0]

            return {
                "message": "Test insert successful",
                "user_id": user_id
            }

    except Exception as e:
        logger.error(f"Test insert failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/auth/request-otp")
async def request_otp(request: OTPRequest):
    """Request OTP for authentication"""
    try:
        with engine.connect() as conn:
            # Verify user exists
            user = conn.execute(
                text("SELECT mobile_number, email FROM users WHERE mobile_number = :mobile_number"),
                {"mobile_number": request.mobile_number}
            ).fetchone()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Generate and store OTP
            otp = generate_otp()
            await store_otp(request.mobile_number, otp, conn)
            
            # Send OTP notification
            if user.email:
                """Send OTP via email."""
                subject = "Your OTP Code"
                body = f"Your OTP is: {otp}. Please use this code to proceed. The OTP is valid for 5 minutes."

                # Create the email
                #msg = MIMEText(body)
                #msg["Subject"] = subject
                #msg["From"] = EMAIL_ADDRESS
                #msg["To"] = to_email

                try:
                # Connect to Gmail SMTP server
                    yag = yagmail.SMTP('gauukaran@gmail.com','fbwf gsur fzwp hfch')
                    yag.send(to=user.email, subject=subject, contents=body)
                except Exception as e:
                    print(f"Failed to send email: {e}")

            
            # Implement SMS sending here
            
            return {"message": "OTP sent successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OTP request error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process OTP request"
        )

@app.post("/auth/verify-otp", response_model=TokenResponse)
async def verify_otp_and_generate_token(validation: OTPValidation):
    """Verify OTP and generate access token"""
    try:
        with engine.connect() as conn:
            # Verify OTP
            query = text("""
                SELECT otp, expires_at, is_valid
                FROM otp_store
                WHERE mobile_number = :mobile_number
                AND otp = :otp
                AND is_valid = TRUE
                
                ORDER BY created_at DESC
                LIMIT 1
            """)
            
            otp_result = conn.execute(query, {
                "mobile_number": validation.mobile_number,
                "otp": validation.otp
            }).fetchone()

            if not otp_result:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or expired OTP"
                )

            # Get user details
            user = conn.execute(
                text("SELECT * FROM users WHERE mobile_number = :mobile_number"),
                {"mobile_number": validation.mobile_number}
            ).fetchone()

            # Invalidate used OTP
            await invalidate_existing_otps(validation.mobile_number, conn)

            # Generate token
            token_data = {
                "sub": validation.mobile_number,
                "name": user.name,
                "type": "access"
            }
            
            access_token = create_access_token(token_data)
            expire = datetime.utcnow() + timedelta(days=settings.ACCESS_TOKEN_EXPIRE_DAYS)

            # Log authentication
            conn.execute(
                text("""
                    INSERT INTO auth_logs (mobile_number, auth_type, success, created_at)
                    VALUES (:mobile_number, 'otp', TRUE, NOW())
                """),
                {"mobile_number": validation.mobile_number}
            )

            # Update last login
            conn.execute(
                text("UPDATE users SET last_login = NOW() WHERE mobile_number = :mobile_number"),
                {"mobile_number": validation.mobile_number}
            )

            return TokenResponse(
                access_token=access_token,
                token_type="bearer",
                expires_at=expire.isoformat(),
                user_details={
                    "mobile_number": user.mobile_number,
                    "name": user.name,
                    "email": user.email,
                    "pin_code": user.pin_code
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OTP verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify OTP"
        )
        

def process_prediction(answers: dict) -> dict:
    """Process the user's answers and determine the next question or final diagnosis"""
    try:
        logger.info(f"Processing prediction for answers: {answers}")
        
        # If no answers yet, return first question
        if not answers:
            return {
                "is_final": False,
                "qid": "Q1",
                "q_title": data.loc[0, "Question1"].capitalize(),
                "q_subtitle": data.loc[0, "Subtitle1"].capitalize(),
                "possible_answers": sorted([x.strip().capitalize() for x in data["Answer1"].dropna().unique().tolist() if x.strip()])
            }
            
        # Filter data based on all previous answers
        filtered_data = data.copy()
        for q_id, answer in answers.items():
            q_num = q_id.replace("Q", "")
            answer_col = f'Answer{q_num}'
            filtered_data = filtered_data[
                filtered_data[answer_col].str.lower() == answer.lower()
            ]
            
        if filtered_data.empty:
            return {
                "is_final": True,
                "diagnosis": "Unable to determine",
                "medication": "Please consult a doctor",
                "advice": "Visit a healthcare provider for proper diagnosis",
                "red_flags": "None",
                "precautions": "None"
            }

        # Check if we've reached the end
        next_q_num = len(answers) + 1
        question_col = f"Question{next_q_num}"
        
        if (question_col not in filtered_data.columns or 
            pd.isna(filtered_data.iloc[0][question_col])):
            
            first_row = filtered_data.iloc[0]
            return {
                "is_final": True,
                "Diagnosis": first_row["Diagnosis"].capitalize(),
                "Over the Counter medication": first_row["Over the counter medication"].capitalize(),
                "Advice": first_row["Advice"].capitalize(),
                "Red_flags": first_row["Red flag symptoms"].capitalize(),
                "Precautions": first_row["Precautions"].capitalize()
            }

        # Return next question
        first_row = filtered_data.iloc[0]
        return {
            "is_final": False,
            "qid": f"Q{next_q_num}",
            "q_title": first_row[question_col].capitalize(),
            "q_subtitle": first_row[f"Subtitle{next_q_num}"].capitalize(),
            "possible_answers": sorted([
                x.strip().capitalize()
                for x in filtered_data[f"Answer{next_q_num}"].dropna().unique().tolist() 
                if x.strip()
            ])
        }

    except Exception as e:
        logger.error(f"Error in prediction processing: {str(e)}", exc_info=True)
        raise Exception(f"Failed to process prediction: {str(e)}")
    
def get_question_columns(data_df):
    """Dynamically get question, subtitle, and answer columns from dataset"""
    columns = data_df.columns.tolist()
    question_cols = sorted([col for col in columns if col.startswith('Question')])
    subtitle_cols = sorted([col for col in columns if col.startswith('Subtitle')])
    answer_cols = sorted([col for col in columns if col.startswith('Answer')])
    return question_cols, subtitle_cols, answer_cols

class PredictRequest(BaseModel):
    id: str
    answers: Dict[str, str] = {}

def get_question_mapping(df):
    """
    Dynamically generate question mapping from dataset columns
    
    Args:
        df: pandas DataFrame containing the dataset
        
    Returns:
        dict: Mapping of question numbers to their corresponding columns
    """
    try:
        mapping = {}
        columns = df.columns.tolist()
        
        logger.debug(f"Available columns: {columns}")
        
        # Find all Answer columns to determine number of questions
        answer_columns = [col for col in columns if col.startswith('Answer')]
        max_questions = len(answer_columns)
        
        logger.debug(f"Found {max_questions} answer columns")
        
        # For each question number
        for i in range(1, max_questions + 1):
            answer_col = f'Answer{i}'
            subtitle_col = f'Subtitle{i}'
            
            # Skip if required columns don't exist
            if answer_col not in columns or subtitle_col not in columns:
                continue
                
            # Find corresponding question column
            # First try Question{i}
            question_col = f'Question{i}'
            if question_col not in columns:
                # Look for non-empty named question column
                for col in columns:
                    if col not in [answer_col, subtitle_col] and not any(c in col for c in ['Answer', 'Subtitle']):
                        if df[col].notna().any():  # Check if column has any non-NA values
                            first_value = df[col].iloc[0]
                            if pd.notna(first_value) and str(first_value).strip():
                                question_col = col
                                break
            
            if question_col in columns:
                mapping[i] = {
                    "question": question_col,
                    "subtitle": subtitle_col,
                    "answer": answer_col
                }
                logger.debug(f"Added mapping for question {i}: {mapping[i]}")
        
        if not mapping:
            raise ValueError("No valid question mappings found in dataset")
            
        logger.info(f"Successfully created mapping for {len(mapping)} questions")
        return mapping
        
    except Exception as e:
        logger.error(f"Error creating question mapping: {str(e)}")
        raise

@app.post("/cough/predict")
async def predict_cough(question_request: PredictRequest, current_user: dict = Depends(get_current_user)):
    try:
        # Get dynamic question mapping
        question_mapping = get_question_mapping(data)
        logger.debug(f"Question mapping: {question_mapping}")
        
        # Normalize answers
        normalized_answers = {
            key: value.strip().lower() 
            for key, value in question_request.answers.items()
        }
        logger.debug(f"Normalized answers: {normalized_answers}")

        # Initial question (no answers yet)
        if question_request.id == "00" and not normalized_answers:
            first_q = question_mapping[1]
            # Get possible answers for first question
            possible_answers = [
                str(x).strip().capitalize() 
                for x in data[first_q["answer"]].dropna().unique()
                if pd.notna(x) and str(x).strip()
            ]
            
            return {
                "qid": "Q1",
                "q_title": str(data[first_q["question"]].iloc[0]).capitalize(),
                "q_subtitle": str(data[first_q["subtitle"]].iloc[0]).capitalize(),
                "possible_answers": sorted(possible_answers),
                "is_final": False
            }

        # Filter data based on previous answers
        filtered_data = data.copy()
        for q_id, answer in normalized_answers.items():
            q_num = int(q_id.replace('Q', ''))
            if q_num in question_mapping:
                answer_col = question_mapping[q_num]["answer"]
                # Convert to string and normalize for comparison
                filtered_data = filtered_data[
                    filtered_data[answer_col].fillna('').str.strip().str.lower() == answer.lower()
                ]
                logger.debug(f"Filtered to {len(filtered_data)} rows after {q_id}")

        if filtered_data.empty:
            logger.warning("No matching path found")
            return {
                "is_final": True,
                "diagnosis": "Unable to determine",
                "medication": "Please consult a doctor",
                "advice": "Please seek medical attention for proper diagnosis",
                "red_flags": "None",
                "precautions": "None"
            }

        # Determine next question number
        next_q_num = len(normalized_answers) + 1

        # Check if we've reached the end
        if next_q_num not in question_mapping:
            return {
                "is_final": True,
                "Diagnosis": str(filtered_data['Diagnosis'].iloc[0]).capitalize(),
                "Over the Counter Medication": str(filtered_data['Over the counter medication'].iloc[0]).capitalize(),
                "Advice": str(filtered_data['Advice'].iloc[0]).capitalize(),
                "Red_flags": str(filtered_data['Red flag symptoms'].iloc[0]).capitalize(),
                "Precautions": str(filtered_data['Precautions'].iloc[0]).capitalize()
            }

        # Get next question details
        next_q = question_mapping[next_q_num]
        
        # Get possible answers for next question
        possible_answers = [
            str(x).strip().capitalize()
            for x in filtered_data[next_q["answer"]].dropna().unique()
            if pd.notna(x) and str(x).strip()
        ]

        if not possible_answers:
            return {
                "is_final": True,
                "Diagnosis": str(filtered_data['Diagnosis'].iloc[0]).capitalize(),
                "Over the Counter Medication": str(filtered_data['Over the counter medication'].iloc[0]).capitalize(),
                "Advice": str(filtered_data['Advice'].iloc[0]).capitalize(),
                "Red_flags": str(filtered_data['Red flag symptoms'].iloc[0]).capitalize(),
                "Precautions": str(filtered_data['Precautions'].iloc[0]).capitalize()
            }

        return {
            "qid": f"Q{next_q_num}",
            "q_title": str(filtered_data[next_q["question"]].iloc[0]).capitalize(),
            "q_subtitle": str(filtered_data[next_q["subtitle"]].iloc[0]).capitalize(),
            "possible_answers": sorted(possible_answers),
            "is_final": False
        }

    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )
        
# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )
    

# Required database tables
"""
-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    mobile_number VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(50) NOT NULL,
    dob DATE NOT NULL,
    gender VARCHAR(10) NOT NULL,
    pin_code VARCHAR(6) NOT NULL,
    email VARCHAR(255),
    created_at TIMESTAMP NOT NULL,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- OTP store table
CREATE TABLE otp_store (
    id SERIAL PRIMARY KEY,
    mobile_number VARCHAR(10) NOT NULL,
    otp VARCHAR(6) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    is_valid BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (mobile_number) REFERENCES users(mobile_number)
);

-- Authentication logs
CREATE TABLE auth_logs (
    id SERIAL PRIMARY KEY,
    mobile_number VARCHAR(10) NOT NULL,
    auth_type VARCHAR(20) NOT NULL,
    success BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY (mobile_number) REFERENCES users(mobile_number)
);

-- Prediction logs
CREATE TABLE prediction_logs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(10) NOT NULL,
    answers JSONB NOT NULL,
    result JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(mobile_number)
);

-- Create indexes
CREATE INDEX idx_users_mobile ON users(mobile_number);
CREATE INDEX idx_otp_mobile_valid ON otp_store(mobile_number, is_valid);
CREATE INDEX idx_auth_logs_mobile ON auth_logs(mobile_number);
CREATE INDEX idx_prediction_logs_user ON prediction_logs(user_id);
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
