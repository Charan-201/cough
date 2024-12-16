from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
password = "password"
hashed_password = pwd_context.hash(password)
print(f"Hashed password: {hashed_password}") 