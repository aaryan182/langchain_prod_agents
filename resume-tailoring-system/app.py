from dotenv import load_dotenv
load_dotenv()

from router import build_router

RESUME = """
Software engineer with 2 years experience in Python, FastAPI, PostgreSQL.
"""

JD = """
Looking for backend engineer with Python, APIs, databases, and system design.
"""

def main():
    router = build_router()
    result = router(RESUME, JD)

    print(result)

if __name__ == "__main__":
    main()