from flask import Flask, render_template, request, redirect, url_for, session
import os
from PyPDF2 import PdfReader
from resume_analyzer import analyze_resume
from dotenv import load_dotenv
import mysql.connector
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.config["UPLOAD_FOLDER"] = "uploads"
def get_db_connection():
    return mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
    )
@app.route("/test-db")
def test_db():

    db = get_db_connection()

    if db.is_connected():
        return "MySQL Connected Successfully!"

    return "MySQL Connection Failed!"
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/features")
def features():
    return render_template("features.html")

@app.route("/workflow")
def workflow():
    return render_template("workflow.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/result")
def result():
    analysis = session.get("analysis")
    return render_template("result.html", analysis=analysis)

@app.route("/preview-resume")
def preview_resume():
    from flask import send_file

    upload_folder = app.config["UPLOAD_FOLDER"]

    files = os.listdir(upload_folder)

    if not files:
        return "No resume found!"

    latest_file = max(
        [os.path.join(upload_folder, f) for f in files],
        key=os.path.getmtime
    )

    return send_file(latest_file, mimetype="application/pdf")


@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "POST":

        file = request.files.get("resume")

        if file:

            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Read PDF
            reader = PdfReader(file_path)

            resume_text = ""

            for page in reader.pages:
                text = page.extract_text()

                if text:
                    resume_text += text

            print("\n----- RESUME TEXT -----")
            print(resume_text)
            print("----- END RESUME TEXT -----\n")

            analysis = analyze_resume(resume_text)

            print("\n----- ANALYSIS RESULT -----")
            print(analysis)
            print("----- END ANALYSIS -----\n")

            session["analysis"] = analysis

            return redirect(url_for("result"))
        return "No file selected!"

    return render_template("upload.html")
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        print("Email:", email)
        print("Password:", password)

        return "Login successful!"

    return render_template("login.html")
@app.route("/register")
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        # Check whether passwords match
        if password != confirm_password:
            return "Passwords do not match!"

        # Connect to MySQL
        db = get_db_connection()
        cursor = db.cursor()

        # Check if email already exists
        cursor.execute(
            "SELECT * FROM users WHERE email = %s",
            (email,)
        )

        existing_user = cursor.fetchone()

        if existing_user:
            cursor.close()
            db.close()
            return "Email already registered!"

        # Insert new user
        cursor.execute(
            """
            INSERT INTO users (name, email, password)
            VALUES (%s, %s, %s)
            """,
            (name, email, password)
        )

        db.commit()

        cursor.close()
        db.close()

        return redirect(url_for("login"))

    return render_template("register.html")


if __name__ == "__main__":
    app.run(debug=True)