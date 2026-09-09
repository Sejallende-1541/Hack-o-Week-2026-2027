from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
from analysis import DataAnalyzer

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form["name"]
        subject = request.form["subject"]
        marks = request.form["marks"]
        attendance = request.form["attendance"]
        assignments = request.form["assignments"]

        # Append to CSV
        df = pd.DataFrame([[name, subject, marks, attendance, assignments]],
                          columns=["Name","Subject","Marks","Attendance","Assignments"])
        df.to_csv("students.csv", mode="a", header=not pd.io.common.file_exists("students.csv"), index=False)

        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    df = pd.read_csv("students.csv")
    analyzer = DataAnalyzer(df)
    clean_df = analyzer.clean_data()

    plot_marks = analyzer.plot_marks()
    plot_attendance = analyzer.plot_attendance()
    plot_grades = analyzer.plot_grades()
    plot_subject_avg = analyzer.plot_subject_average()
    summary = analyzer.class_summary()
    top_students = analyzer.top_student_per_subject()

    return render_template("dashboard.html",
                           result=clean_df.to_html(classes="table table-striped"),
                           plot_marks=plot_marks,
                           plot_attendance=plot_attendance,
                           plot_grades=plot_grades,
                           plot_subject_avg=plot_subject_avg,
                           summary=summary,
                           top_students=top_students.to_html(classes="table table-bordered"))

# ✅ New route for downloading student data
@app.route("/download")
def download():
    return send_file("students.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
