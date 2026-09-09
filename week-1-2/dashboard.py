# dashboard.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Define Student class (OOP) ---
class Student:
    def __init__(self, prn, name, marks, attendance):
        self.prn = prn
        self.name = name
        self.marks = marks  # dict of subject: marks
        self.attendance = attendance

    def average_score(self):
        return np.mean(list(self.marks.values()))

    def __repr__(self):
        return f"{self.name} (PRN: {self.prn}) - Avg: {self.average_score():.2f}, Attendance: {self.attendance}%"

# --- Step 2: Function to collect student input ---
def collect_students():
    students = []
    while True:
        prn = input("Enter PRN: ")
        name = input("Enter Name: ")

        # Marks input
        subjects = ["Math", "Science", "English"]
        marks = {sub: int(input(f"Enter marks for {sub}: ")) for sub in subjects}

        attendance = int(input("Enter attendance percentage: "))

        students.append(Student(prn, name, marks, attendance))

        more = input("Add another student? (y/n): ").lower()
        if more != "y":
            break
    return students

# --- Step 3: Convert to Pandas DataFrame ---
def make_dashboard(students):
    data = {
        "PRN": [s.prn for s in students],
        "Name": [s.name for s in students],
        "Math": [s.marks["Math"] for s in students],
        "Science": [s.marks["Science"] for s in students],
        "English": [s.marks["English"] for s in students],
        "Average": [s.average_score() for s in students],
        "Attendance": [s.attendance for s in students]
    }
    df = pd.DataFrame(data)
    return df

# --- Step 4: Dashboard Analytics ---
def show_dashboard(df):
    print("\n📊 Student Dashboard Data:")
    print(df)

    # Top student by average score
    top_student = df.loc[df["Average"].idxmax()]
    print(f"\n🏆 Top Student by Marks: {top_student['Name']} (Avg: {top_student['Average']:.2f})")

    # Top student by attendance
    top_attendance = df.loc[df["Attendance"].idxmax()]
    print(f"📅 Top Student by Attendance: {top_attendance['Name']} ({top_attendance['Attendance']}%)")

    # Subject averages
    print("\n📈 Average marks by subject:")
    print(df[["Math", "Science", "English"]].mean())

    # --- Visualizations ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Name", y="Average", data=df, palette="viridis")
    plt.xticks(rotation=45)
    plt.title("Average Marks per Student")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.heatmap(df[["Math", "Science", "English"]], annot=True, cmap="coolwarm")
    plt.title("Subject Scores Heatmap")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Name", y="Attendance", data=df, palette="magma")
    plt.xticks(rotation=45)
    plt.title("Attendance Percentage per Student")
    plt.show()

# --- Step 5: Main Program ---
if __name__ == "__main__":
    students = collect_students()
    df = make_dashboard(students)
    show_dashboard(df)
