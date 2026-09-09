import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")   # ✅ non-GUI backend
import matplotlib.pyplot as plt

class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        self.df = self.df.dropna().reset_index(drop=True)
        self.df["Marks"] = pd.to_numeric(self.df["Marks"], errors="coerce")
        self.df["Attendance"] = pd.to_numeric(self.df["Attendance"], errors="coerce")
        self.df["Grade"] = self.df["Marks"].apply(self.calculate_grade)
        return self.df

    def calculate_grade(self, marks):
        if marks >= 90: return "A+"
        elif marks >= 75: return "A"
        elif marks >= 60: return "B"
        elif marks >= 40: return "C"
        else: return "F"

    def class_summary(self):
        avg_marks = np.mean(self.df["Marks"])
        top_student = self.df.loc[self.df["Marks"].idxmax()]["Name"]
        grade_counts = {g:c for g,c in self.df["Grade"].value_counts().items()}
        return {"Average Marks": round(avg_marks,2),
                "Top Student": top_student,
                "Grade Distribution": grade_counts}

    def top_student_per_subject(self):
        top_students = self.df.loc[self.df.groupby("Subject")["Marks"].idxmax()]
        return top_students[["Subject", "Name", "Marks"]]

    def plot_marks(self, save_path="static/marks.png"):
        os.makedirs("static", exist_ok=True)
        plt.figure(figsize=(6,4))
        sns.barplot(data=self.df, x="Name", y="Marks", hue="Name", palette="Blues_d", legend=False)
        plt.title("Marks by Student")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_attendance(self, save_path="static/attendance.png"):
        os.makedirs("static", exist_ok=True)
        plt.figure(figsize=(6,4))
        sns.scatterplot(data=self.df, x="Attendance", y="Marks", hue="Grade", palette="coolwarm", s=100)
        plt.title("Attendance vs Marks")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_grades(self, save_path="static/grades.png"):
        os.makedirs("static", exist_ok=True)
        plt.figure(figsize=(5,5))
        self.df["Grade"].value_counts().plot.pie(autopct="%1.1f%%", colors=sns.color_palette("pastel"))
        plt.title("Grade Distribution")
        plt.ylabel("")
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_subject_average(self, save_path="static/subject_avg.png"):
        os.makedirs("static", exist_ok=True)
        plt.figure(figsize=(6,4))
        subject_avg = self.df.groupby("Subject")["Marks"].mean().reset_index()
        sns.barplot(data=subject_avg, x="Subject", y="Marks", palette="viridis")
        plt.title("Average Marks per Subject")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path
