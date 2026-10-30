import numpy as np
import pandas as pd

# 1. Generate Mock Student Dataset using NumPy
np.random.seed(42)
num_students = 100

data = {
    "Student_ID": [f"STU{i:03d}" for i in range(1, num_students + 1)],
    "Gender": np.random.choice(["Male", "Female"], size=num_students),
    "Attendance_%": np.random.randint(55, 100, size=num_students).astype(float),
    "Math_Marks": np.random.randint(30, 100, size=num_students).astype(float),
    "Science_Marks": np.random.randint(30, 100, size=num_students).astype(float),
    "English_Marks": np.random.randint(30, 100, size=num_students).astype(float),
}

# Inject missing values (NaN) to demonstrate Pandas cleaning capabilities
data["Math_Marks"][np.random.choice(num_students, 5, replace=False)] = np.nan
data["Attendance_%"][np.random.choice(num_students, 3, replace=False)] = np.nan

df = pd.DataFrame(data)


# 2. Data Cleaning Module (Pandas & NumPy)
def clean_student_data(data_frame):
    cleaned_df = data_frame.copy()

    # Fill missing marks with the subject median
    math_median = cleaned_df["Math_Marks"].median()
    cleaned_df["Math_Marks"].fillna(math_median, inplace=True)

    # Fill missing attendance with the mean using NumPy/Pandas
    attendance_mean = cleaned_df["Attendance_%"].mean()
    cleaned_df["Attendance_%"].fillna(attendance_mean, inplace=True)

    return cleaned_df


df_cleaned = clean_student_data(df)


# 3. Performance Processing Engine
def process_performance(data_frame):
    processed_df = data_frame.copy()

    # Calculate Total & Average Marks across subjects
    subjects = ["Math_Marks", "Science_Marks", "English_Marks"]
    processed_df["Total_Marks"] = processed_df[subjects].sum(axis=1)
    processed_df["Average_Marks"] = processed_df[subjects].mean(axis=1)

    # Vectorized grading criteria using NumPy select conditions
    conditions = [
        (processed_df["Average_Marks"] >= 90),
        (processed_df["Average_Marks"] >= 80) & (processed_df["Average_Marks"] < 90),
        (processed_df["Average_Marks"] >= 70) & (processed_df["Average_Marks"] < 80),
        (processed_df["Average_Marks"] >= 60) & (processed_df["Average_Marks"] < 70),
        (processed_df["Average_Marks"] < 60),
    ]
    grades = ["A+", "A", "B", "C", "F"]
    processed_df["Grade"] = np.select(conditions, grades, default="F")

    # Pass/Fail status (Fail if Average < 60 or any subject score < 40)
    processed_df["Status"] = np.where(
        (processed_df["Average_Marks"] >= 60) & (processed_df[subjects].min(axis=1) >= 40),
        "Pass",
        "Fail",
    )

    return processed_df


df_final = process_performance(df_cleaned)


# 4. Analytics & Aggregated Insights
print("--- FIRST 5 RECORDS ---")
print(df_final.head(), "\n")

print("--- OVERALL CLASS STATISTICS ---")
print(df_final[["Attendance_%", "Total_Marks", "Average_Marks"]].describe(), "\n")

print("--- PERFORMANCE BY GENDER ---")
gender_analysis = df_final.groupby("Gender")[["Average_Marks", "Attendance_%"]].mean()
print(gender_analysis, "\n")

print("--- TOP 3 PERFORMERS ---")
top_performers = df_final.nlargest(3, "Average_Marks")
print(top_performers[["Student_ID", "Average_Marks", "Grade"]])

# 5. Export Report
df_final.to_csv("student_performance_report.csv", index=False)
print("\n[Success] Cleaned analysis report exported to 'student_performance_report.csv'")
