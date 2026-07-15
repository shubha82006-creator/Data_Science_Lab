import pandas as pd
import matplotlib.pyplot as plt
import os

# Create images folder if it doesn't exist
os.makedirs("images", exist_ok=True)

# Read the Excel file
df = pd.read_excel("students_fake_data.xlsx")

print("\n========== STUDENT DATA ==========")
print(df.head())

subjects = ["Maths", "Science", "English", "Computer"]

# -----------------------------
# Overall Class Topper
# -----------------------------
topper = df.loc[df["Total"].idxmax()]

print("\n========== OVERALL CLASS TOPPER ==========")
print(f"Student ID : {topper['Student ID']}")
print(f"Name       : {topper['Student Name']}")
print(f"Total      : {topper['Total']}")
print(f"Percentage : {topper['Percentage']}%")

# -----------------------------
# Subject-wise Analysis
# -----------------------------
for subject in subjects:

    highest = df[subject].max()
    lowest = df[subject].min()

    highest_students = df[df[subject] == highest]
    lowest_students = df[df[subject] == lowest]

    print(f"\n========== {subject.upper()} ==========")

    print(f"\nHighest Marks : {highest}")
    print("Student(s):")
    for name in highest_students["Student Name"]:
        print("•", name)

    print(f"\nLowest Marks : {lowest}")
    print("Student(s):")
    for name in lowest_students["Student Name"]:
        print("•", name)

    # Histogram
    plt.figure(figsize=(8,5))

    plt.hist(df[subject],
             bins=10,
             edgecolor="black")

    plt.axvline(highest,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Highest = {highest}")

    plt.axvline(lowest,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Lowest = {lowest}")

    plt.title(f"{subject} Marks Distribution")
    plt.xlabel("Marks")
    plt.ylabel("Number of Students")
    plt.legend()

    # Save image
    plt.savefig(f"images/{subject}_Histogram.png",
                dpi=300,
                bbox_inches="tight")

    plt.show()
    plt.close()

# -----------------------------
# Overall Class Histogram
# -----------------------------
plt.figure(figsize=(8,5))

plt.hist(df["Total"],
         bins=10,
         edgecolor="black")

plt.axvline(topper["Total"],
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Topper = {topper['Student Name']}")

plt.title("Overall Class Performance")
plt.xlabel("Total Marks")
plt.ylabel("Number of Students")
plt.legend()

plt.savefig("images/Overall_Class_Performance.png",
            dpi=300,
            bbox_inches="tight")

plt.show()
plt.close()

print("\n==========================================")
print("Analysis Completed Successfully!")
print("Graphs saved in the 'images' folder.")
print("==========================================")