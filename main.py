import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import tkinter as tk

df = pd.read_csv('data.csv')
df = pd.DataFrame(df,columns= ['gmat', 'gpa','work_experience','age','admitted'])
#print (df)
X = df[['gmat', 'gpa','work_experience','age']]
y = df['admitted']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
# tkinter GUI
root = tk.Tk()

canvas1 = tk.Canvas(root, width=500, height=350)
canvas1.pack()

# GMAT
label1 = tk.Label(root, text='            GMAT:')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry(root)
canvas1.create_window(270, 100, window=entry1)

# GPA
label2 = tk.Label(root, text='GPA:     ')
canvas1.create_window(120, 120, window=label2)

entry2 = tk.Entry(root)
canvas1.create_window(270, 120, window=entry2)

# work_experience
label3 = tk.Label(root, text='     Work Experience: ')
canvas1.create_window(140, 140, window=label3)

entry3 = tk.Entry(root)
canvas1.create_window(270, 140, window=entry3)

# Age input
label4 = tk.Label(root, text='Age:                               ')
canvas1.create_window(160, 160, window=label4)

entry4 = tk.Entry(root)
canvas1.create_window(270, 160, window=entry4)


def values():
    global gmat
    gmat = float(entry1.get())

    global gpa
    gpa = float(entry2.get())

    global work_experience
    work_experience = float(entry3.get())

    global age
    age = float(entry4.get())
    a = clf.predict([[gmat, gpa, work_experience, age]])
    if a == 2:
        b = "Candidate is admitted"
    elif a == 1:
        b = "Candidate is on the waiting list"
    else:
        b = "Candidate is not admitted"
    Prediction_result = ('  Predicted Result: ',a, b)
    label_Prediction = tk.Label(root, text=Prediction_result, bg='sky blue')
    canvas1.create_window(270, 280, window=label_Prediction)


button1 = tk.Button(root, text='      Predict      ', command=values, bg='green', fg='white', font=11)
canvas1.create_window(270, 220, window=button1)

root.mainloop()
