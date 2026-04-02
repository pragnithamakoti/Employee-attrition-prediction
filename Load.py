import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/Employee.csv')

print(df.head())

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

sns.countplot(x='Attrition', data=df)
plt.show()

sns.countplot(x='OverTime', hue='Attrition', data=df)
plt.title("Overtime vs Attrition")
plt.show()
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.title("Salary vs Attrition")
plt.show()