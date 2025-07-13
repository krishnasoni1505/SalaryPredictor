import streamlit as st
import pandas as pd
import joblib

# Load trained Random Forest model
model = joblib.load(r'C:\Users\krish\OneDrive\Desktop\JupyterNotebooks\SalaryPredictionModel\salaryPredictionRf.pkl')


# App Title
st.title("💼 Salary Prediction App")
st.write("Estimate an employee's salary based on their personal and professional profile.")

# Sidebar for user inputs
st.sidebar.header("Enter Employee Details")

job_titles = [
    'Software Engineer', 'Data Scientist', 'Others', 'Software Engineer Manager',
    'Data Analyst', 'Senior Project Engineer', 'Product Manager',
    'Full Stack Engineer', 'Marketing Manager', 'Senior Software Engineer',
    'Back end Developer', 'Front end Developer', 'Marketing Coordinator',
    'Junior Sales Associate', 'Financial Manager', 'Marketing Analyst',
    'Software Developer', 'Operations Manager', 'Human Resources Manager',
    'Director of Marketing', 'Web Developer', 'Research Director', 'Product Designer',
    'Content Marketing Manager', 'Sales Associate', 'Director of HR',
    'Research Scientist', 'Senior Product Marketing Manager', 'Marketing Director',
    'Sales Director', 'Senior Data Scientist', 'Junior HR Generalist',
    'Junior Software Developer', 'Receptionist', 'Director of Data Science',
    'Sales Manager', 'Digital Marketing Manager', 'Junior Marketing Manager',
    'Junior Software Engineer', 'Senior Research Scientist',
    'Human Resources Coordinator', 'Senior Human Resources Manager',
    'Junior Web Developer', 'Senior HR Generalist', 'Junior Sales Representative',
    'Financial Analyst', 'Sales Executive', 'Sales Representative',
    'Front End Developer', 'Junior HR Coordinator'
]


# Input form
def user_input_features():
    age = st.sidebar.slider('Age', 18, 65, 30)
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    education = st.sidebar.selectbox('Education Level', ("High School", "Bachelor's", "Master's", "PhD"))
    job_title = st.sidebar.selectbox(
        'Job Title',
        options=job_titles,
        index=0  # default value
    )

    experience = st.sidebar.slider('Years of Experience', 0, 40, 5)

    data = {
        'Age': age,
        'Gender': gender,
        'Education Level': education,
        'Job Title': job_title,
        'Years of Experience': experience
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Show input values
st.subheader("Employee Profile")
st.write(input_df)

# Mapping function matching notebook's LabelEncoder order
# This function encodes the input DataFrame to match the model's expected input format
def encode_inputs(df):
    gender_map = {'Male': 1, 'Female': 0}
    education_map = {
        "High School": 0, 
        "Bachelor's": 1, 
        "Master's": 2, 
        "PhD": 3
    }
    
    df['Gender'] = df['Gender'].map(gender_map)
    df['Education Level'] = df['Education Level'].map(education_map)
    
    job_titles = [
        'Software Engineer', 'Data Analyst', 'Senior Manager',
        'Sales Associate', 'Director', 'Sales Executive',
        'Marketing Manager', 'Financial Manager', 'Director of Marketing'
    ]
    job_title_map = {title: idx for idx, title in enumerate(job_titles)}
    df['Job Title'] = df['Job Title'].map(job_title_map).fillna(0)  # unknown titles mapped as 0
    
    return df

# Encode input
encoded_input = encode_inputs(input_df)

# Function to format numbers in Indian style
def indian_format(n):
    s = str(int(n))
    if len(s) <= 3:
        return s
    else:
        last_three = s[-3:]
        other = s[:-3]
        parts = []
        while len(other) > 2:
            parts.append(other[-2:])
            other = other[:-2]
        if other:
            parts.append(other)
        return ','.join(parts[::-1]) + ',' + last_three

# Prediction button
if st.button('Predict Salary'):
    salary = model.predict(encoded_input)
    formatted_salary = indian_format(salary[0])
    st.subheader(f"🤑 Predicted Salary: ₹{formatted_salary}")

# Footer
st.markdown("---")
st.markdown("Created by **Krishna Soni** | ML Salary Prediction App")
