import numpy as np
import pickle
import streamlit as st
import sklearn

FOLDER_PATH=""
STUDENT_TRAINED_MODEL_FILENAME = r"C:\Users\GOUTHAMI\OneDrive\Diabetes project\model_diabetes.pickle"
file_path=STUDENT_TRAINED_MODEL_FILENAME 
with open(file_path,'rb') as readfile:
    loaded_model=pickle.load(readfile)
    
def make_pred(ip_data):
    ip_num=np.array(ip_data).reshape(1,-1)
    pred=loaded_model.predict(ip_num)
    print(pred)
    if pred==1:
         st.success('Person has diabetes :thumbsdown:')
    else:
        st.success('Person does not have diabtes:thumbsup:')

        
def main():
    st.title("Diabetes Prediction  :honey_pot:")
    age=st.slider("Choose age",0,100)
    preg=st.slider("Choose pregnancies",0,20)
    glucose=st.number_input("glucose level")
    bp=st.number_input("Blood Pressure")
    skin_thickness=st.slider("Skin Thickness",0,100)
    insulin=st.number_input("Insulin level")
    bmi=st.number_input("BMI")
    DiabetesPedigreeFunction=st.number_input("Diabetes Pedigree Function")
    
    res=''
    if st.button('Check Results'):
        res=make_pred([preg,glucose,bp,skin_thickness,insulin,bmi,DiabetesPedigreeFunction,age])
        
    
if __name__=='__main__':
    main()
