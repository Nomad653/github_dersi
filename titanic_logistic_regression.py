# %%
import pandas as pd
from PIL import Image
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# %% [markdown]
# Data yuklenir
# 
print("Salam")
print(st.title("Titanik faciəsindən sağ çıxa bilərdinizmi?"))

age = 0
sibling = 0 
gender = 0
p_class = 0
p_class = 0
gender = 0

@st.cache
def load_data():
	data = pd.read_csv("train.csv")


	le = LabelEncoder()

	data["Sex"] = le.fit_transform(data["Sex"])



	# %%
	data.drop("Cabin",axis=1,inplace=True)

	# %%
	data["Embarked"] = le.fit_transform(data["Embarked"])

	# %%
	data.drop(["Name","Ticket"],inplace=True,axis=1)

	# %%
	data["Age"].dropna(axis=0,inplace=True)

	# %%
	data.dropna(inplace=True)

	# %%
	data.drop("PassengerId",axis=1,inplace=True)

	# %%


	# %%
	x =  data.drop(["Survived","Parch","Fare","Embarked"],axis=1)
	y = data["Survived"]

	# %%



	# %%
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)


	LR = LogisticRegression()
	LR.fit(X_train,y_train)
	return LR
	# %%
loaded_data = load_data()	

def predict_data(f1,user_input):
	result = f1.predict(user_input)
	return result
# %%
#

# %%

def start(loaded_data):
	global p_class
	global gender
	global age
	global sibling
	if  p_class =="Birinci sinif":
		p_class = 1
	elif p_class =="İkinci sinif":
		p_class = 2
	else:
		p_class = 3
	input_data = {'Pclass':p_class,"Sex":gender,"Age":age,"SibSp":sibling}
	df = pd.DataFrame(data=input_data,index=[0])
	prediction = predict_data(loaded_data,df)
	predict_probability = loaded_data.predict_proba(df)
	if prediction[0] == 1:
		st.subheader('{}% ehtimalla sağ qalardınız.'.format(round(predict_probability[0][1]*100 , 3)))
	else:
		st.subheader('{}% ehtimalla ölərdiniz'.format(round(predict_probability[0][0]*100 , 3)))
# %%

age = st.slider("Yaşınız", 1, 100,1)
sibling = st.slider("Sizinlə birlikdə olan ailə üzvlərinizin sayı",1,10,1)
gender = st.selectbox("Cins", options = ["Kişi","Qadın"] )
p_class = st.selectbox("Sərnişin sinfi",options=['Birinci sinif' , 'İkinci sinif' , 'Üçüncü sinif'])
gender = 1 if gender =="Kişi" else 0
if st.button("Hesabla"):
	start(loaded_data)

#print(classification_report(y_test,prediction))

# %%


# %%



