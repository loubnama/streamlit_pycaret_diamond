import streamlit as st
from pycaret.datasets import get_data
from pycaret.classification import *
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
st.title("project1")
st.header("Diamond")
df=get_data("/Users/lubna/Desktop/diamond")
st.write(df.shape)
st.write(df.columns)
df.head()
print(df['Carat Weight'].mean())
df['over Price']=df['Price']>12000
df["over weight"]=df["Carat Weight"]>1.4
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='Cut',hue='over Price',data=df)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
st.write("Notice from the figure that <Ideal> wind has the highest price of all types, unlike <Fair Cut> which has a lower price")
plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
sns.countplot(x='Color',hue='over Price',data=df)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.pyplot()
st.write("We notice the price relationship with the color of the diamond, when the color G is the most expensive of all the colors")
st.write("then the color H, then F, then I, then D")
st.write("and the least expensive and desirable is the color E")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.subplot(2,2,2)
image_path = '/Users/lubna/Desktop/screen/Screenshot 2024-04-22 at 1.15.55.png'
img = mpimg.imread(image_path)
st.set_option('deprecation.showPyplotGlobalUse', False)

plt.imshow(img)
plt.axis('off')
plt.show()
st.pyplot()
sns.countplot(x='Report',hue='over Price',data=df)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.pyplot()
st.write("We notice a very large difference in the price of diamonds that have a certificate     from GIA, unlike AGSL")
sns.countplot(x='Clarity',hue='over Price',data=df)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.pyplot()
st.write("We notice the effect of diamond purity on its price. VS2 diamonds are more expensive, then VS2, and then SL2. Unlike FL, they are very rare.")
sns.countplot(x='over weight',hue='over Price',data=df)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.pyplot()
st.write("We notice that when a diamond has a high weight, its price is very expensive compared to a lower weight")
plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
sns.countplot(x='Polish',hue='over Price',data=df)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.pyplot()
st.write("We note that Ex is the most expensive, then VG, then ID, and the last is G")
image_path = '/Users/lubna/Desktop/screen/Screenshot 2024-04-22 at 1.32.40.png'
img = mpimg.imread(image_path)
plt.figure(figsize=(30,30))
plt.subplot(2,2,2)
st.set_option('deprecation.showPyplotGlobalUse', False)

plt.imshow(img)
plt.axis('off')
plt.show()
st.pyplot()
sns.countplot(x='Symmetry',hue='over Price',data=df)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("We note that EX is the most expensive But at a similar rate whith VG, then G, and the last is ID")
st.pyplot()
#There are no missing values
df.isna().sum()
sns.catplot(x='Carat Weight',col='over Price',data=df,kind='box')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.pyplot()
st.write("From the following chart BOX, we find that the heavier the diamond, the higher its price")
st.write("It explains the spread of diamond types according to purity, and we discover the absence and rarity of the Fl type") 
cr=df['Clarity'].value_counts().head().plot.pie()
st.set_option('deprecation.showPyplotGlobalUse', False)

st.pyplot(cr.figure)
plt.show()
st.write(df.dtypes)
#
setup(data=df, target="over Price")
best_model = compare_models()
st.write("In the diagrams we find that it is classification better ")
plt.figure(figsize=(8, 6))
preds = predict_model(best_model)
cm = confusion_matrix(preds['over Price'], preds['over weight'])
st.set_option('deprecation.showPyplotGlobalUse', False)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
st.pyplot()

