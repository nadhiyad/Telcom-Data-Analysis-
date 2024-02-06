import streamlit as st
import pandas as pd
import altair as alt
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np


st.set_page_config(layout="wide")

with open('webstyle.css') as f:
    css_content = f.read()

st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)

st.markdown("<header>Telecom Data Analytics Platform</header>", unsafe_allow_html=True)


st.title("Welcome to Telecom Dashboard")

satisfaction_dashboard = pd.read_csv('satisfaction_data.csv')
engagement_dashboard = pd.read_csv('engagement_data.csv')
experience_dashboard = pd.read_csv('experience_data.csv')
telecom_data = pd.read_csv('telcom_data.csv')

# Row A
a1, a2, a3, a4 = st.columns(4)
a1.image(Image.open('telecom_logo.png'))  
with a2:
    st.write("###  Score for Cluster  (Kmean=2)")
    st.metric(".", "Satisfaction_Score", "1.65M")
    st.metric(".","Experience_Score", "99.89%")

with a3:
    st.write("### Heatmap for Satisfaction Analysis")
    numeric_columns = satisfaction_dashboard.select_dtypes(include=np.number).columns
    columns_to_drop = ["Satisfaction Score", "MSISDN/Number"]
    numeric_columns_filtered = [col for col in numeric_columns if col not in columns_to_drop]
    corr_matrix = satisfaction_dashboard[numeric_columns_filtered].corr()
    plt.figure(figsize=(11, 6.5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap for Satisfaction Analysis')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    plt.clf()

with a4:
    st.write("### Top Handset types for Top 3 Manufacturer")
    top_3_manufacturers =telecom_data.groupby('Handset Manufacturer')['MSISDN/Number'].count().sort_values(ascending=False).head(3)
    filtered_data = telecom_data[telecom_data['Handset Manufacturer'].isin(top_3_manufacturers.index)]
    top_1_handsets_per_manufacturer = (
    filtered_data.groupby(['Handset Manufacturer', 'Handset Type'])
    ['MSISDN/Number'].count().groupby('Handset Manufacturer', group_keys=False).nlargest(1)
)
    top_1_handsets_df = top_1_handsets_per_manufacturer.reset_index(name='Count')
    table_height = 100  
    table_width = 600 
    st.table(top_1_handsets_df)

b1, b2, b3 = st.columns([3,3,3])  
with b1:
    st.write("### Top 10 Satisfied Customers")
    top_10_satisfied_customers = satisfaction_dashboard.nlargest(10, 'Satisfaction Score')[['MSISDN/Number', 'Satisfaction Score']]
    top_10_satisfied_customers['MSISDN/Number'] = top_10_satisfied_customers['MSISDN/Number'].astype(str)
    colors = sns.color_palette('husl', n_colors=len(top_10_satisfied_customers))
    plt.figure(figsize=(5, 3.5))
    plt.bar(range(len(top_10_satisfied_customers)), top_10_satisfied_customers['Satisfaction Score'], color=colors)  # Corrected here
    plt.title('Top 10 Satisfied Customers')
    plt.xlabel('MSISDN/Number')
    plt.ylabel('Satisfaction Score')
    plt.xticks(range(len(top_10_satisfied_customers)), top_10_satisfied_customers['MSISDN/Number'], rotation=45, ha='right')  # Adjusted here
    plt.tight_layout()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Display the bar plot in Streamlit
    st.pyplot()


with b2:
    st.write("### Less Engaged Users")
    data = engagement_dashboard
    plt.figure(figsize=(5, 3))
    sns.scatterplot(x='Euclidean Distance', y='Total Duration', hue='Cluster', data=data, palette='viridis', alpha=0.7)
    plt.title('Engagement Scores and Clusters')
    plt.xlabel('Euclidean Distance to Less Engaged Cluster')
    plt.ylabel('Total Duration')
    plt.legend(title='Cluster')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

with b3:
    st.write("### Worst Experience Cluster")
    data = experience_dashboard
    plt.figure(figsize=(5, 3.2))
    sns.scatterplot(x='Euclidean Distance', y='Avg RTT DL (ms)', hue='Cluster', data=data, palette='viridis', alpha=0.7)
    plt.title('Experience Scores and Clusters')
    plt.xlabel('Euclidean Distance to Worst Experience Cluster')
    plt.ylabel('Avg RTT DL (ms)')
    plt.legend(title='Cluster')
    st.pyplot()


c1, c2, c3=st.columns([4,4,4])

with c1:
    st.write("### Top 3 Manufacturer Types")
    data=telecom_data
    top_3_manufacturers =data.groupby('Handset Manufacturer')['MSISDN/Number'].count().sort_values(ascending=False).head(3)

    plt.figure(figsize=(5,4))
    sns.barplot(x=top_3_manufacturers.index, y=top_3_manufacturers.values, palette=["skyblue", "salmon", "lightgreen"])
    plt.title('Top 3 Manufacturer Types')
    plt.xlabel('Handset Manufacturer')
    plt.ylabel('Count')
    st.pyplot()
plt.clf() 

    
with c2:
    st.write("### Top 10 Handset Type")
    data=telecom_data
    top_10_handsets = data.groupby('Handset Type')['MSISDN/Number'].count().sort_values(ascending=False).head(10)
    plt.figure(figsize=(5, 2.1))
    sns.barplot(x=top_10_handsets.index, y=top_10_handsets.values, palette="viridis")
    plt.title('Top 10 Handset Types')
    plt.xlabel('Handset Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')  
    st.pyplot()
plt.clf() 


with c3:
    st.write("### Actuals Vs Predicted")
    X = satisfaction_dashboard[['Session Frequency', 'Total Duration', 'Total DL', 'Total UL']]
    y = satisfaction_dashboard['Satisfaction Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    predictions = rf_model.predict(X_test)
    plt.figure(figsize=(5, 4))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.title('Actual vs Predicted Satisfaction Score')
    plt.xlabel('Actual Satisfaction Score')
    plt.ylabel('Predicted Satisfaction Score')
    x_line = np.linspace(min(y_test), max(y_test), 100)
    plt.plot(x_line, x_line, color='red', linestyle='--', label='Perfect Fit')
    plt.legend()
    st.pyplot()


# Footer
st.markdown("<footer>© 2024 Telcom_Data_Analysis</footer>", unsafe_allow_html=True)