### **User Analytics in the Telecommunication Industry**

>> This project involves analyzing customer data from TellCo, a mobile service provider in the Republic of Pefkakia. The goal is to provide a detailed report to a potential investor on opportunities for growth and to make a recommendation on whether TellCo is worth buying or selling. The project will analyze a telecommunication dataset that includes information about customer behavior and activities on the network.

### **Project Overview**

This project is divided into four main objectives:

1. User Overview Analysis: Understand the customer base by identifying key patterns and trends in user behavior, handset usage, and app engagement.
2. User Engagement Analysis: Measure and analyze how engaged customers are with various applications and services.
3. User Experience Analysis: Examine network performance metrics and device characteristics to evaluate the quality of user experience.
4. User Satisfaction Analysis: Assess overall customer satisfaction by combining insights from user engagement and experience analyses.

### **Data**

The project uses a dataset extracted from a month of aggregated xDR (data session detail records) that includes detailed information about customer activities, such as:

* Number of sessions, session duration, and data usage across different applications (Social Media, Google, Email, YouTube, Netflix, Gaming, and Others).
* Network performance metrics like TCP retransmission, Round Trip Time (RTT), and Throughput.
* Handset types and other device characteristics.

### **Key Deliverables**

* Data Preparation and Cleaning: Reusable code for data preprocessing, including handling missing values, outliers, and transformations.
* Dashboard: An interactive web-based dashboard to present key insights and findings from the data analysis.
* Feature Store: A reusable feature store for storing selected features for later use in similar problems.
* Project Code: Code that is installable via pip, includes unit tests with good coverage.

### **Tasks and Sub-Tasks**

**1. User Overview Analysis**
* Identify top handsets and manufacturers used by customers.
* Analyze user behavior across various applications (Social Media, Google, Email, YouTube, Netflix, Gaming).
* Perform exploratory data analysis (EDA) to understand customer trends and behaviors.
  
**2. User Engagement Analysis**
* Calculate engagement metrics like session frequency, duration, and total traffic per user.
* Perform clustering to segment users based on engagement levels.
* Visualize user engagement data and interpret key findings.

**3. User Experience Analysis**
* Evaluate network parameters like TCP retransmission, RTT, and throughput.
* Analyze user experience based on device characteristics and network performance.
* Perform clustering to segment users based on experience metrics.

**4. User Satisfaction Analysis**
* Calculate engagement and experience scores for each user.
* Combine these scores to derive a satisfaction score.
* Build a regression model to predict customer satisfaction and identify key influencing factors.

### **Tools and Libraries**

* Python: For data processing, analysis, and model building.
* Pandas, NumPy, Scikit-learn: For data manipulation and machine learning.
* Matplotlib, Seaborn: For data visualization.
* Streamlit or Flask: For building the interactive dashboard.

### **Project Structure**

* data/: Contains raw and processed data files.
* notebooks/: Jupyter notebooks for data exploration and analysis.
* scripts/: Python scripts for data preprocessing, modeling, and visualization.
* dashboard/: Code for the web-based dashboard.

