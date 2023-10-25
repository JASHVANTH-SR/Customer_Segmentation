import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# Customize the layout
st.set_page_config(page_title="Customer Classification", page_icon=":bar_chart:")

st.title("Customer Classification and Profiling")
st.write("Explore and visualize the customer dataset.")

# Load the dataset
data = pd.read_csv("customer_classification_data.csv")

# Sidebar with options
st.sidebar.header("Customer Classification and Profiling")
selected_feature = st.sidebar.selectbox("Select Feature for Analysis", data.columns)

# Data filtering
st.sidebar.header("Data Filtering")
min_age = st.sidebar.slider("Minimum Age", min_value=int(data["Age"].min()), max_value=int(data["Age"].max()))
max_age = st.sidebar.slider("Maximum Age", min_value=int(data["Age"].min()), max_value=int(data["Age"].max()), value=int(data["Age"].max()))

filtered_data = data[(data["Age"] >= min_age) & (data["Age"] <= max_age)]

# Display basic dataset info
st.write("### Dataset Overview")
st.write(f"Number of Rows: {filtered_data.shape[0]}")
st.write(f"Number of Columns: {filtered_data.shape[1]}")

# Display a sample of the data
st.write("### Sample Data")
st.write(filtered_data.head())

# Visualize data
st.write("### Data Visualizations")

# Histogram for selected feature
st.write("#### Histogram for Selected Feature")
fig = px.histogram(filtered_data, x=selected_feature, title=f"{selected_feature} Distribution")
st.plotly_chart(fig)

# Scatter plot for two features
st.write("#### Scatter Plot")
x_feature = st.selectbox("X-Axis Feature", data.columns)
y_feature = st.selectbox("Y-Axis Feature", data.columns)
scatter_fig = px.scatter(filtered_data, x=x_feature, y=y_feature, title=f"{x_feature} vs {y_feature}")
st.plotly_chart(scatter_fig)

# Box plot for selected feature
st.write("#### Box Plot")
box_fig = px.box(filtered_data, x="Target", y=selected_feature, title=f"Box Plot of {selected_feature} by Target")
st.plotly_chart(box_fig)

# Correlation matrix heatmap
st.write("#### Correlation Matrix Heatmap")
numeric_data = filtered_data.select_dtypes("number")
correlation_matrix = numeric_data.corr()
fig = go.Figure(data=go.Heatmap(z=correlation_matrix.values, x=correlation_matrix.columns, y=correlation_matrix.index))
st.plotly_chart(fig)
# Classification by Target
st.write("### Customer Classification by Target")
st.write("#### Count of Customers by Target")
target_count = filtered_data["Target"].value_counts()
st.bar_chart(target_count)

st.write("#### Customer Satisfaction by Target")
customer_satisfaction = filtered_data.groupby("Target")["Customer_Satisfaction"].mean()
st.bar_chart(customer_satisfaction)

st.write("#### Purchase Frequency by Target")
purchase_frequency = filtered_data.groupby("Target")["Purchase_Frequency"].mean()
st.bar_chart(purchase_frequency)

# Additional Features
st.write("### Additional Features")

# Age distribution by Target
st.write("#### Age Distribution by Target")
fig = px.violin(filtered_data, x="Target", y="Age", box=True, points="all", title="Age Distribution by Target")
st.plotly_chart(fig)

# Education level distribution
st.write("#### Education Level Distribution")
education_counts = filtered_data["Education"].value_counts()
st.bar_chart(education_counts)

# Credit Score distribution
st.write("#### Credit Score Distribution")
fig = px.histogram(filtered_data, x="Credit_Score", color="Target", barmode="overlay", title="Credit Score Distribution by Target")
st.plotly_chart(fig)

# Loan Approval Amount by Target
st.write("#### Loan Approval Amount by Target")
fig = px.box(filtered_data, x="Target", y="Loan_Approval_Amount", title="Loan Approval Amount by Target")
st.plotly_chart(fig)


st.write("### Customer Segmentation")


# Select features for clustering (e.g., Age, Credit_Score, Purchase_Frequency)
features_for_clustering = ["Age", "Credit_Score", "Purchase_Frequency"]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[features_for_clustering])

# Determine the optimal number of clusters (you can use techniques like the Elbow Method)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Visualize the Elbow Method
st.write("#### Elbow Method to Determine the Optimal Number of Clusters")
fig = px.line(x=list(range(1, 11)), y=inertia, title="Elbow Method")
st.plotly_chart(fig)

# Choose the optimal number of clusters (e.g., 3)
n_clusters = 3

# Perform clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize the clusters
st.write("#### Customer Segmentation Results")
fig = px.scatter_3d(data, x=features_for_clustering[0], y=features_for_clustering[1], z=features_for_clustering[2], color=data['Cluster'], title="Customer Segmentation")
st.plotly_chart(fig)

# Predictive Modeling
st.write("### Predictive Modeling")

# Select features and target variable for the predictive model
X = data[['Age', 'Credit_Score', 'Purchase_Frequency']]
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a predictive model (e.g., Random Forest Classifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write("#### Predictive Model Evaluation")
st.write(f"Accuracy: {accuracy:.2f}")

# Recommendations
st.write("### Recommendations")

# Select a customer to provide recommendations for
customer_index = st.number_input("Enter a customer index:", min_value=0, max_value=data.shape[0]-1, value=0, step=1)
selected_customer = data.iloc[customer_index]

# For simplicity, let's recommend products based on the purchase frequency of the selected customer
recommendations = data[data['Purchase_Frequency'] > selected_customer['Purchase_Frequency']]

st.write(f"#### Recommendations for Customer {customer_index}")
st.write(recommendations)


st.write("### Business Strategy")

# Identify the most profitable customer segment
profitable_segment = data.groupby('Cluster')['Balance'].mean().idxmax()

st.write("#### Most Profitable Customer Segment")
st.write(f"The most profitable customer segment is Cluster {profitable_segment}.")

# Suggest strategies for the most profitable segment
if profitable_segment == 0:
    st.write("#### Strategies for Cluster 0")
    st.write("1. Target this segment with higher credit limit offers to increase card usage.")
    st.write("2. Promote high-value products and services to increase revenue.")
elif profitable_segment == 1:
    st.write("#### Strategies for Cluster 1")
    st.write("1. Enhance customer satisfaction and loyalty through personalized services.")
    st.write("2. Implement referral programs to attract more customers with similar profiles.")
else:
    st.write("#### Strategies for Cluster 2")
    st.write("1. Focus on reducing customer churn to retain high-value customers.")
    st.write("2. Provide incentives to increase online activity and engagement.")
