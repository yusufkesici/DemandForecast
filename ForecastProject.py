import numpy as np
import streamlit as st
import pandas as pd
import sklearn
import joblib
import altair as alt
from statsmodels.tsa.seasonal import seasonal_decompose


st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="chart_with_upwards_trend"
)

def predict_function(month, store, item, train_last_month=3, trainShow=True):
        if month < 10:
            month = f"0{month}"

        if trainShow:
            train_for_func = df.loc[
                             ((df["date"] < "2017-01-01") & (df[f"store_{store}"] == 1) & (df[f"item_{item}"] == 1)),
                             :]

        no_see_pred = df.loc[
                      (((df["date"] >= "2017-01-01") & (df["date"] <= f"2017-{month}-28")) & (
                                  df[f"store_{store}"] == 1) & (
                               df[f"item_{item}"] == 1)), :]
        no_see_pred_X = no_see_pred[cols]
        no_see_predicted_y = model.predict(no_see_pred_X, num_iteration=model.best_iteration)

        Actual_values = pd.DataFrame({"Actual Sales": np.expm1(no_see_pred["sales"])}).set_index(no_see_pred["date"])
        Predicted_values = pd.DataFrame({"Predicted Sales": np.expm1(no_see_predicted_y)}).set_index(
            no_see_pred["date"])

        Report_df = pd.merge(Actual_values, Predicted_values, left_index=True, right_index=True)
        Report_df = Report_df.reset_index()
        Report_df.columns = ["Date", "Actual Sales", "Predicted Sales"]

        Actual_values["Color"] = "Actual Sales"
        Predicted_values["Color"] = "Predicted Sales"

        Actual_values.columns = ["Sales", "Color"]
        Predicted_values.columns = ["Sales", "Color"]

        Predicted_values["Sales"] = np.round(Predicted_values["Sales"])

        concat1 = pd.concat([Actual_values, Predicted_values])

        if trainShow:
            Train_values = pd.DataFrame({"Train Data": np.expm1(train_for_func["sales"])}).set_index(
                train_for_func["date"])[-(30 * train_last_month):]
            Train_values["Color"] = "Train Data"
            Train_values.columns = ["Sales", "Color"]

            all_sales_df = pd.concat([Train_values, concat1]).reset_index()
            all_sales_df.columns = ["Date", "Sales", "Color"]
        else:
            all_sales_df = concat1.reset_index()
            all_sales_df.columns = ["Date", "Sales", "Color"]



  

        return all_sales_df, Report_df




st.title("Store Item Demand Forecasting Application 📚")

option_store = st.selectbox('Select a store number', range(1,11))


option_item = st.selectbox('Select an item number',
range(1,51))

option_month = st.selectbox('How many months of sales forecast would you like to make?',
range(1,13))

Train_Show = True
option_train_last_month = 1
selected_trainShow = st.radio(
    "Would you like to observe the train dataset?",
    ('No', 'Yes'))

if selected_trainShow == 'Yes':
    Train_Show = True

    option_train_last_month = st.selectbox('How many recent months of the train dataset would you like to observe?',
    range(1,13))
else:
    Train_Show = False



month = int(option_month)
store = int(option_store)
item = int(option_item)
trainShow = Train_Show
train_last_month = int(option_train_last_month)



if st.button('Forecast'):

    with st.spinner(f"For Store: {store} and Item: {item}, a {month}-month forecast is being conducted..."):
            
            model = joblib.load("SalesPredictModel.pkl")
            
            df = pd.read_csv("processed_StoreItemData.csv")

            cols = [col for col in df.columns if col not in ['date', 'id', "sales", "year"]]

            train = df.loc[(df["date"] < "2017-01-01"), :]
            data = train[(train[f"store_{1}"] == 1) & (train[f"item_{1}"] == 1)]  

            all_sales_df, Report_df = predict_function(month, store, item, train_last_month=train_last_month, trainShow=trainShow)

            
    # Succes message
    st.success("Process completed!")

    MAE = sklearn.metrics.mean_absolute_error(Report_df["Actual Sales"], Report_df["Predicted Sales"])
    st.subheader(f"Comparing, Mean Absolute Error (MAE) : {round(MAE, 2)}")
    st.dataframe(Report_df, width=800, hide_index=True)
            
    base_chart = alt.Chart(all_sales_df).mark_line().encode(
    x='Date:T',
    y=alt.Y('Sales:Q', scale=alt.Scale(zero=False)),
    color=alt.Color('Color:N', scale=alt.Scale(scheme='category10'))
    ).properties(
    width=800,
    height=500
    )

            
    base_chart = alt.Chart(all_sales_df).mark_line().encode(
    x='Date:T',
    y=alt.Y('Sales:Q', scale=alt.Scale(zero=False)),
    color=alt.Color('Color:N', scale=alt.Scale(scheme='category10'))
    ).properties(
    width=800,
    height=500
    )

            
    zoomable_chart = base_chart.interactive().add_params(
    alt.selection_interval(bind='scales', encodings=['x'])
    )

            
    st.subheader("Result")
    st.altair_chart(zoomable_chart)

            
    result1 = seasonal_decompose(data.set_index('date')["sales"], model='additive', period=365).seasonal
    result2 = seasonal_decompose(data.set_index('date')["sales"], model='additive', period=365).trend

    st.subheader("Seasonality")
    st.line_chart(result1)
    st.subheader("Trend")
    st.line_chart(result2)



else:
    st.write("*************")


    







