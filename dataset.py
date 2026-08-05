import pandas as pd

data = {
    "Customer_ID":[101,102,103,104,105,105,106,107,108,109],
    "Age":[25,30,-5,150,40,40,29,35,22,55],
    "Salary":[50000,None,45000,2000000,65000,65000,None,72000,38000,None],
    "City":["Bangalore","Banglore","Delhi","Mumbia","Bangaluru","Bangaluru","Mumbai","Delhii","Bangalore","Delhi"],
    "Gender":["Male","Female","Male","Female","Male","Male","Female","Male","Female","Male"],
    "Purchase_Date":["12/05/2024","2024-06-10","10-May-24","2024/07/15","15/08/2024","15/08/2024","2024-08-20","01-Jan-24","2024-03-11","11/11/2024"],
    "Currency":["INR","INR","USD","USD","INR","INR","USD","INR","INR","USD"],
    "Spending":[7000,12000,3000,40000,18000,18000,9000,25000,2000,15000],
    "Purchase_Frequency":[10,8,5,20,12,12,6,15,3,9]
}

df = pd.DataFrame(data)

df.to_csv("ecommerce.csv", index=False)

print("ecommerce.csv created successfully!")