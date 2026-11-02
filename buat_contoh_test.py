import pandas as pd

FITUR = ["Age", "Chronic_Pain_Level", "BMI",
         "Menstrual_Irregularity", "Hormone_Level_Abnormality", "Infertility"]

df = pd.read_csv("dataset_endometriosis-2.csv")
contoh = df[df["bagian"] == "test"].sample(10, random_state=3)

contoh[FITUR].round(2).to_csv("contoh_data_baru.csv", index=False)
contoh[FITUR + ["status"]].round(2).to_csv("contoh_data_baru_kunci.csv", index=False)
print(contoh[FITUR + ["status"]])