import pandas as pd

csv_file1 = "restatements.csv"
csv_data1 = pd.read_csv(csv_file1, low_memory = False)
csv_df1 = pd.DataFrame(csv_data1)

csv_file2 = "datei.csv"
csv_data2 = pd.read_csv(csv_file2, low_memory = False)
csv_df2 = pd.DataFrame(csv_data2)

csv_fusion = pd.concat([csv_df1,csv_df2], axis=1)

csv_fusion.to_csv('testdateimitlabels.csv',index=False)