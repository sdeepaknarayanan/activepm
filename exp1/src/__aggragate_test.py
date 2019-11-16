import os
import sys
import pandas as pd


###################################
where = sys.argv[1]
###################################

def get_data(Pfolder):
    '''Updates the gobal full_df as and when function finds dfs'''
    global count
    global full_df
    if not os.path.isdir(Pfolder):
        if 'results_test.csv' in Pfolder:
            count += 1
            if count % 1000 == 0:
                print (count)
            temp = pd.read_csv(Pfolder).reindex(
                columns=full_df.columns,
                copy=False
            )
            full_df = full_df.append(temp, ignore_index=True)
            return None
        else: # we ignore
            return None
    else:
        folders = os.listdir(Pfolder)
        for folder_file in folders:
            get_data(os.path.join(Pfolder, folder_file))
        return None


def get_df_struct(Pfolder):
    if not os.path.isdir(Pfolder):
        if 'results_test.csv' in Pfolder:
            return pd.read_csv(Pfolder)
        else: # we ignore
            return None
    folders = os.listdir(Pfolder)
    for folder_file in folders:
        temp = get_df_struct(os.path.join(Pfolder, folder_file))
        if temp is not None:
            return temp


# runner code
df = get_df_struct(where)
full_df = pd.DataFrame(columns=df.columns)
for c, d in zip(df.columns, df.dtypes):
    full_df[c] = full_df[c].astype(d)

count = 0
get_data(where) # updates global full_df. impure function, I know.
for c, d in zip(df.columns, df.dtypes):
    full_df[c] = full_df[c].astype(d)
full_df.to_csv(os.path.join(where, 'final_test.csv'), index=False)
print(full_df.head())
print(full_df.shape)
