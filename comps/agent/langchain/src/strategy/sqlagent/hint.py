import pandas as pd
import os
import glob

def generate_column_descriptions(db_name):
    output = []
    decriptions_only = []
    working_dir = os.getenv("WORKDIR")
    assert working_dir is not None, "WORKDIR environment variable is not set."
    DESCRIPTION_FOLDER=os.path.join(working_dir, f"TAG-Bench/dev_folder/dev_databases/{db_name}/database_description/")
    table_files = glob.glob(os.path.join(DESCRIPTION_FOLDER, "*.csv"))
    for table_file in table_files:
        table_name = os.path.basename(table_file).split(".")[0]
        print("Table name: ", table_name)
        df = pd.read_csv(table_file)
        for _, row in df.iterrows():
            col_name = row["original_column_name"]
            if not pd.isnull(row["value_description"]):
                description = str(row["value_description"]) 
                if description.lower() in col_name.lower():
                    print("Description {} is same as column name {}".format(description, col_name))
                    pass
                else:            
                    description = description.replace("\n", " ")
                    description = " ".join(description.split())
                    output.append(f"{table_name}.{col_name}: {description}")
                    # decriptions_only.append(description)
                    decriptions_only.append(f"{col_name}: {description}")
    return output, decriptions_only

def read_hints(hints_file):
    """
    hints_file: csv with columns: table_name, column_name, description    
    """
    hints_df = pd.read_csv(hints_file)
    cols_descriptions = []
    values_descriptions = []
    for _, row in hints_df.iterrows():
        table_name = row["table_name"]
        col_name = row["column_name"]
        description = row["description"]
        if not pd.isnull(description):
            cols_descriptions.append(f"{table_name}.{col_name}: {description}")
            values_descriptions.append(f"{col_name}: {description}")
    return cols_descriptions, values_descriptions


def sort_list(list1, list2):
    import numpy as np
    # Use numpy's argsort function to get the indices that would sort the second list
    idx = np.argsort(list2)# ascending order
    return np.array(list1)[idx].tolist()[::-1], np.array(list2)[idx].tolist()[::-1]# descending order

def get_topk_cols(topk, cols_descriptions, similarities):
    sorted_cols, similarities = sort_list(cols_descriptions, similarities)
    top_k_cols = sorted_cols[:topk]
    output = []
    for col, sim in zip(top_k_cols, similarities[:topk]):
        # print(f"{col}: {sim}")
        if sim > 0.5: 
            output.append(col)
    return output


def pick_hints(query, model, column_embeddings, complete_descriptions, topk=5):
    # use similarity to get the topk columns
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = model.similarity(query_embedding, column_embeddings).flatten()

    topk_cols_descriptions = get_topk_cols(topk, complete_descriptions, similarities)

    hint = ""
    for col in topk_cols_descriptions:
        hint += (col +'\n')
    return hint