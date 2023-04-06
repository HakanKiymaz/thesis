import os
import pandas as pd

from pdf_parse import read_pdf, clean_report

COLUMNS = {"breast": ["bcr_patient_barcode",
                      "type",
                      "histological_type",
                      "tumor_status",
                      "ajcc_pathologic_tumor_stage",
                      "new_tumor_event_type"]}


def get_punctuations():
    import string
    punctuations = string.punctuation
    return punctuations


def create_kidney_label(text):
    if "left" in text.lower():
        return "left"
    elif "right" in text.lower():
        return "right"
    else:
        return "None"


def preprocess(text):
    text = text.translate(str.maketrans("", "", get_punctuations()))
    return text


def extract_report_ids(directory: str = "./breast-reports-TCGA/gdc_data/gdc_data"):
    files = {
        "barcode": [],
        "text": []
    }
    for folder in os.listdir(directory):
        fpath = directory + "/" + folder
        if os.path.isdir(fpath):
            report_file = [file for file in os.listdir(fpath) if file.endswith(".pdf")][0]
            report = read_pdf(fpath + "/" + report_file)
            report = clean_report(report)
            files["barcode"].append(report_file.split(".")[0])
            files["text"].append(report)
    reports = pd.DataFrame(files)
    reports.to_excel("Existing_breast_reports.xlsx", index=False)
    return reports


def get_existing_reports():
    try:
        files_ = pd.read_excel("Existing_breast_reports.xlsx")
    except:
        files_ = extract_report_ids()

    spp_tbl = pd.read_excel("./TCGA-CDR-SupplementalTableS1.xlsx", sheet_name="TCGA-CDR",
                            usecols=COLUMNS["breast"])
    spp_tbl = spp_tbl[spp_tbl["type"] == "BRCA"]
    spp_tbl.reset_index(drop=True, inplace=True)

    def merge_(files, sup_table):
        df = files.merge(sup_table, how="inner", left_on="barcode",right_on="bcr_patient_barcode")
        return df

    merged = merge_(files_, spp_tbl)
    merged.drop("bcr_patient_barcode", axis=1, inplace=True)
    merged.drop_duplicates(subset="barcode", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    merged.fillna("None", inplace=True)
    return merged


if __name__ == "__main__":
    in_model_files = get_existing_reports()

    print("done")
