from PyPDF2 import PdfReader
import re

def read_pdf(path):
    reader = PdfReader(path)
    number_of_pages = len(reader.pages)
    text = ""
    for pg in range(number_of_pages):
        page = reader.pages[pg]
        text += "\n" + page.extract_text()
    text = text.strip()
    return text


def clean_report(report):
    clean1 = re.sub(r'[^\w\s]', ' ', report)
    clean2 = re.sub('\n', ' ', clean1)
    clean3 = re.sub(' +', ' ', clean2)

    return clean3


if __name__ == "__main__":
    from helpers import get_existing_reports
    # in_model_files = get_existing_reports()
    pdf_path = "./breast-reports-TCGA/gdc_data/gdc_data/0a3ba494-24b8-4bda-972c-274aa7f891dd/TCGA-A2-A1FZ.F93E326D-F4E0-4000-8A67-AA19C4D3637A.pdf"
    pdf_path2 = "./breast-reports-TCGA/gdc_data/gdc_data/0a3ec7da-3208-42ec-b8c2-44822fbedbae/TCGA-AO-A0JI.468411B5-1719-46B4-835D-E469D6AEFF12.pdf"

    report = read_pdf(pdf_path)

    cleaned_report = clean_report(report)


    print("done")
