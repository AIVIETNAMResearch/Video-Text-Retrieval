from utils.bert_processing import BERTSearch
from utils.ocr_processing import fill_ocr_results

if __name__ == "__main__":
    with open("dict/info_ocr.txt", "r", encoding="utf8") as fi:
        list_ocr_results = list(map(lambda x: x.replace("\n",""), fi.readlines()))

    list_ocr = fill_ocr_results("trương mỹ hoa", list_ocr_results)
    print(list_ocr)