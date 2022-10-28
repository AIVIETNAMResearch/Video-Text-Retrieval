from utils.bert_processing import BERTSearch

if __name__ == "__main__":
    mybert = BERTSearch(dict_bert_search='dict/keyframes_id_bert.json', bin_file='dict/faiss_bert.bin', mode='search')

    text = 'lũ lụt'
    scores, idx_video, infos_query, image_paths = mybert.bert_search(text, k=9)
    print(image_paths)
    print(idx_video)