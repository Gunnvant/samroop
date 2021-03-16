from utils import *
path_parent=r'C:\Users\gunnv\iCloudDrive\Books'
meta=MetaCreation(path_parent)
file_paths=meta.generate_pdf_paths()
reader=ReadFile(file_paths)
gen_dict=GenerateDict(reader)
dictionary=gen_dict.generate_dict()
bow= list(BoWCorpus(dictionary=dictionary))
tfidf=GenerateTfidf(bow).generate_tfidf()
tfidf_corpus=tfidf[bow]
lsi=GenerateLsi(tfidf_corpus,dictionary).generate_lsi()
index=GenerateIndex(lsi,tfidf_corpus).generate_index()