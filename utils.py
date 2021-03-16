'''Reads stream of pdf files from target folder and creates lsi indices'''
import os
import json
from pathlib import Path
import shutil
from tqdm import tqdm
import tika 
tika.initVM()
from tika import parser
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora,models
from gensim import similarities
from multiprocessing import Pool,cpu_count

home = str(Path.home())
meta_folder=".file_org"
meta_name="data_pdf.json"
dict_name='dictionary_dict'
meta_parser_name="data_pdf_parsed.json" #Final json containing files successfully read
tfidf_name='tfidf.model'
lsi_name='lsi.model'
index_name='lsi.index'

class MetaCreation():
    def __init__(self,path_parent):
        self.path_parent=path_parent
        self.path_json=os.path.join(home,meta_folder,meta_name)
    def _generate_meta(self):
        '''Extracts meta information and stores in home directory'''
        dir_names=[]
        file_names=[]
        for root,dirs,files in tqdm(os.walk(self.path_parent)):
            for name in dirs:
                dir_names.append(os.path.join(root, name))
            for name in files:
                file_names.append(os.path.join(root, name))
        file_names=[file_name for file_name in file_names if file_name.endswith(".pdf")]
        result={'file_names':file_names,'dir_names':dir_names}
        path_meta_folder=os.path.join(home,meta_folder)
        if not os.path.isdir(path_meta_folder):
            os.mkdir(path_meta_folder)
        with open(self.path_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        return result
    def generate_pdf_paths(self):
        if os.path.isfile(self.path_json):
            with open(self.path_json,encoding='utf-8') as f:
                result=json.loads(f.read())
        else:
            result=self._generate_meta()
        file_paths=result.get('file_names')
        return file_paths
class ReadFile():
    def __init__(self,file_paths):
        self.file_paths=file_paths
        self.paths_read=[]
    def __len__(self):
        return len(self.file_paths)
    def __iter__(self):
        for file_path in tqdm(self.file_paths):
            parsed=parser.from_file(file_path)
            text=parsed['content']
            if text is not None:
                self.paths_read.append(file_path)
                text=text.replace("\r\n","")
                yield (token for token in simple_preprocess(text) if token not in STOPWORDS)

class GenerateDict():
    def __init__(self,reader_obj):
        self.reader_obj=reader_obj
    def generate_dict(self):
        path_dict=os.path.join(home,meta_folder,dict_name)
        if os.path.isfile(path_dict):
            print(f"Reading cached dictionary from: {path_dict}" )
            dictionary=corpora.Dictionary()
            dictionary=dictionary.load(path_dict)
        else:
            print(f"No cache found. Generating dictionary, this will take a while")
            dictionary=corpora.Dictionary(self.reader_obj)
            print(f"Saving dictionary at {path_dict}")
            dictionary.save(path_dict)
            paths_read=self.reader_obj.paths_read
            paths_all=self.reader_obj.file_paths
            perc_read=round(len(paths_read)/len(paths_all),2)
            print(f'{perc_read*100} % paths successfully parsed')
            result={'paths_successfully_parsed':paths_read}
            path_results=os.path.join(home,meta_folder,meta_parser_name)
            with open(path_results,'w',encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
        return dictionary

class BoWCorpus():
    def __init__(self,dictionary):
        self.dict=dictionary
        path_results=os.path.join(home,meta_folder,meta_parser_name)
        self.file_paths=json.loads(open(path_results,encoding='utf-8').read())['paths_successfully_parsed']
    def __len__(self):
        return self.dict.num_docs
    
    def __iter__(self):
        for path in tqdm(self.file_paths):
            parsed=parser.from_file(path)
            text=parsed['content']
            if text is not None:
                tokenized_list = [token for token in simple_preprocess(text.replace("\r\n","")) if token not in STOPWORDS]
                bow = self.dict.doc2bow(tokenized_list, allow_update=True)               
                yield bow

class GenerateTfidf():
    def __init__(self,bow):
        self.bow=bow
    def generate_tfidf(self):
        path_tfidf=os.path.join(home,meta_folder,tfidf_name)
        if os.path.isfile(path_tfidf):
            print(f"Tfidf model cache found at {path_tfidf}, loading existing tfidf model")
            tfidf=models.TfidfModel()
            tfidf=tfidf.load(path_tfidf)
        else:
            print(f"Fitting tfidf model, this will take a while")
            tfidf = models.TfidfModel(self.bow, smartirs='ntc')
            print(f"Saving tfidf model at {path_tfidf}")
            tfidf.save(path_tfidf)
        return tfidf
class GenerateLsi():
    def __init__(self,tfidf_corpus,dictionary):
        self.tfidf_corpus=tfidf_corpus
        self.dictionary=dictionary
        print('Generating LSI model')
    def generate_lsi(self):
        path_lsi=os.path.join(home,meta_folder,lsi_name)
        if os.path.isfile(path_lsi):
            print(f"Found cache at {path_lsi}, loading model")
            lsi_model=models.LsiModel.load(path_lsi)
        else:
            print(f'Generating lsi model, this will take a while')
            lsi_model = models.LsiModel(self.tfidf_corpus, id2word=self.dictionary, num_topics=150)
            print("Saving lsi model")
            lsi_model.save(path_lsi)
        return lsi_model
class GenerateIndex():
    def __init__(self,lsi,tfidf_corpus):
        self.lsi=lsi
        self.tfidf_corpus=tfidf_corpus
    def generate_index(self):
        path_index=os.path.join(home,meta_folder,index_name)
        if os.path.isfile(path_index):
            print(f"Index cache found at {path_index}, loading cache")
            index = similarities.MatrixSimilarity.load(path_index)
        else:
            print(f"No index cache found, generating new index, this will take a while")
            index = similarities.MatrixSimilarity(self.lsi[self.tfidf_corpus])
            print("Saving Index..")
            index.save(path_index)
        return index


class Refresh():
    def __init__(self):
        print("This will erase all the cache and recompute everything once again. To destroy the cache call refresh() method")
    def refresh(self):
        shutil.rmtree(os.path.join(home,meta_folder))
