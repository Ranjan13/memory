import logging, gensim
import sys
import os
sys.path.append("../../../i2p/i2pThrift/gen-py")
sys.path.append("../../../i2p/i2pThrift/tools/py")


import Inputs.LSAService as LSA_Service
from I2P.ttypes import *
import ThriftTools

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



class LSAHandler:
    def __init__(self):
        self.load()

    def getCoordinates(self,queries):
        try:
            vec_bow = self.dictionary.doc2bow(queries)
            sparse_vec_lsi = self.lsi[vec_bow] # convert the query to LSI space
            vec_lsi=self.getDenseVec(sparse_vec_lsi)
            print "Input: "," ".join(queries)
            print "LSA coordinates: "
            print(vec_lsi[:5])
            print("\n")
            return vec_lsi
        except:
            print "Error:", sys.exc_info()[0]


    def load(self):
        # load id->word mapping (the dictionary), one of the results of step 2 above
        self.dictionary = gensim.corpora.Dictionary.load_from_text('wiki_lsa_wordids.txt')
        #self.dictionary = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
        #mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
        # load lsi model
        self.lsi = gensim.models.LsiModel.load("model.lsi")
        #self.lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=self.dictionary, num_topics=400)

    def getDenseVec(self,sparseVec,dim=400):
        if len(sparseVec)==dim:
            res=[entry for (_,entry) in sparseVec]
            return res
        else:
            res=[0]*dim
            for (idx,entry) in sparseVec:
                res[idx]=entry
            return res





if __name__=="__main__":
    import socket
    ip_address = socket.gethostbyname(socket.gethostname())
    print ip_address
    #ip_address="172.22.239.97"

    lsa_handler = LSAHandler()
    lsa_server = ThriftTools.ThriftServerThread(12018,LSA_Service,lsa_handler,'LSA Server',ip_address)
    #  server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
    lsa_server.start()

