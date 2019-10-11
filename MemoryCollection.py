__author__ = 'Zhang Juzheng'

import os
import cPickle as pickle
import time
import re

class MemoryCollection:
    def __init__(self,userName):
        self.getUserName(userName)


    def getUserName(self,userName):
        if userName!="":
            self.user=userName
        else:
            self.user="Unknown"

        if self.user=="Unknown":
            basepath=os.getcwd()+"\\"+"UnknownUser"
            self.path=basepath
            if not os.path.exists(basepath):
                os.makedirs(basepath)
        else:
            basepath=os.getcwd()+"\\"+"Episodes"
            if not os.path.exists(basepath):
                os.makedirs(basepath)
            self.path=basepath+"\\"+self.user
            if not os.path.exists(self.path):
                os.makedirs(self.path)

    def saveEpisode(self,episodes):
        filename=time.strftime("%c")+".pickle"
        filename=re.sub(r'''[\:/]''',"-",filename)
        filename=self.path+"\\"+filename
        file=open(filename,'wb')
        pickle.dump(episodes,file)
        file.close()
        print "save episode successfully!!"
