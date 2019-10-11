__author__ = 'Ranjan Satapathy'



import sys
import time
import os 
sys.path.append("../../i2p/i2pThrift/gen-py")
sys.path.append("../../i2p/i2pThrift/tools/py")
from EpisodicMemory import EpisodicMemory
import cPickle as pickle
import Inputs.EMNadineService as EM_Service
import Inputs.RelationExtractionService as Relation_Service
from Inputs.ttypes import *
import Inputs.constants
import ThriftTools
import Definition
from textblob import TextBlob
from findTime import findTime
import BasicOperation as opt
from MemoryCollection import MemoryCollection
from Knowledge import *
from randomFunc import randomFunc
from datetime import datetime
from copy import copy,deepcopy

class EMNadineHandler:
    def __init__(self):
        self.EM=EpisodicMemory()
        self.EM.loadMemory()
        self.tempMem=EpisodicMemory()
        self.episode=None
        self.lastRetrievedEventIdx=None
        self.timeAnalyser=findTime()
        #self.knowledge=Knowledge(relation_client)
        self.rdm=randomFunc()
        self.knownUsers=self.getKnownUsers()


    def getCurrentEventIdx(self):
        return self.getLastMaxEventIdx()

    def getLastMaxEventIdx(self):
        if len(self.EM.episodes)>0:
            lastEp=self.EM.episodes[-1]
            res=lastEp.maxEvIdx
            return res
        else:
            return -1

    def getUser(self,event):
        user=event.getAttrLabelType("user")
        return user

    def getAttr(self,attr_class,event):
        res=event.getAttrLabelType(attr_class)
        return res


    def isSentDuplication(self,sent1,sent2):
        s1=TextBlob(sent1)
        s2=TextBlob(sent2)
        w1=s1.words
        w2=s2.words
        if len(w1)>=len(w2):
            for word in w2:
                if word not in w1:
                    return False
        else:
            for word in w1:
                if word not in w2:
                    return False
        return True

    def isDuplicateFromList(self,sentList,cue):
        for sent in sentList:
            if self.isSentDuplication(sent,cue):
                return True
        return False

    def checkUpdatingTopics(self,cue,events): # both cue and events contain the same topic
        topics=["hobby","nationality"]
        return self.checkTopics(topics,cue,events)

    def checkFixedTopics(self,cue,events): # both cue and events contain the same topic
        topics=["research topic","current work"]
        return self.checkTopics(topics,cue,events)

    def checkTopics(self,topics,cue,events):
        res=[]
        for topic in topics:
            if topic in cue.sentence:
                for event in events:
                    if topic in event.sentence:
                        res.append(event)
                if len(res)>0:
                    return True, res
        return False, events

    # def getRandomRelatedEvent(self,events,n_max,cue,threshold,delta=0.1,tempFlag=False):
    #     flag1,events=self.checkUpdatingTopics(cue,events)
    #     if flag1:
    #         res1=self.getRelatedEvent(events,n_max,tempFlag,reorder=True)
    #         return res1
    #     flag2,events=self.checkFixedTopics(cue,events)
    #     if flag2:
    #         res2=self.getRelatedEvent(events,n_max,tempFlag,reorder=False)
    #         return res2
    #
    #     ########### Normal Retrieval #####################
    #     List,similarList=[],[]
    #     prob,similarProb=[],[]
    #     similar_th=max(n_max[0]-delta,0)
    #     for i in range(len(n_max)):
    #         if n_max[i]>=threshold:
    #             try:
    #                 if n_max[i]>=similar_th: # qualified and near to the best
    #                     similarList.append(events[i])
    #                     similarProb.append(n_max[i])
    #                 else: # qualified
    #                     List.append(events[i])
    #                     prob.append(n_max[i])
    #             except IndexError:
    #                 print "Error Events: ",events
    #                 print "Error n_max: ",n_max
    #         else:
    #             break
    #
    #     res1=self.getRelatedEvent(similarList,similarProb,tempFlag,reorder=True)
    #     if res1!=None:
    #         return res1
    #     res2=self.getRelatedEvent(List,prob,tempFlag,reorder=True)
    #     return res2

    def getRandomRelatedEvent(self,events,n_max,cue,threshold,delta=0.1,tempFlag=False):
        flag1,events=self.checkUpdatingTopics(cue,events)
        if flag1:
            res1=self.getRelatedEvent(events,n_max,tempFlag,reorder=True)
            return res1
        flag2,events=self.checkFixedTopics(cue,events)
        if flag2:
            res2=self.getRelatedEvent(events,n_max,tempFlag,reorder=False)
            return res2

        ########### Normal Retrieval #####################
        List,similarList=[],[]
        prob,similarProb=[],[]
        for i in range(len(n_max)):
            if n_max[i]>=threshold:
                try:
                    List.append(events[i])
                    prob.append(n_max[i])
                except IndexError:
                    print "Error Events: ",events
                    print "Error n_max: ",n_max
            else:
                break


        res2=self.getRelatedEvent(List,prob,tempFlag,reorder=True)
        return res2

    def getRelatedEvent(self,eventList,prob,tempFlag,reorder=True):
        if len(eventList)==0:
            return None
        List=deepcopy(eventList)
        if reorder:
            List=self.EM.reorderEvents(List)
        while len(List)>0:
            #event,sim=self.rdm.randomChoose_w_prob(List,prob)
            event=List[0]
            if event==None or event.sentence.startswith("let"):
                List.remove(event)
                continue
            # self.lastRetrievedEventIdx=event.index
            # Add the selected episode to STM
            if tempFlag:
                epidx=self.tempMem.getEpisodeIdx(event.index)
                self.tempMem.STM.encode(epidx)
            else:
                epidx=self.EM.getEpisodeIdx(event.index)
                self.EM.STM.encode(epidx)
            ##################
            attr=event.getAttrLabelType("user")
            who=attr.split("=")[1]
            #sim=prob[eventList.index(event)]
            if who==None: who="Unknown"
            res=[event.sentence,who,"1"]
            print res
            return res
        return None


    # stm_bool transfer between searching in STM or LTM
    def retrieveAnswer(self,inputs,candidateIdx,stm_bool=False,curFlag=False, th=0.4,mustFlag=False):
        # curFlag refers to if consider the latest event of the current episode
        cue=self.EM.processEvent.buildEvent(inputs)
        if cue==None:
            return None

        if stm_bool:
            # only consider STM index in STM searching
            candidateIdx=list(set(candidateIdx).intersection(set(self.EM.STM.getIndex())))
        else:
            #don't consider STM index in LTM searching
            candidateIdx=list(set(candidateIdx).difference(set(self.EM.STM.getIndex())))

        if candidateIdx==None or candidateIdx==[]:
            return None
        emRes=self.EMRetrieval(cue,candidateIdx,curFlag,th,mustFlag=mustFlag)
        return emRes




    def EMRetrieval(self,cue,candidateIdx,curFlag,th,mustFlag=False):
        # for consider exact match
        (exact_events,exact_n_max)=self.EM.getExactSimilarEvent(cue,candidateIdx,curFlag=curFlag)
        if exact_events:
            res=self.getRandomRelatedEvent(exact_events,exact_n_max,cue,th)
            if res!=None:
                return res

        # for consider exact question match
        (exact_questions,exact_ques_max)=self.EM.getRelatedQuestion(cue,candidateIdx,curFlag=curFlag)
        if exact_questions:
            res=self.getRandomRelatedEvent(exact_questions,exact_ques_max,cue,th)
            if res!=None:
                return res


        # else, consider similar
        (events,n_max)=self.EM.getMostSimilarEventOverSubset(cue,candidateIdx,curFlag=curFlag)
        if events:
            res=self.getRandomRelatedEvent(events,n_max,cue,threshold=0.9)
            if res!=None:
                return res

        # for consider similar question match
        (questions,ques_max)=self.EM.getRelatedQuestion(cue,candidateIdx,exactFlag=False,curFlag=curFlag)
        if questions:
            res=self.getRandomRelatedEvent(questions,ques_max,cue,threshold=0.9)
            if res!=None:
                return res

        if mustFlag:
            # mustFlag indicate if this retrieval must be done if possible
            # if no candidate, just retrieve the most related event in current episode,
            # sharing all attributes with the retrieval cue.
            # only for online retrieval for the latest event
            (exact_all_events,exact_n_all_max)=self.EM.getExactSimilarEvent(cue,candidateIdx,curFlag=curFlag,allFlag=True)
            if exact_all_events!=None and len(exact_all_events)>0:
                curEpEvIdx=self.episode.getEventIndex()
                for event in exact_all_events:
                    if event.index in curEpEvIdx:
                        res=self.getRandomRelatedEvent([event],[1.0],cue,th)
                        if res!=None:
                            return res
        return None

    def retrieveTempAnswer(self,inputs,th):
        cue=self.tempMem.processEvent.buildEvent(inputs)
        if cue==None:
            return None
        # for consider exact match
        (exact_events,exact_n_max)=self.tempMem.getExactSimilarEvent(cue,curFlag=False)
        if exact_events:
            res=self.getRandomRelatedEvent(exact_events,exact_n_max,cue,th,tempFlag=True)
            if res!=None:
                return res

        # for consider exact question match
        (exact_questions,exact_ques_max)=self.tempMem.getRelatedQuestion(cue,curFlag=False)
        if exact_questions:
            res=self.getRandomRelatedEvent(exact_questions,exact_ques_max,cue,th,tempFlag=True)
            if res!=None:
                return res


        # else, consider similar
        (events,n_max)=self.tempMem.getMostSimilarEventOverSubset(cue,curFlag=False)
        if events:
            res=self.getRandomRelatedEvent(events,n_max,cue,threshold=0.9,tempFlag=True)
            if res!=None:
                return res

        # for consider similar question match
        (questions,ques_max)=self.tempMem.getRelatedQuestion(cue,exactFlag=False,curFlag=False)
        if questions:
            res=self.getRandomRelatedEvent(questions,ques_max,cue,threshold=0.9,tempFlag=True)
            if res!=None:
                return res

        # (exact_all_events,exact_n_all_max)=self.tempMem.getExactSimilarEvent(cue,curFlag=False,allFlag=True)
        # if exact_all_events!=None and len(exact_all_events)>0:
        #     for event in exact_all_events:
        #         res=self.getRandomRelatedEvent([event],[1.0],cue,th,tempFlag=True)
        #         if res!=None:
        #             return res
        return None



############################################################
######################### Service ##########################
############################################################


    def getConceptNetAnswer(self,sent):
        reply=self.EM.getConceptNetAnswer(sent)
        if reply==None:
            reply="None"
        return reply


    def getKnowledge(self,sentence,user):
        return []
        # results=self.knowledge.getQuery(sentence,user)
        # if results==None:
        #     return []
        # else:
        #     res=[]
        #     (sent,n_max)=opt.find_n_max_dict(results,3)
        #     res.append(sent[0])
        #     for i in range(1,len(n_max)):
        #         if not self.isDuplicateFromList(res,sent[i]):
        #                 res.append(sent[i])
        #         else:
        #             break
        #     return res



    def getSimilarEvent(self,inputs,date,user):
        try:
            print "\n search for similar event: ",inputs
            if user.split("=")[1] not in self.knownUsers:
                tempRes=self.retrieveTempAnswer(inputs,th=0.6)
                if tempRes!=None:
                    return tempRes
                else:
                    return []

            dateIdx=None
            userIdx=None
            if date!=None:
                if date in self.EM.timeIndex.keys():
                    dateIdx=self.EM.timeIndex[date]
                else:
                    print "Error: invalid date"
                    return []

            if user!=None:
                if user in self.EM.userIndex.keys():
                    userIdx=self.EM.userIndex[user]
                else:
                    print "Error: invalid user ",user
                    return []

            candidateIdx=None
            if dateIdx==None and userIdx!=None:
                candidateIdx=userIdx
            elif dateIdx!=None and userIdx==None:
                candidateIdx=dateIdx
            elif dateIdx!=None and userIdx!=None:
                candidateIdx=list(set(userIdx).intersection(set(dateIdx)))
                if len(candidateIdx)==0:
                    print "Error: invalid user or date"
                    return []
            else:
                candidateIdx=range(len(self.EM.episodes))

            #self.EM.STM.encode(self.episode.index)
            stmBool=True
            res=self.retrieveAnswer(inputs,candidateIdx,stmBool,th=0.85)
            if res==None:
                stmBool=not stmBool
                res=self.retrieveAnswer(inputs,candidateIdx,stmBool,th=0.7)

            if res!=None:
                # print res
                return res
            print "Error: no satisfied memory, try to decrease the threshold"
            return []
        except:
            print sys.exc_info()[0]
            return []



    def getPeerEvent(self, inputs, date, user):
        try:
            print "\n search for events of other person: ",inputs
            userIdx=[]
            if user.split("=")[1] not in self.knownUsers:
                userIdx=range(len(self.EM.episodes))
            elif user!=None:
                if user in self.EM.userIndex.keys():
                    not_idx=self.EM.userIndex[user]
                    for i in range(len(self.EM.episodes)):
                        if i not in not_idx:
                            userIdx.append(i)
                else:
                    userIdx=range(len(self.EM.episodes))

            stmBool=True
            curFlag=True #The current episode has nothing to do with searching for others
            res=self.retrieveAnswer(inputs,userIdx,stmBool,curFlag,th=0.85)
            if res==None:
                stmBool=not stmBool
                res=self.retrieveAnswer(inputs,userIdx,stmBool,curFlag,th=0.7) # LTM searching
            if res!=None:
                print res
                return res
            print "Error: no satisfied memory, try to decrease the threshold"
            return []
        except:
            print sys.exc_info()[0]
            return []

    def getGreetings(self, user):
        try:
            print "Greeting for "+user
            userIdx=None
            if user!=None:
                if user in self.EM.userIndex.keys():
                    userIdx=self.EM.userIndex[user]
                else:
                    print "Error: invalid user"
                    return []

            userIdx.reverse()
            res=[]
            work_ans=self.getSimilarEvent(["sentence=current work"],None,user)
            if len(work_ans)>0:
                res.append(work_ans[0])
            nationality_ans=self.getSimilarEvent(["sentence=nationality"],None,user)
            if len(nationality_ans)>0:
                res.append(nationality_ans[0])
            # research_ans=self.getSimilarEvent(["sentence=research topic"],None,user)
            # if len(research_ans)>0:
            #     res.append(research_ans[0])
            if len(res)>0:
                return res
            else:
                for epIdx in userIdx:
                    ans=self.EM.getAllQuestion([epIdx])
                    if ans==None:
                        continue
                    if work_ans in ans:
                        ans.remove(work_ans)
                    res.extend(ans)
                    if len(res)>=3:
                        return res[:3]
                if len(res)>0:
                    return res
            print "Error: no satisfied memory"
            return []
        except:
            print sys.exc_info()
            print user
            return []

    def getKnownUsers(self):
        try:
            users=self.EM.userIndex.keys()
            res=[user.split("=")[1] for user in users]
            return res
        except:
            print sys.exc_info()[0]
            return []

    # def getLastDate(self, user):
    #     if user in self.EM.userIndex.keys():
    #         epIdx=self.EM.userIndex[user][0]
    #         numDate=len(self.EM.timeIndex)
    #         for i in range(numDate):
    #             j=numDate-1-i
    #             date=self.EM.timeIndex.keys()[j]
    #             epList=self.EM.timeIndex[date]
    #             if epIdx in epList:
    #                 return date
    #     else:
    #         return "None"

    def getLastDate(self, user):
        try:
            if user in self.EM.userIndex.keys():
                lastDateStr=self.EM.episodes[self.EM.userIndex[user][0]].date
                lastDate=datetime.strptime(lastDateStr,"%d-%b-%Y")
                for epIdx in self.EM.userIndex[user][1:]:
                    dateStr=self.EM.episodes[epIdx].date
                    date=datetime.strptime(dateStr,"%d-%b-%Y")
                    if date>lastDate:
                        lastDate=date
                        lastDateStr=dateStr
                return lastDateStr

            else:
                return "None"
        except:
            print sys.exc_info()[0]
            return "None"

    def updateUser(self, user):
        try:
            user=user.capitalize()
            print "Update User: ",user
            self.episode.user=user
            for i in range(len(self.episode.content)):
                event=self.episode.content[i]
                for j in range(len(event.content)):
                    attr=event.content[j]
                    if attr.type=="subject":
                        if attr.value!="subject=Robot":
                            attr.value="subject="+user
                    elif attr.type=="user":
                        if attr.value!="user=Robot":
                            attr.value="user="+user
        except:
            print sys.exc_info()[0]


    def isSeenUser(self, user, date=None):
        try:
            if user not in self.EM.userIndex.keys():
                return []
            if date==None:
                date=self.timeAnalyser.getDate(self.timeAnalyser.currentTime)
            if date not in self.EM.timeIndex.keys():
                return []
            userIdx=self.EM.userIndex[user]
            dateIdx=self.EM.timeIndex[date]
            common=set(userIdx) & set(dateIdx)
            if len(common)==0:
                return []
            res=[]
            for idx in common:
                episode=self.EM.episodes[idx]
                socialTime=self.getAttr("socialTime",episode.content[0])
                res.append(socialTime.split("=")[1].lower())
            return res
        except:
            print sys.exc_info()[0]


    def getRecentUsers(self):
        try:
            weekDates=self.timeAnalyser.getWeekDates()
            res={}
            valid_dates=self.EM.timeIndex.keys()
            valid_dates.reverse()
            for date in weekDates:
                if date in valid_dates:
                    res[date]=[]
                    dateIdx=self.EM.timeIndex[date]
                    for idx in dateIdx:
                        user=self.getAttr("user",self.EM.episodes[idx].content[0])
                        res[date].append(user.split("=")[1])
            return res
        except:
            print sys.exc_info()[0]
            return {}

    def getUsersDate(self, date):
        try:
            if date not in self.EM.timeIndex:
                return []
            res=[]
            dateIdx=self.EM.timeIndex[date]
            for idx in dateIdx:
                user=self.getAttr("user",self.EM.episodes[idx].content[0])
                res.append(user.split("=")[1])
            return res
        except:
            print sys.exc_info()[0]
            return []

    def isDuplicateQuestion(self, user, inputs):
        if user==None:
            return False

        if user in self.EM.userIndex.keys():
            userIdx=self.EM.userIndex[user]
        else:
            print "Error: invalid user"
            return False

        questionIdx=self.EM.questionIndex
        if userIdx:
            exact_events=self.EM.getExactEvent(inputs,userIdx)
        else:
            return False
        # for consider exact match
        if exact_events:
            for event in exact_events:
                if event.question:
                    return True
        return False

    def getSimilarEpisode (self,inputs,date,user):
        pass

    def getLastEvent(self,inputs,date,user):
        if len(inputs)==0:
            episode=self.EM.retrieveRightEpisode(self.lastRetrievedEventIdx)
            lastEp=self.EM.episodes[episode.index-1]
            lowerIdx=lastEp.maxEvIdx
            if self.lastRetrievedEventIdx==lowerIdx+1:
                print "Error: it is the first event of episode"
                return []
            else:
                res=episode[self.lastRetrievedEventIdx-lowerIdx-2]
                return res.getFlatAttr()
        else:
            self.getSimilarEvent(inputs,date,user)
            return self.getLastEvent([],None,None)

    def getNextEvent(self, inputs,date,user):
        if len(inputs)==0:
            episode=self.EM.retrieveRightEpisode(self.lastRetrievedEventIdx)
            lastEp=self.EM.episodes[episode.index-1]
            lowerIdx=lastEp.maxEvIdx
            upperIdx=episode.maxEvIdx
            if self.lastRetrievedEventIdx==upperIdx:
                print "Error: it is the last event of episode"
                return []
            else:
                res=episode[self.lastRetrievedEventIdx-lowerIdx]
                return res.getFlatAttr()
        else:
            self.getSimilarEvent(inputs,date,user)
            return self.getLastEvent([],None,None)

    def getEpisodeSummery(self,date,user):
        return []
        # if user not in self.EM.userIndex.keys():
        #     return []
        # userIdx=self.EM.userIndex[user]
        # if date!=None and date not in self.EM.timeIndex.keys():
        #     return []
        # elif date==None:
        #     common=[userIdx[-1]]
        # else:
        #     dateIdx=self.EM.timeIndex[date]
        #     common=set(userIdx) & set(dateIdx)
        # if len(common)==0:
        #     return []
        # noun_dict={}
        # idx=common[-1]
        # episode=self.EM.episodes[idx]
        # for event in episode.content:
        #     sentence=event.sentence
        #     nouns=self.EM.processEvent.getNouns(sentence)
        #     if nouns!=None:
        #         for word in nouns:
        #             if word not in noun_dict.keys():
        #                 noun_dict[word]=0
        #             noun_dict[word]+=1
        # (res,_)=opt.find_n_max_dict(noun_dict,3)
        # return res


    def getRelatedAttribute(self,inputs,attrType,num):
        pass

    def getRecentPlan(self, inputs,user):
        event=self.EM.userPlan[user].pop()
        return event.getFlatAttr()

    def storeKnowledge(self,inputs):
        # event=self.EM.processEvent.buildEvent(inputs)
        # if event!=None:
        #     self.knowledge.addKnowledge(event)
        print "Add Knowledge"

    def startNewEpisode(self,user):
        try:
            print "Start New Episode!!"
            if "=" in user:
                user=user.split("=")[1]
            if user not in self.knownUsers:
                self.tempMem.updateEpisodes(Definition.Episode())
                self.episode=self.tempMem.episodes[-1]
                self.episode.index=0
                self.episode.maxEvIdx=-1
                self.tempMem.STM.addFixedIndex(self.episode.index)
            else:
                self.EM.updateEpisodes(Definition.Episode())
                self.episode=self.EM.episodes[-1]
                self.episode.index=len(self.EM.episodes)-1 # Since we add an empty episode into the repository
                self.episode.maxEvIdx=self.EM.episodes[-2].maxEvIdx #mark the largest event id
                self.EM.STM.addFixedIndex(self.episode.index)
        except:
            print sys.exc_info()

    def configureEpisode(self,user,date):
        try:
            print "Configure New Episode!!"
            self.episode.date=date
            self.episode.latestDate=date
            self.episode.user=user
            if user not in self.knownUsers:
                self.tempMem.indexingNewEpisode(self.episode)
            else:
                self.EM.indexingNewEpisode(self.episode)
        except:
            print sys.exc_info()

    def addEpisodeEvIndex(self):
        self.episode.maxEvIdx+=1



    def storeEvent(self,inputs,End,date):
        try:
            if End:
                if self.episode!=None and len(self.episode.content)>0:
                    user=copy(self.episode.user)
                    self.closeMemory(user,None,True)
                    self.episode=None
                    #return []
            else:
                event=self.EM.processEvent.buildEvent(inputs)
                if event==None:
                    print "Discards trivial event"
                else:
                    user=self.getUser(event)
                    user=user.split("=")[1]

                    if self.episode==None:
                        # update for new episode
                        self.startNewEpisode(user)
                        self.configureEpisode(user,date)

                    if user not in self.knownUsers: # unknown user
                        event.index=len(self.episode.content)
                        self.tempMem.indexingNewEvent(event)
                    else: # known user
                        event.index=self.getCurrentEventIdx()+1
                        self.EM.indexingNewEvent(event)
                    self.episode.content.append(event)
                    self.addEpisodeEvIndex()
                    print "Store Event"

        except:
            print "Unexpected error:", sys.exc_info()[0]


    def openMemory(self, user, date):
        pass

    def closeMemory(self, user, date, save):
        try:
            if save:
                if len(self.episode.content)>0:
                    print "Store Episode for User: ",user
                    memory=MemoryCollection(user)
                    memory.saveEpisode(self.episode)
                if user not in self.knownUsers and user!="Unknown":
                    # for new registered user, add to EM
                    print "Load Episode for New User: ",user
                    self.EM.loadEpisode(deepcopy(self.episode),None)
                    self.knownUsers=self.getKnownUsers()
                if len(self.tempMem.episodes)>0:
                    self.tempMem._build()
        except:
            print sys.exc_info()

            #self.knowledge.saveKnowledge()
            #self.EM.saveMemory()

    def deleteMemory(self, user, date):
        pass


if __name__=="__main__":
    import socket

    ip_address = socket.gethostbyname(socket.gethostname())
    print ip_address
    #ip_address="172.22.239.97"
    # relation_client = ThriftTools.ThriftClient(ip_address,Inputs.constants.DEFAULT_RELATON_EXTRACTION_PORT,Relation_Service,'RelationExtraction')
    # while not relation_client.connected:
    #     relation_client.connect()
    # print "Successfully connect with Relation Extraction"


    em_handler = EMNadineHandler()
    print "Inputs.constants.DEFAULT_EMNADINE_PORT",Inputs.constants.DEFAULT_EMNADINE_PORT
    em_server = ThriftTools.ThriftServerThread(Inputs.constants.DEFAULT_EMNADINE_PORT,EM_Service,em_handler,'EMNadine Service',ip_address)
    em_server.start()
    print "Started the server"

    inputs=['sentence= Hello there', 'subject=Unknown', 'emotion=None 0.5', 'mood= 0.5', 'eventState=Present', 'user=Alex', 'weekday=Thu', 'socialTime=Afternoon']
    em_handler.getPeerEvent(inputs,None,"Nadia")
    em_handler.getGreetings("user=Alex")

