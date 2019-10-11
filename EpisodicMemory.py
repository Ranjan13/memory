
from numpy import array,zeros,dot
import numpy as np
import cPickle as pickle
import sys
import os
import math
from copy import copy
import BasicOperation as opt
import Definition
from nltk.metrics.distance import edit_distance
import processInput
from loadFolder import getPath
from shortTermMemory import shortTermMemory
from findTime import findTime
from conceptNetWrapper import conceptNetWrapper
from randomFunc import randomFunc

class EpisodicMemory:
    def __init__(self):
        #self.processEvent=processInput.processInput({})
        self.connectNLP()
        self._build()
        #self.loadFixedData()
        #self.loadMemory()
        #self.refineDataStructure()
        #self.saveFixedData()
        #print "Start Working"
        #self.saveMemory()


############################  EM + NLP ######################################
    def connectNLP(self):
        print "Connect NLP..."
        self.processEvent=processInput.processInput({})

    def _build(self):
        print "Initializing..."
        self.STM=shortTermMemory()
        self.cnw=conceptNetWrapper()
        self.timeAnalyser=findTime()
        self.episodes=[]
        self.timeIndex={}
        self.userIndex={}
        self.attrIndex={}
        self.forbidIndex=[]
        self.incompleteSentenceIndex=[]
        self.questionIndex=[]
        self.YOUIndex=[]
        self.eventClusterDB=[] # to be stored
        self.episodeClusterDB=[] # to be stored
        self.userPlan={}
        self.S2=None
        self.V2=None
        self.AttrCoord={}
        self.users=[]
        self.randomFunc=randomFunc()


    def reorderEpisodeIndex(self,EpIndexList):
        indexList=list(set(EpIndexList))
        if len(indexList)==1:
            return [range(len(EpIndexList))]
        # reorder if there are multiple episode indexes
        timeList=[self.episodes[idx].getTimeStr() for idx in indexList]
        reorderedIndex=self.timeAnalyser.reorderFullTimeIndex(timeList)
        reorderedIndexList=[indexList[i] for i in reorderedIndex]
        groupedIndex=[]
        for idx in reorderedIndexList:
            temp=[i for i,x in enumerate(EpIndexList) if x==idx]
            groupedIndex.append(temp)
        return groupedIndex

    def reorderEvents(self,events):
        # from the latest to the earliest
        if len(events)<=1:
            return events
        episodeIndexList=[self.getEpisodeIdx(event.index) for event in events]
        groupedIndex=self.reorderEpisodeIndex(episodeIndexList)
        res=[]
        for group in groupedIndex:
            groupEv=[events[i] for i in group]
            temp=self.reorderEventInSameEpisode(groupEv)
            res.extend(temp)
        return res

    def reorderEventInSameEpisode(self,candidates):
        # from the latest to the earliest
        res=sorted(candidates, key=lambda ev:-ev.index)
        return res




    def getUsers(self):
        self.users=[]
        for label in self.userIndex.keys():
            _u=label.split("=")[1]
            self.users.append(_u)


    def refineDataStructure(self):
        for i in range(len(self.episodes)):
            episode=self.episodes[i]
            epDict=episode.__dict__
            del epDict["eventClusterIdx"]
            epDict["latestDate"]=None
            for j in range(len(episode.content)):
                event=episode.content[j]
                for k in range(len(event.content)):
                    attr=event.content[k]
                    atDict=attr.__dict__
                    atDict["substitution"]=None

    def loadFixedData(self):
        file=open("clusterDB.pickle", "rb")
        self.eventClusterDB=pickle.load(file)
        self.episodeClusterDB=pickle.load(file)
        self.S2=pickle.load(file)
        self.V2=np.diag(pickle.load(file))
        self.AttrCoord=pickle.load(file)
        self.processEvent.AttrCoord=self.AttrCoord
        self.updateClusterIdx()
        print "data loaded"

    def clearClusterIdx(self):
        for i in range(len(self.episodeClusterDB)):
            self.episodeClusterDB[i].elementIdxList=[]
        for i in range(len(self.eventClusterDB)):
            self.eventClusterDB[i].elementIdxList=[]

    def updateClusterIdx(self):
        #self.clearClusterIdx()
        for episode in self.episodes:
            self.episodeClusterDB[episode.category].elementIdxList.append(episode.index)
            for event in episode.content:
                self.eventClusterDB[event.category].elementIdxList.append(event.index)

    def removeEventNoInfo(self,episode):
        for event in episode.content:
            if self.isTrivialEvent(event):
                episode.content.remove(event)

    def isTrivialEvent(self,event):
        words=event.getAttrLabelType("knownWord")
        if words == None:
            return True
        elif len(event.sentence)<=5:
            return True
        return False



    def saveFixedData(self):
        self.clearClusterIdx()
        file=open("clusterDB.pickle", "wb")
        pickle.dump(self.eventClusterDB,file)
        pickle.dump(self.episodeClusterDB,file)
        pickle.dump(self.S2,file)
        pickle.dump(np.diag(self.V2),file)
        pickle.dump(self.AttrCoord,file)
        file.close()

    def loadMemory(self,user=None):
        print "Loading Past Episodes..."
        #curDate=self.timeAnalyser.getDate(self.timeAnalyser.currentTime)
        path=os.getcwd()+"\\Episodes"
        if user!=None:
            path=path+"\\"+user
        if os.path.exists(path):
            filePaths=getPath(path)
            for _p in filePaths:
                try:
                    episode=pickle.load(open(_p, "rb"))
                    self.loadEpisode(episode,_p)
                except:
                    print sys.exc_info()
                    print "Fail in loading episode: ",_p
            print "Memory Loaded!"
        else:
            print "The file doesn't exist!!!"

    def updateUser(self, episode, user):
        user=user.capitalize()
        print "Update User: ",user
        episode.user=user
        for i in range(len(episode.content)):
            event=episode.content[i]
            for j in range(len(event.content)):
                attr=event.content[j]
                if attr.type=="subject":
                    if attr.value!="subject=Robot":
                        attr.value="subject="+user
                elif attr.type=="user":
                    if attr.value!="user=Robot":
                        attr.value="user="+user

    def updateMemIntensity(self,episode,curDate):
        if episode.latestDate!=None:
            days=self.timeAnalyser.getPastDays(episode.latestDate)
            days=abs(days)
            episode.start(days)
        episode.latestDate=curDate
        forgetFlag=episode.dealForgetting()
        return forgetFlag

    def getConceptNetAnswer(self,querySent):
        nouns=self.processEvent.getNouns(querySent)
        if nouns==None:
            return None
        if len(self.users)==0:
            self.getUsers()
        common=set(nouns) & set(self.users)
        for word in common:
            nouns.remove(word)

        while len(nouns)>0:
            cue=self.randomFunc.randomChoose(nouns)
            reply=self.cnw.get_reply(cue)
            if reply!=None:
                return reply
            nouns.remove(cue)
        return None





    def getAttrSubstitution(self,attr,event):
        if attr.isForgotten and attr.substitution==None:
            if attr.type=="knownWord":
                if attr.tag!=None and attr.tag.startswith("NN"):
                    word=attr.label.split("=")[1]
                    word="/c/en/"+word
                    candidate=self.cnw.get_abstract_concept(word)
                    if len(candidate)>0:
                        self.processEvent.getSimilarWord(attr,event,candidate)
                        return attr.substitution
                    else:
                        attr.substitution=False
                else:
                    attr.substitution=False
            elif attr.type=="subject":
                attr.substitution="somebody"
                return attr.substitution
            elif attr.type=="emotion":
                attr.substitution="Neutral"
            elif attr.type=="mood":
                attr.substitution="Neutral"
            elif attr.type=="user":
                attr.substitution="somebody"
            elif attr.type=="weekday":
                attr.substitution="someday"
            elif attr.type=="socialTime":
                attr.substitution="sometime"
        elif attr.substitution!=False:
            return attr.substitution
        return None

    def getEventAnswer(self,event):
        sent_change_flag="False"
        who_change_flag="False"
        sent=copy(event.sentence)
        who=None
        for i in range(len(event.content)):
            attr=event.content[i]
            if attr.isForgotten:
                self.getAttrSubstitution(attr,event)
                if attr.type=="knownWord" and attr.tag.startswith("NN"):
                    word=attr.label.split("=")[1]
                    if attr.substitution and (word in sent):
                        sent=self.processEvent.substituteWord(word,attr.substitution,sent)
                        sent_change_flag="True"
            if attr.type=="user":
                if attr.isForgotten:
                    who=attr.substitution # True stands for forgotten
                    who_change_flag="True"
                else:
                    who=attr.label.split("=")[1]
        res=[sent,who,sent_change_flag,who_change_flag]
        return res






    def saveMemory(self):
        for ep in self.episodes:
            ep.save()
        #self.saveFixedData()
        print "Memory Saved!"


    def loadEpisode(self,episode,path):
        try:
            episode.index=len(self.episodes)
            evIdx=self.getLastMaxEventIdx()+1
            #episode=self.rebuildEpisode(episode)
            for event in episode.content:
                event.index=evIdx
                self.indexingNewEvent(event)
                evIdx+=1

            episode.maxEvIdx=self.getCurrentMaxEventIdx(episode)
            # update user index
            self.updateUserIndex("user="+episode.user,episode.index)
            # update TimeIndex
            self.updateTimeIndex(episode.date,episode.index)
            episode.path=path
            self.episodes.append(episode)
        except:
            print sys.exc_info()

    # rebuild events of loaded episode to new data structure
    def rebuildEvent(self,event):
        inputs=self.getEventInput(event)
        ev=self.processEvent.buildEvent(inputs)
        return ev

    def getEventInput(self,event):
        inputs=[]
        inputs.append("sentence="+event.sentence)
        for attr in event.content:
            if attr.type in ["emotion","mood"]:
                if attr.label=="mood=":
                    attr.label="mood=None"
                inputs.append(attr.label+" "+str(attr.value/2.0))
            elif attr.type!="knownWord":
                inputs.append(attr.label)
        return inputs

    # rebuild loaded episode to new data structure
    def rebuildEpisode(self,episode):
        res=Definition.Episode()
        for event in episode.content:
            ev=self.rebuildEvent(event)
            if ev!=None:
                res.content.append(ev)
        res.maxEvIdx=episode.maxEvIdx
        res.date=episode.date
        res.index=episode.index
        res.user=episode.user
        return res

    # delete episode from software storage
    def unloadEpisode(self,episode):
        self.episodes.pop(episode.index)
        for event in episode.content:
            self.removeAttrIndex(event)
            self.removeQuestionIndex(event)
        self.removeUserIndex(episode.user,episode.index)
        self.removeTimeIndex(episode.date,episode.index)

    # delete episode from software and hardware storage
    def deleteEpisode(self,episode):
        self.unloadEpisode(episode)
        episode.delete()
        if len(self.userIndex["user="+episode.user])==0:
            self.deleteUserFolder(episode.user)
            self.userIndex.pop("user="+episode.user)

    # delete user from software and hardware storage
    def deleteUser(self,user):
        if "=" not in user:
            user="user="+user
        epIdxs=self.userIndex[user]
        for epIdx in epIdxs:
            episode=self.episodes[epIdx]
            self.deleteEpisode(episode)
        self.userIndex.pop(user)
        self.deleteUserFolder(user)

    # delete user from hardware storage
    def deleteUserFolder(self,user):
        if "=" in user:
            user=user.split("=")[1]
        path=os.getcwd()+"\\Episodes"
        if user!=None:
            path=path+"\\"+user
        if os.path.exists(path):
            os.rmdir(path)

    # delete event from software storage
    # it will be removed from hardware when the episode is stored
    def deleteEvent(self,episode,event):
        episode.content.remove(event)
        self.removeAttrIndex(event)
        self.removeQuestionIndex(event)
        if len(episode.content)==0:
            self.deleteEpisode(episode)


    def getCurrentMaxEventIdx(self,episode):
        return self.getLastMaxEventIdx()+len(episode.content)

    def getLastMaxEventIdx(self):
        if len(self.episodes)>0:
            lastEp=self.episodes[-1]
            res=lastEp.maxEvIdx
            return res
        else:
            return -1



    # def eventOnlineClustering(self,event,th=0.3):
    #     evIdx=event.index
    #     if len(self.EventExampler_L1)==0:
    #         # initialize
    #         self.EventExampler_L1.append(event.coordinate)
    #         self.event_cluster_Index[evIdx]=0
    #         self.cluster_event_Index[0]=[evIdx]
    #     else:
    #         cue_coord=event.coordinate
    #         clusterIdx=self.getEventClusterIdx(cue_coord,th)
    #         if not clusterIdx:
    #             # create new cluster
    #             self.EventExampler_L1.append(event.coordinate)
    #             clusterIdx=len(self.EventExampler_L1)-1
    #         else:
    #             clusterIdx=clusterIdx[0]
    #         event.category=clusterIdx
    #         self.event_cluster_Index[evIdx]=clusterIdx
    #         if clusterIdx not in self.cluster_event_Index.keys():
    #             self.cluster_event_Index[clusterIdx]=[]
    #         self.cluster_event_Index[clusterIdx].append(evIdx)


    # def getEventClusterIdx(self,cue_coord,th=0.3):
    #     cos_array=[opt._cos(cue_coord,cluster) for cluster in self.EventExampler_L1]
    #     (ev_cluster_idx,_)=opt.find_above_threshold(cos_array,th)
    #     if len(ev_cluster_idx)==0:
    #         return None
    #     else:
    #         return ev_cluster_idx
    #
    # def getMostMatchedCluster(self,cue_coord,num=5):
    #     cos_array=[opt._cos(cue_coord,cluster) for cluster in self.EventExampler_L1]
    #     (ev_cluster_idx,_)=opt.find_n_max(cos_array,num)
    #     return ev_cluster_idx

    def getSimilarEventOverSubset(self,cueInputs,eventIdxList,th=0.3):
        cue_coord=None
        if isinstance(cueInputs,np.ndarray):
            cue_coord=cueInputs
        elif isinstance(cueInputs,list):
            cue=self.processEvent.buildEvent(cueInputs)
            if cue==None:
                return None
            cue_coord=cue.coordinate
        else:
            return None
        eventList=[self.retrieveRightEvent(idx) for idx in eventIdxList]
        cos_array=[opt._cos(cue_coord,event.coordinate) for event in eventList]
        (candidate_idx,_)=opt.find_above_threshold(cos_array,th)
        if len(candidate_idx)==0:
            return None
        else:
            res=[eventList[id] for id in candidate_idx]
            return res

    def getSTMSimilarEvent(self,cue,eventList,num=5):
        if isinstance(cue,np.ndarray):
            cue_coord=cue
        else:
            cue_coord=cue.coordinate

        if cue_coord==None:
            return (None,None)
        #eventList=[self.retrieveRightEvent(idx) for idx in eventIdxList]

        if len(eventList)==0:
            return (None,None)


        cos_array=[opt._cos(cue_coord,event.coordinate) for event in eventList]
        (candidate_idx,n_max)=opt.find_n_max(cos_array,num)
        if len(candidate_idx)==0:
            return (None,None)
        else:
            res=[eventList[id] for id in candidate_idx]
            return (res,n_max)

    def getMostSimilarEventOverSubset(self,cue,episodeIdxList=None,num=5,curFlag=True):
        cue_coord=None
        if isinstance(cue,np.ndarray):
            cue_coord=cue
        else:
            cue_coord=cue.coordinate

        if cue_coord==None:
            return (None,None)
        #eventList=[self.retrieveRightEvent(idx) for idx in eventIdxList]
        eventList=[]
        clearIdx=set(self.questionIndex) | set(self.forbidIndex) \
                 | set(self.incompleteSentenceIndex) | set(self.YOUIndex)
        if self.attrIndex.has_key("subject=Robot"):
            clearIdx=clearIdx | set(self.attrIndex["subject=Robot"])
        if not episodeIdxList:
            episodeIdxList=range(0,len(self.episodes))


        for epIdx in episodeIdxList:
            episode=self.episodes[epIdx]
            for event in episode.content:
                if event.index not in clearIdx:
                    eventList.append(event)
        if len(eventList)==0:
            return (None,None)

        ################Added by James 31/05/2016###################
        if not curFlag:
            curEp=self.episodes[-1]
            if len(curEp.content)>0:
                curEvIdx=curEp.maxEvIdx
                for i in range(len(eventList)):
                    if eventList[i].index==curEvIdx:
                        eventList.pop(i)
                        break
        ######################################

        cos_array=[opt._cos(cue_coord,event.coordinate) for event in eventList]
        #cos_array=[opt._cos(cue_coord,event.coordinate) for event in eventList]
        (candidate_idx,n_max)=opt.find_n_max(cos_array,num)
        if len(candidate_idx)==0:
            return (None,None)
        else:
            res=[eventList[id] for id in candidate_idx]
            return (res,n_max)

    def getExactEventOverSubset(self,attrs,eventIdxList,num=None,allFlag=False):
        EvIdx=self.getExactEvIdx(attrs,union=not allFlag)
        if not EvIdx:
            return None
        bound=set(eventIdxList)
        EvIdx=EvIdx.intersection(bound)
        if len(EvIdx)==0:
            return None
        candidates=[self.retrieveRightEvent(idx) for idx in EvIdx]
        if num==None or len(EvIdx)<num:
            return candidates
        else:
            return candidates[:num]

    def getExactEvent(self,attrs,episodeIdxList=None,num=None):
        words,_=self.processEvent.getKnownWords(attrs)
        EvIdx=self.getExactEvIdx(words,False)
        eventList=[]
        clearIdx=[]
        if self.attrIndex.has_key("subject=Robot"):
            clearIdx=self.attrIndex["subject=Robot"]
        if not episodeIdxList:
            episodeIdxList=range(0,len(self.episodes))
        for epIdx in episodeIdxList:
            episode=self.episodes[epIdx]
            for event in episode.content:
                if event.index not in clearIdx:
                    eventList.append(event.index)
        Idx=set(eventList) & set(EvIdx)
        if len(Idx)==0:
            return None
        candidates=[self.retrieveRightEvent(idx) for idx in Idx]
        if num==None or len(EvIdx)<num:
            return candidates
        else:
            return candidates[:num]

    def removeLatestEvents(self,curFlag,eventIdxList,num=2,eventFlag=False):
        # num: online memory retrieval is started after how many rounds of interaction
        # if eventFlag is true, consider list of events, else, consider list of index
        if not curFlag:
            try:
                curEp=self.episodes[-1]
                if len(curEp.content)>0:
                    latestEvIdx=curEp.getEventIndex()[-num:]
                    for idx in latestEvIdx:
                        if not eventFlag:
                            if idx in eventIdxList:
                                eventIdxList.remove(idx)
                        else:
                            for i in range(len(eventIdxList)):
                                if eventIdxList[i].index==idx:
                                    eventIdxList.pop(i)
                                    break
            except:
                print sys.exc_info()[0]


    def getExactSimilarEvent(self,cue,candidateIdxList=None,num=5,curFlag=True,allFlag=False):
        # curFlag: if consider current event in the searching scope
        reswords,cue_coord=None,None
        if isinstance(cue,np.ndarray):
            cue_coord=cue
        else:
            cue_coord=cue.coordinate
            reswords=cue.getAttrLabelType("knownWord")

        clearIdx=set(self.questionIndex) | set(self.forbidIndex)\
                 | set(self.incompleteSentenceIndex) | set(self.YOUIndex)
        if self.attrIndex.has_key("subject=Robot"):
            clearIdx=clearIdx | set(self.attrIndex["subject=Robot"])

        eventIdxList=[]
        if not candidateIdxList:
            candidateIdxList=range(0,len(self.episodes))
        for epidx in candidateIdxList:
            for event in self.episodes[epidx].content:
                if event.index not in clearIdx:
                    eventIdxList.append(event.index)
        if len(eventIdxList)==0:
            return (None,None)
        ################Added by James 31/05/2016###################
        self.removeLatestEvents(curFlag=curFlag,eventIdxList=eventIdxList)
        ######################################
        eventList=self.getExactEventOverSubset(reswords,eventIdxList,allFlag=allFlag)
        if eventList==None:
            return (None,None)

        #print eventList
        cos_array=[opt._cos(cue_coord,event.coordinate) for event in eventList]
        #cos_array=[opt._cos(cue_coord,event.coordinate) for event in eventList]
        (candidate_idx,n_max)=opt.find_n_max(cos_array,num)
        if len(candidate_idx)==0:
            return (None,None)
        else:
            res=[eventList[id] for id in candidate_idx]
            return (res,n_max)


    def getRelatedQuestion(self,cue,candidateIdxList=None,exactFlag=True,num=5,curFlag=True):
        reswords,cue_coord=None,None
        if isinstance(cue,np.ndarray):
            cue_coord=cue
        else:
            cue_coord=cue.coordinate
            reswords=cue.getAttrLabelType("knownWord")


        hardConIdx=set(self.questionIndex)
        if self.attrIndex.has_key("subject=Robot"):
            hardConIdx=hardConIdx & set(self.attrIndex["subject=Robot"])
        if not candidateIdxList:
            candidateIdxList=range(0,len(self.episodes))

        eventList=None
        if exactFlag:
            eventIdxList=[]
            for epidx in candidateIdxList:
                for event in self.episodes[epidx].content:
                    if event.index in hardConIdx:
                        eventIdxList.append(event.index)
            if len(eventIdxList)==0:
                return (None,None)
            ################Added by James 31/05/2016###################
            self.removeLatestEvents(curFlag=curFlag,eventIdxList=eventIdxList)
            ######################################
            eventList=self.getExactEventOverSubset(reswords,eventIdxList)
        else:
            eventList=[]
            for epidx in candidateIdxList:
                for event in self.episodes[epidx].content:
                    if event.index in hardConIdx:
                        eventList.append(event)
            ################Added by James 31/05/2016###################
            self.removeLatestEvents(curFlag=curFlag,eventIdxList=eventList,eventFlag=True)
            ######################################
        if eventList==None:
            return (None,None)


        #print eventList
        cos_array=[opt._cos(cue_coord,event.coordinate) for event in eventList]
        #cos_array=[opt._cos(cue_coord,event.coordinate) for event in eventList]
        (candidate_idx,n_max)=opt.find_n_max(cos_array,num)
        if len(candidate_idx)==0:
            return (None,None)
        else:
            quesEvents=[eventList[id] for id in candidate_idx]
            res,n_max=self.getAnswerEvent(quesEvents,n_max)
            return (res,n_max)

    def getAllQuestion(self,candidateIdxList):
        hardConIdx=set(self.questionIndex)
        if self.attrIndex.has_key("subject=Robot"):
            hardConIdx=hardConIdx & set(self.attrIndex["subject=Robot"])


        questionEvents=[]
        for epidx in candidateIdxList:
            for event in self.episodes[epidx].content:
                if event.index in hardConIdx:
                    questionEvents.append(event)
        if len(questionEvents)==0:
            return None

        ansEv=self.getAnswerEvent(questionEvents)
        res=[ev.sentence for ev in ansEv]
        return res

    def getAnswerEvent(self,questionEvents,n_max=None):
        res=[]
        res_max=[]
        flag=False
        if n_max!=None and len(n_max)==len(questionEvents):
            flag=True
        idx=0
        for event in questionEvents:
            question=event.sentence
            nextEvent=self.retrieveRightEvent(event.index+1)
            if nextEvent.question or nextEvent.index in self.YOUIndex:
                continue
            answer=nextEvent.sentence

            que_noun=self.processEvent.getNouns(question)
            ans_noun=self.processEvent.getNouns(answer)
            x1=que_noun!=None
            x2=ans_noun!=None
            if x1 and x2 and self.processEvent.isContain(que_noun,ans_noun):
                res.append(nextEvent)
                if flag:
                    res_max.append(n_max[idx])
            elif answer.startswith("i") or answer.startswith("my"):
                res.append(nextEvent)
                if flag:
                    res_max.append(n_max[idx])
            idx+=1
        if flag:
            return res,res_max
        return res






    def getExactEvIdx(self,attrs,union=True):
        _atts=[]
        for att in attrs:
            if att in self.attrIndex.keys():
                _atts.append(att)
        if len(_atts)==0:
            return None
        res=set(self.attrIndex[_atts[0]])
        for i in range(1,len(_atts)):
            ss=set(self.attrIndex[_atts[i]])
            if union:
                res=res | ss  # get_union
            else:
                res=res & ss # get common
            if len(res)==0:
                return None
        return res

    def indexingNewEpisode(self,episode):
        # update user index
        self.updateUserIndex("user="+episode.user,episode.index)
        # update TimeIndex
        self.updateTimeIndex(episode.date,episode.index)

    def indexingNewEvent(self,event):
        self.updateQuestionIndex(event)
        self.updateForbiddenIndex(event)
        self.updateIncompleteIndex(event)
        self.updateAttrIndex(event)
        self.updateYOUIndex(event)

    def updateUserPlan(self,event):
        eventState=event.getAttrLabelType("eventState")
        user=event.getAttrLabelType("user")
        if eventState in ["eventState=Future"]:
            if user not in self.userPlan.keys():
                self.userPlan[user]=opt.Queue()
            self.userPlan[user].push(event)

    def updateUserIndex(self,user,epIdx):
        if user not in self.userIndex.keys():
            self.userIndex[user]=[]
        self.userIndex[user].append(epIdx)

    def removeUserIndex(self,user,epIdx):
        self.userIndex[user].remove(epIdx)
        if len(self.userIndex[user])==0:
            self.userIndex.pop(user)


    def updateTimeIndex(self,time,epIdx):
        if time not in self.timeIndex.keys():
            self.timeIndex[time]=[]
        self.timeIndex[time].append(epIdx)

    def removeTimeIndex(self,time,epIdx):
        self.timeIndex[time].remove(epIdx)
        if len(self.timeIndex[time])==0:
            self.timeIndex.pop(time)

    def updateQuestionIndex(self,event):
        if event.question and event.index not in self.questionIndex:
            self.questionIndex.append(event.index)

    def removeQuestionIndex(self,event):
        if event.question and event.index in self.questionIndex:
            self.questionIndex.remove(event.index)


    def updateEpisodes(self,episode):
        self.episodes.append(episode)

    def removeEpisodes(self,episode):
        self.episodes.remove(episode)

    def updateAttrIndex(self,event):
        for attr in event.content:
            if attr.label not in self.attrIndex.keys():
                self.attrIndex[attr.label]=[]
            self.attrIndex[attr.label].append(event.index)

    def removeAttrIndex(self,event):
        for attr in event.content:
            self.attrIndex[attr.label].remove(event.index)
            if len(self.attrIndex[attr.label])==0:
                self.attrIndex.pop(attr.label)

    def checkForbidWords(self,sentence):
        forbidWords=["silly","stupid","dummy","ugly","horrible","hate"]
        for word in forbidWords:
            if word in sentence:
                return True
        return False

    def updateForbiddenIndex(self,event):
        if self.checkForbidWords(event.sentence):
            self.forbidIndex.append(event.index)

    def removeForbiddenIndex(self,event):
        if self.checkForbidWords(event.sentence):
            self.forbidIndex.remove(event.index)

    def updateIncompleteIndex(self,event):
        if not self.processEvent.isCompleteSentence("sentence="+event.sentence):
            self.incompleteSentenceIndex.append(event.index)


    def removeIncompleteIndex(self,event):
        if not self.processEvent.isCompleteSentence("sentence="+event.sentence):
            self.incompleteSentenceIndex.remove(event.index)

    def updateYOUIndex(self,event):
        referWords=["you","your","he","she","his","her","it"]
        tokens=event.sentence.lower().split()
        for word in referWords:
            if word in tokens:
                self.YOUIndex.append(event.index)
                break


    def removeYOUIndex(self,event):
        referWords=["you","your","he","she","his","her","it"]
        tokens=event.sentence.lower().split()
        for word in referWords:
            if word in tokens:
                self.YOUIndex.remove(event.index)
                break



############################  Supporting Functions ######################################

    def retrieveRightEvent(self,event_idx):
        if len(self.episodes)==0:
            return None
        LargestEvIdx=self.episodes[-1].maxEvIdx
        if event_idx>LargestEvIdx:
            return None
        (epIdx,order)=self.getEventIndex(event_idx,0,len(self.episodes)-1)
        episode=self.episodes[epIdx]
        event=episode.content[order]
        return event

    def retrieveRightEpisode(self,event_idx):
        epIdx=self.getEpisodeIdx(event_idx)
        episode=self.episodes[epIdx]
        return episode

    def getEpisodeIdx(self,event_idx):
        LargestEpIdx=len(self.episodes)-1
        (epIdx,order)=self.getEventIndex(event_idx,0,LargestEpIdx)
        return epIdx

    def getEventIndex(self,event_idx,n1,n2):
        # n1=0
        # n2=len(self.training_episodes)
        if event_idx<=self.episodes[0].maxEvIdx:
            return (0,event_idx)
        else:
            return self.iterGetIdx(event_idx,n1,n2)

    def iterGetIdx(self,event_idx,n1,n2):
        if n2-n1<=1:
            episode=self.episodes[n1]
            maxIdx=episode.maxEvIdx
            return (n2,event_idx-maxIdx-1)
        i=n1+(n2-n1)/2
        episode=self.episodes[i]
        maxIdx=episode.maxEvIdx
        if maxIdx==event_idx:
            # return (episodeIdx,which event)
            return (i,len(episode.content)-1)
        elif maxIdx>event_idx:
            return self.getEventIndex(event_idx,n1,i)
        else:
            return self.getEventIndex(event_idx,i,n2)

    # def retrieveRelatedAttributes(self,cue,attr_class,num=5):
    #     # cue_coord_L1 is mdarray
    #     # attr_class contains "Any","Pre","Sub","DirObj","IndirObj","Loc"
    #
    #     cue_coord=self.processEvent.getAttrCoord(cue)
    #
    #     if attr_class not in self.AttributeClass.keys(): # search for any attribute
    #         cos_array=[opt._cos(cue_coord,coord) for coord in self.processEvent.AttrCoord.values()]
    #         n_idx,n_max=opt.find_n_max(cos_array,num)
    #         res=[self.processEvent.AttrCoord.keys()[idx] for idx in n_idx]
    #         if len(res)==1:
    #             return res[0],n_max[0]
    #         elif len(res)==0:
    #             return None
    #         return res,n_max
    #     else: # search in this class
    #         same_class=False
    #         if len(cue)==1:
    #             _cue=cue[0]
    #             if _cue.startswith(attr_class): # if the cue just in the attr_class
    #                 num+=1 # retrieve one more and remove the cue from the result
    #                 same_class=True
    #
    #         attrs=self.AttributeClass[attr_class]
    #         coords=[self.processEvent.AttrCoord[attr] for attr in attrs]
    #         cos_array=[opt._cos(cue_coord,coord) for coord in coords]
    #         n_idx,n_max=opt.find_n_max(cos_array,num)
    #         res=[attrs[idx] for idx in n_idx]
    #         if same_class:
    #             if cue[0] in res:
    #                 del res[res.index(cue[0])]
    #         if len(res)==1:
    #             return res[0],n_max[0]
    #         elif len(res)==0:
    #             return None
    #         return res,n_max



if __name__=="__main__":
    EM=EpisodicMemory()
    EM.loadMemory()
    events=[]
    for episode in EM.episodes[:10]:
        events.append(episode.content[0])
        events.append(episode.content[-1])
    res,_=EM.reorderEvents(events)
    print res
    while 1:
         sent=raw_input("User: ")
         if sent in ["q","quit"]:
             break
         reply=EM.getConceptNetAnswer(sent)
         if reply!=None:
             print "Robot: "+reply
         else:
             print "Robot: I have no answer"
    print 1