import csv;
import sklearn;
import sklearn.utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statistics
import math
import copy
import numpy as np
import random
def openFile(x):
    file = open(x)
    data = csv.reader(file)
    rows = [];

    for row in data:
        rows.append(row)

    rows.pop()
    return rows;

def shuffleAndSplit(rows):
    train, test = sklearn.model_selection.train_test_split(rows, test_size = .2, shuffle = True)
    return train, test

###############

def makeBags(set, number):
    bagList = [];
    for x in range(number):
        bag = []
        for y in range(len(set)):
            bag.append(copy.deepcopy(set[random.randint(0,len(set)-1)]))
        bagList.append(bag);
    return copy.deepcopy(bagList)

def selectIssues(issues, names):
    M = math.ceil(math.sqrt(len(issues)))
    issueHold = copy.deepcopy(issues);
    nameHold = copy.deepcopy(names)
    issueSubset = []
    nameSubset = []
    for x in range(M):
        choice = random.randint(0,len(issueHold)-1)
        issueSubset.append(copy.deepcopy(issueHold[choice]))
        nameSubset.append(copy.deepcopy(nameHold[choice]))
        issueHold.remove(issueHold[choice])
        nameHold.remove(nameHold[choice])

    return issueSubset, nameSubset

def majorityVote(guessArray):
    trueguesses = []
    for x in range(len(guessArray[0])-1):
        oneCount = 0;
        for array in guessArray:
            if(array[x]== '1'):
                oneCount +=1

        if((len(guessArray)-oneCount)>oneCount):
            trueguesses.append(0)
        elif((len(guessArray)-oneCount)<oneCount):
            trueguesses.append(1)
        else:
            trueguesses.append(random.randint(0,1))
    return trueguesses

def calculateMetrics(guesses, truth):
    truePos = 0
    falsePos = 0
    trueNeg = 0
    falseNeg = 0
    for x in range(len(guesses)-1):
        if(guesses[x]== truth[x]):
           if(truth[x] == 0):
               trueNeg += 1
           else:
               truePos +=1
        else:
            if(truth[x] == 0):
                falsePos +=1
            else:
                falseNeg +=1 
            
    accuracy = (trueNeg + truePos)/len(guesses)

    if(truePos+falsePos == 0):
        precision = 0;
    else:
        precision = truePos/(truePos+falsePos)

    if(truePos + falseNeg == 0):
        recall = 0
    else:
        recall =  truePos/(truePos+falseNeg)
    if((precision==0)&(recall==0)):
        fScore = 0;
    else:
        fScore = 2*(precision*recall)/(precision+recall)
    return accuracy, precision, recall, fScore

def gatherParty(people):
    parties = []
    for person in people:
        parties.append(int(person[16]))
    return parties

def sortVotes(votes):
    zeroVoters = []
    oneVoters = []
    twoVoters = []
    for x in range(len(votes)):
        if(votes[x] == '0'):
            zeroVoters.append(x)
        if(votes[x] == '1'):
            oneVoters.append(x)
        if(votes[x] == '2'):
            twoVoters.append(x)
    return [zeroVoters, oneVoters, twoVoters];

def calculateEntropy(total):
    entropies = []
    for branch in total:
        case0 = 0;
        case1 = 0;
        if(len(branch)>0):
            for x in branch:
                if(x == '0'):
                    case0 = case0+1;
                if(x == '1'):
                    case1 = case1+1;

            frac0 = case0/len(branch);
            frac1 = case1/len(branch);
    
            if(frac0 == 0):
                entropy = -frac1*math.log(frac1,2);
            elif(frac1 == 0):
                entropy = -(frac0*math.log(frac0,2))
            else:
                entropy = -frac0*math.log(frac0,2) - frac1*math.log(frac1,2)
            entropies.append(entropy)

        else:
            entropies.append(0);
    return entropies


def splitIntoIssues(people):
    allIssues = []
    for x in range(len(people[0])):
        issue = []
        for y in range(len(people)-1):
            issue.append(people[y][x]);
        allIssues.append(issue)

    return allIssues;

def findMore(array):
    zeroCount = 0;
    oneCount = 0;
    for x in array:
        if(x == '0'):
            zeroCount = zeroCount + 1
        if(x == '1'):
            oneCount = oneCount + 1
    if(zeroCount>=oneCount):
        return '0'
    else:
        return '1'
    
def findMoreOpp(array):
    zeroCount = 0;
    oneCount = 0;
    for x in array:
        if(x == '0'):
            zeroCount = zeroCount + 1
        if(x == '1'):
            oneCount = oneCount + 1
    if(zeroCount>=oneCount):
        return '0'
    else:
        return '1'

def calcualteInformationGain(proportions, entropies, baseGroupEntropy):
    infoGain =  baseGroupEntropy - (proportions[0]*entropies[0] + proportions[1]*entropies[1] + proportions[2]*entropies[2])
    return infoGain

def findParties(group, lables):
    labledGroup = []    
    for x in group:
        hold = []
        for y in x:
            hold.append(lables[int(y)])
        labledGroup.append(hold)
    return labledGroup

def constructDecisionTree(voters, issueNames, level, stopLevel):
    reorganized = splitIntoIssues(voters);
    issues = copy.deepcopy(reorganized[:len(reorganized)-1])
    issueSubset, issueNameSubset = selectIssues(issues, issueNames);

    lables = copy.deepcopy(reorganized[len(reorganized)-1]);
    bestSplit = [-1, -1, ""]
    bestEntropies = [-1,-1,-1]
    if(len(issueNameSubset)==1):
        fullGroupEntropy = calculateEntropy([lables])
        IDgroups = sortVotes(issues[0]);
        labelGroups = findParties(IDgroups, lables)
        entropies = calculateEntropy(labelGroups);
        proportions = [len(IDgroups[0])/len(voters), len(IDgroups[1])/len(voters),len(IDgroups[2])/len(voters)]
        infoGain = calcualteInformationGain(proportions, entropies, fullGroupEntropy[0])


        if(infoGain>bestSplit[0]):
            bestSplit = [infoGain, 0, issueNameSubset[0]];
    else:
        for x in range(len(issueSubset)-1):
            fullGroupEntropy = calculateEntropy([lables])
            IDgroups = sortVotes(issueSubset[x]);
            labelGroups = findParties(IDgroups, lables)
            entropies = calculateEntropy(labelGroups);
            proportions = [len(IDgroups[0])/len(voters), len(IDgroups[1])/len(voters),len(IDgroups[2])/len(voters)]
            infoGain = calcualteInformationGain(proportions, entropies, fullGroupEntropy[0])


            if(infoGain>bestSplit[0]):
                bestEntropies = [entropies[0],entropies[1],entropies[2],copy.deepcopy(labelGroups)]
                bestSplit = [infoGain, issueNames.index(issueNameSubset[x]), issueNameSubset[x]];



    #######AFTER THE BEST SPLIT HAS BEEN FOUND
    
    choices = [-1,-1,-1, bestSplit[2]]
    issueNames.remove(bestSplit[2])
    default = findMore(lables)
    if((bestEntropies[0] == 0)&(len(bestEntropies[3][0])>0)):
        if(bestEntropies[3][0][0] == '0'):
            choices[0] = '0'
        else:
            choices[0] = '1'
    else:
        nextBatch = []
        for x in IDgroups[0]:
            del voters[x][bestSplit[1]]
            nextBatch.append(voters[x])

        if(len(nextBatch) == 0):
            choices[0] = default
        elif((len(issueNames)>0)&(level<stopLevel)&(len(nextBatch)>=3)):
            choices[0] = constructDecisionTree(copy.deepcopy(nextBatch), copy.deepcopy(issueNames), level+1, stopLevel)
        else:
            choices[0] = findMore(labelGroups)

    if((bestEntropies[1] == 0)&(len(bestEntropies[3][1])>0)):
            if(bestEntropies[3][1][0] == '0'):
                choices[1] = '0'
            else:
                choices[1] = '1'
    else:
        nextBatch = []
        for x in IDgroups[1]:
            del voters[x][bestSplit[1]]
            nextBatch.append(voters[x])
        
        if(len(nextBatch) == 0):
            choices[1] = default
        elif((len(issueNames)>0)&(level<stopLevel)&(len(nextBatch)>=3)):
            choices[1] = constructDecisionTree(copy.deepcopy(nextBatch), copy.deepcopy(issueNames), level+1, stopLevel)
        else:
            choices[1] = findMore(labelGroups[1])
    
    if((bestEntropies[2] == 0)&(len(bestEntropies[3][2])>0)):
            if(bestEntropies[3][2][0] == '0'):
                choices[2] = '0'
            else:
                choices[2] = '1'
    else:
        nextBatch = []
        for x in IDgroups[2]:
            del voters[x][bestSplit[1]]
            nextBatch.append(voters[x])
        
        if(len(nextBatch) == 0):
            choices[2] = default#Chosing a default if there aren't any options for this choice
        elif((len(issueNames)>0)&(level<stopLevel)&(len(nextBatch)>=3)):
            choices[2] = constructDecisionTree(copy.deepcopy(nextBatch), copy.deepcopy(issueNames), level+1, stopLevel)
        else:
            choices[2] = findMore(labelGroups[2])


    


    return choices;


def testIndividual(decisionTree, indiv, billNames):
    answer = decisionTree
    while (answer != '0')&(answer!='1'):
        answer = answer[int(indiv[billNames.index(answer[3])])]
    return answer;



def runTestSet(decisionTree, testSet, billNames):

    guessArray = []
    for person in testSet:
        guessArray.append(testIndividual(decisionTree,person, billNames))
            
    return guessArray

def calcRandomForest(rows, ntree, billNames,billNamesForLater,trainSet, testSet, stopLevel):
    hold = copy.deepcopy(rows[1:]);

    bags = makeBags(trainSet, ntree)

    forest = [];
    guessArray = []
    for data in bags:
        decisionTree = constructDecisionTree(copy.deepcopy(data), copy.deepcopy(billNames), 0, stopLevel)
        forest.append(decisionTree)
        guessArray.append(runTestSet(decisionTree, testSet, billNamesForLater))

    trueGuesses = majorityVote(guessArray)
    affiliation = gatherParty(testSet)
    accuracy, precision, recall, fScore  = calculateMetrics(trueGuesses, affiliation)
    return accuracy, precision, recall, fScore 

def generateProportions(data):
    parties = gatherParty(data[1:])
    zeros = 0;
    zeroArray = []
    oneArray = []
    for x in range(len(parties)):
        if(parties[x] == 0):
            zeros +=1;
            zeroArray.append(data[x+1])
        else:
            oneArray.append(data[x+1])
    return zeros/len(parties), (len(parties)-zeros)/len(parties), zeroArray, oneArray



def generateFolds(data,k):
    zeroProp, oneProp, zeroArray, oneArray = generateProportions(data)#Use these to form the folds with proper proportions

    zeroCountPer = math.floor(len(data)*zeroProp/k)
    oneCountPer = math.floor(len(data)*oneProp/k)
    folds = []
    for x in range(k-1):
        fold = []
        for zero in range(zeroCountPer):
            fold.append(zeroArray[random.randint(0,len(zeroArray)-1)])
            zeroArray.remove(fold[len(fold)-1])
        for one in range(oneCountPer):
            fold.append(oneArray[random.randint(0,len(oneArray)-1)])
            oneArray.remove(fold[len(fold)-1])
        fold = sklearn.utils.shuffle(fold)
        folds.append(fold)
    fold = []
    for zero in range(len(zeroArray)-1):
        fold.append(zeroArray[zero])
    for one in range(len(oneArray)-1):
        fold.append(oneArray[one])
    fold = sklearn.utils.shuffle(fold)
    folds.append(fold)
    return folds


def mergeFolds(folds, testFoldIndex):
    trainSet = []
    testSet = folds[testFoldIndex]

    for fold in range(len(folds)):
        if(fold != testFoldIndex):
            for x in folds[fold]:
                trainSet.append(x)
    
    return trainSet, testSet
##########
k = 10
stopLevel = 10
ntreeArray = [1,5,10,20,30,40,50]
accuracies = []
precisions = []
recalls = []
fScores = []
for x in ntreeArray:
    ntree = x
    forestAverages = []
    forestPrecision = []
    forestRecall = []
    forestFScore = []
    rows = openFile("C:/Users/mike1/Documents/GitHub/Random-Forest/house_votes_84.csv")#Put the filepath for house_votes_84.csv here
    billNamesForLater = copy.deepcopy(rows[0][:len(rows[0])-1])
    billNames =copy.deepcopy(rows[0][:len(rows[0])-1]);

    folds = generateFolds(rows,k)

    for foldCount in range(len(folds)-1):
        trainSet, testSet = mergeFolds(copy.deepcopy(folds), copy.deepcopy(foldCount))

        accuracy, precision, recall, fScore  = calcRandomForest(copy.deepcopy(rows), ntree, copy.deepcopy(billNames), copy.deepcopy(billNamesForLater), copy.deepcopy(trainSet), copy.deepcopy(testSet), stopLevel)
        forestAverages.append(accuracy)
        forestPrecision.append(precision)
        forestRecall.append(recall)
        forestFScore.append(fScore)
    accuracy = statistics.mean(forestAverages)
    precision = statistics.mean(forestPrecision)
    recall = statistics.mean(forestRecall)
    fScore = statistics.mean(forestFScore)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    fScores.append(fScore)

plt.plot(ntreeArray, accuracies)
plt.xlabel("nTree")
plt.ylabel("accuracy")
plt.show()

plt.plot(ntreeArray, precisions)
plt.ylabel("precision")
plt.xlabel("nTree")
plt.show()

plt.plot(ntreeArray, recalls)
plt.ylabel("recall")
plt.xlabel("nTree")
plt.show()
plt.plot(ntreeArray, fScores)
plt.ylabel("fScore")
plt.xlabel("nTree")
plt.show()


