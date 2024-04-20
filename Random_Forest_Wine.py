import csv;
import sklearn;
import sklearn.utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statistics
import math
import pandas as pd
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

def selectFactors(factors, lables):
    M = math.ceil(math.sqrt(len(factors)))
    factorsHold = copy.deepcopy(factors);
    labelHold = copy.deepcopy(lables)
    factorsSubset = []
    lablesSubset = []
    for x in range(M):
        choice = random.randint(0,len(factorsHold)-1)
        factorsSubset.append(copy.deepcopy(factorsHold[choice]))
        lablesSubset.append(copy.deepcopy(labelHold[choice]))
        factorsHold.remove(factorsHold[choice])
        labelHold.remove(labelHold[choice])

    return factorsSubset, lablesSubset

def majorityVote(guessArray):
    trueguesses = []
    # print(len(guessArray))
    # print(len(guessArray[0]))
    for x in range(len(guessArray[0])):
        oneCount = 0;
        twoCount = 0;
        threeCount = 0;
        for array in guessArray:
            if(array[x]== '1'):
                oneCount +=1
            if(array[x]=='2'):
                twoCount +=1
            if(array[x] =='3'):
                threeCount +=1

        if(max(oneCount,twoCount,threeCount)==oneCount):
            trueguesses.append(1)
        elif(max(oneCount,twoCount,threeCount)==twoCount):
            trueguesses.append(2)
        elif(max(oneCount,twoCount,threeCount)==threeCount):
            trueguesses.append(3)
        else:
            trueguesses.append(random.randint(1,3))
    return trueguesses

def calculateMetrics(guesses, truth):
    valueArray = [[0,0,0],[0,0,0],[0,0,0]] #Make first [] predicted (guesses), and second truth
    for x in range(len(guesses)):
        valueArray[guesses[x]-1][truth[x]-1] += 1
    accuracy = (valueArray[0][0] + valueArray[1][1] + valueArray[2][2])/len(guesses)

    if((valueArray[0][0]+valueArray[0][1]+valueArray[0][2])==0):
        precision1 =0
    else:
        precision1 = (valueArray[0][0]/(valueArray[0][0]+valueArray[0][1]+valueArray[0][2]))
    if((valueArray[1][1]+valueArray[1][0]+valueArray[1][2]==0)):
        precision2 =0
    else:
        precision2 = (valueArray[1][1]/(valueArray[1][1]+valueArray[1][0]+valueArray[1][2]))
    if((valueArray[2][2]+valueArray[2][1]+valueArray[2][0]) == 0):
        precision3 =0
    else:
        precision3 = (valueArray[2][2]/(valueArray[2][2]+valueArray[2][1]+valueArray[2][0]))
    precision = statistics.mean([precision1,precision2,precision3])

    if((valueArray[0][0]+valueArray[1][0]+valueArray[2][0])==0):
        recall1 = 0;
    else:
        recall1 =(valueArray[0][0]/(valueArray[0][0]+valueArray[1][0]+valueArray[2][0])) #tp + fn
    if((valueArray[1][1]+valueArray[0][1] + valueArray[2][1])==0):
        recall2 =0
    else:
        recall2 = (valueArray[1][1]/(valueArray[1][1]+valueArray[0][1] + valueArray[2][1]))
    if((valueArray[2][2] + valueArray[0][2] + valueArray[1][2])==0):
        recall3=0
    else:
        recall3 = (valueArray[2][2]/(valueArray[2][2] + valueArray[0][2] + valueArray[1][2]))
    
    recall = statistics.mean([recall1,recall2,recall3])

    if((precision==0)&(recall==0)):
        fScore = 0;
    else:
        fScore = 2*(precision*recall)/(precision+recall)
    return accuracy, precision, recall, fScore

def gatherCategory(list):
    cats = []
    for wine in list:
        cats.append(int(wine[0]))
    return cats

def sortlables(factorPoints):
    higher = []
    lower = []
    intArray = []
    for x in factorPoints:
        intArray.append(float(x))
    average = statistics.mean(intArray);
    if(len(intArray)>0):
        for x in range(len(factorPoints)):
            if(float(factorPoints[x])>=average):
                higher.append(x)
            else:
                lower.append(x)
        
    return [higher, lower], average;

def calculateEntropyBase(total):
    entropies = []
    for branch in total:
        case3 = 0;
        case1 = 0;
        case2 = 0;
        if(len(branch)>0):
            for x in branch:
                if(x == '3'):
                    case3 = case3+1;
                if(x == '1'):
                    case1 = case1+1;
                if(x == '2'):
                    case2 = case2+1;
                
            
            frac3 = case3/len(branch);
            frac1 = case1/len(branch);
            frac2 = case2/len(branch);

            if(frac3 ==0):
                frac3Ent = 0;
            else:
                frac3Ent = -frac3*math.log(frac3,3);
            if(frac1 ==0):
                frac1Ent = 0;
            else:
                frac1Ent = -frac1*math.log(frac1,3);
            
            if(frac2 ==0):
                frac2Ent = 0;
            else:
                frac2Ent = -frac2*math.log(frac2,3);
    
            entropy = frac3Ent + frac1Ent + frac2Ent
            entropies.append(entropy)

        else:
            entropies.append(0);
    return entropies

def calculateEntropy(total):
    entropies = []
    for branch in total:
        if(len(branch)>0):
            case1 = 0;
            case2 = 0;
            case3 = 0;

            for x in branch:
                if(x == '1'):
                    case1 +=1
                if(x == '2'):
                    case2 +=1
                if(x == '3'):
                    case3 +=1
            frac3 = case3/len(branch);
            frac1 = case1/len(branch);
            frac2 = case2/len(branch);

            if(frac3 ==0):
                frac3Ent = 0;
            else:
                frac3Ent = -frac3*math.log(frac3,3);
            if(frac1 ==0):
                frac1Ent = 0;
            else:
                frac1Ent = -frac1*math.log(frac1,3);
            
            if(frac2 ==0):
                frac2Ent = 0;
            else:
                frac2Ent = -frac2*math.log(frac2,3);

            entropy = frac3Ent + frac1Ent + frac2Ent
            entropies.append(entropy)

        else:
            entropies.append(0);
    return entropies

def splitIntoIssues(wines):#keep a key of lables to keep track of which array is which
    allFactors = []
    for x in range(len(wines[0])):#This should be the number of issues remaining in the array
        factor = []
        for y in range(len(wines)):
            factor.append(wines[y][x]);
        allFactors.append(factor)

    return allFactors;#can then calculate the entropy of each of these issues

def findMore(array):
    threeCount = 0;
    oneCount = 0;
    twoCount = 0
    for x in array:
        if(x == '3'):
            threeCount = threeCount + 1
        if(x == '1'):
            oneCount = oneCount + 1
        if(x == '2'):
            twoCount += 1
    if(max(oneCount,twoCount,threeCount) == int(oneCount)):
        return '1'
    elif(max(oneCount,twoCount,threeCount) == int(twoCount)):
        return '2'
    else:
        return '3'
    
def findMoreOpp(array):
    return findMore(array)

def calcualteInformationGain(proportions, entropies, baseGroupEntropy):#proportions is the ratio of people in that selection compared to the whole group beforehand
    infoGain =  baseGroupEntropy - (proportions[0]*entropies[0] + proportions[1]*entropies[1])
    return infoGain

def findLabels(group, lables):
    labledGroup = []    
    for x in group:
        hold = []
        for y in x:
            hold.append(lables[int(y)])
        labledGroup.append(hold)
    return labledGroup

def constructDecisionTree(wines, factorNames, level, stopLevel):
    reorganized = splitIntoIssues(wines);
    factors = copy.deepcopy(reorganized[1:])
    factorSubset, factorNameSubset = selectFactors(factors, factorNames);

    lables = copy.deepcopy(reorganized[0]);
    bestSplit = [-1, -1, ""]
    bestEntropies = [-1,-1]
    if(len(factorNameSubset)==1):
        fullGroupEntropy = calculateEntropy([lables])
        IDgroups, average = sortlables(factorSubset[0]);
        labelGroups = findLabels(IDgroups, lables)
        entropies = calculateEntropy(labelGroups);
        proportions = [len(IDgroups[0])/len(wines), len(IDgroups[1])/len(wines)]
        infoGain = calcualteInformationGain(proportions, entropies, fullGroupEntropy[0])


        if(infoGain>bestSplit[0]):
            bestEntropies = [entropies[0],entropies[1]]
            bestSplit = [infoGain, factorNames.index(factorNameSubset[0]), factorNameSubset[0], average];
    else:
        for x in range(len(factorSubset)):
            fullGroupEntropy = calculateEntropyBase([lables])
            IDgroups, average = sortlables(factorSubset[x]);
            labelGroups = findLabels(IDgroups, lables)
            entropies = calculateEntropy(labelGroups);
            proportions = [len(IDgroups[0])/len(wines), len(IDgroups[1])/len(wines)]
            infoGain = calcualteInformationGain(proportions, entropies, fullGroupEntropy[0])


            if(infoGain>bestSplit[0]):
                bestEntropies = [entropies[0],entropies[1], copy.deepcopy(labelGroups)]
                bestSplit = [infoGain, factorNames.index(factorNameSubset[x]), factorNameSubset[x], average];



    #######AFTER THE BEST SPLIT HAS BEEN FOUND
    
    choices = [-1,-1, bestSplit[2], bestSplit[3]]
    factorNames.remove(bestSplit[2])
    default = findMore(lables)
    if((bestEntropies[0] == 0)&(len(bestEntropies[2][0])>0)):
        if(bestEntropies[2][0][0] == '1'):
            choices[0] = '1'
        elif(bestEntropies[2][0][0] == '2'):
            choices[0] = '2'
        else:
            choices[0] = '3'
    else:
        nextBatch = []
        for x in IDgroups[0]:
            del wines[x][bestSplit[1]+1]
            nextBatch.append(wines[x])

        if(len(nextBatch) == 0):
            choices[0] = default
        elif((len(factorNames)>0)&(level<stopLevel)&(len(nextBatch)>=3)):
            choices[0] = constructDecisionTree(copy.deepcopy(nextBatch), copy.deepcopy(factorNames), level+1, stopLevel)
        else:
            choices[0] = findMore(labelGroups[0])

    if((bestEntropies[1] == 0)&(len(bestEntropies[2][1])>0)):
        if(bestEntropies[2][1][0] == '1'):
            choices[1] = '1'
        elif(bestEntropies[2][1][0] == '2'):
            choices[1] = '2'
        else:
            choices[1] = '3'
    else:
        nextBatch = []
        for x in IDgroups[1]:
            del wines[x][bestSplit[1]+1]
            nextBatch.append(wines[x])
        
        if(len(nextBatch) == 0):
            choices[1] = default
        elif((len(factorNames)>0)&(level<stopLevel)&(len(nextBatch)>=3)):
            choices[1] = constructDecisionTree(copy.deepcopy(nextBatch), copy.deepcopy(factorNames), level+1, stopLevel)
        else:
            choices[1] = findMore(labelGroups[1])

    return choices;


def testIndividual(decisionTree, indiv, categories):
    answer = decisionTree
    while (answer != '1')&(answer!='2')&(answer!='3'):
        guess = random.randint(0,1)
        if(float(indiv[int(categories.index(answer[2]))+1])>answer[3]):
            answer = answer[0]
        elif(float(indiv[int(categories.index(answer[2]))+1])<answer[3]):
            answer = answer[1]
        else:
            answer = answer[guess]
    return answer;



def runTestSet(decisionTree, testSet, categories):

    guessArray = []
    for person in testSet:
        guessArray.append(testIndividual(decisionTree,person, categories))
            
    return guessArray

def calcRandomForest(rows, ntree, categories,categoriesForLater,trainSet, testSet, stopLevel):#This will make 1 forest using 
    hold = copy.deepcopy(rows[1:]);

    bags = makeBags(copy.deepcopy(trainSet), ntree)

    forest = [];
    guessArray = []
    for data in bags:
        decisionTree = constructDecisionTree(copy.deepcopy(data), copy.deepcopy(categories), 0, stopLevel)
        forest.append(decisionTree)
        guessArray.append(runTestSet(decisionTree, testSet, categoriesForLater))

    trueGuesses = majorityVote(guessArray)
    affiliation = gatherCategory(testSet)#???s
    accuracy, precision, recall, fScore  = calculateMetrics(trueGuesses, affiliation)
    return accuracy, precision, recall, fScore 

def generateProportions(data):
    cats = gatherCategory(data[1:])
    ones = 0;
    twos = 0;
    threes = 0;
    oneArray = []
    twoArray = []
    threeArray = []
    for x in range(len(cats)):
        if(cats[x] == 1):
            ones +=1;
            oneArray.append(data[x+1])
        if(cats[x] == 2):
            twos += 1;
            twoArray.append(data[x+1])
        if(cats[x] == 3):
            threes += 1;
            threeArray.append(data[x+1])

    return ones/len(cats), twos/len(cats), threes/len(cats), oneArray, twoArray, threeArray



def generateFolds(data,k):
    oneProp, twoProp, threeProp, oneArray, twoArray, threeArray = generateProportions(data)#Use these to form the folds with proper proportions

    oneCountPer = math.floor(len(data)*oneProp/k)
    twoCountPer = math.floor(len(data)*twoProp/k)
    threeCountPer = math.floor(len(data)*threeProp/k)
    folds = []
    for x in range(k-1):
        fold = []
        for one in range(oneCountPer):
            fold.append(oneArray[random.randint(0,len(oneArray)-1)])
            oneArray.remove(fold[len(fold)-1])
        for two in range(twoCountPer):
            fold.append(twoArray[random.randint(0,len(twoArray)-1)])
            twoArray.remove(fold[len(fold)-1])
        for three in range(threeCountPer):
            fold.append(threeArray[random.randint(0,len(threeArray)-1)])
            threeArray.remove(fold[len(fold)-1])
        fold = sklearn.utils.shuffle(fold)
        folds.append(fold)
    fold = []
    for one in range(len(oneArray)-1):
        fold.append(oneArray[one])
    for two in range(len(twoArray)-1):
        fold.append(twoArray[two])
    for three in range(len(threeArray)-1):
        fold.append(threeArray[three])

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
    rows = openFile("C:/Users/mike1/Documents/Coding Projects/CS589 Machine Learning/Homework 3/datasets/hw3_wine.csv")#Put the filepath for house_votes_84.csv here
    categoriesForLaterForLater = copy.deepcopy(rows[0][1:len(rows[0])])
    categories =copy.deepcopy(rows[0][1:len(rows[0])]);

    folds = generateFolds(rows,k)

    for foldCount in range(len(folds)):
        trainSet, testSet = mergeFolds(copy.deepcopy(folds), copy.deepcopy(foldCount))

        accuracy, precision, recall, fScore  = calcRandomForest(copy.deepcopy(rows), ntree, copy.deepcopy(categories), copy.deepcopy(categoriesForLaterForLater), copy.deepcopy(trainSet), copy.deepcopy(testSet), stopLevel)
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
