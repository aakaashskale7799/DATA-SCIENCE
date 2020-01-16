import pandas as pd
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("train.csv")
bata=pd.read_csv("test.csv")
print("==========================")
print(data)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
df=pd.DataFrame(data, index=None)
bf=pd.DataFrame(bata, index=None)
#print(df)
##print(test)
#print(df["Age"].isnull())
#print(df.isnull().sum())
median=bf["Age"].mode()
median=df["Age"].mode()
#print(median)
fill=df["Age"].fillna(median)
fill=bf["Age"].fillna(median)

bata["age"]=fill
data["age"]=fill
bata
#print(data)
#print(fill)
#print(sdata.head())
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@22222222")
inputs=data.drop(["Survived","Age","Name","Ticket","Cabin","Embarked"], axis="columns", index=None)
binputs=bata.drop(["Age","Name","Ticket","Cabin","Embarked"], axis="columns", index=None)

print(inputs)
print("==========================")
target=data["Survived"]
#print(target)
print("now label")

PclassL=LabelEncoder()
SexL=LabelEncoder()
AgeL=LabelEncoder()
FareL=LabelEncoder()
SibSpL=LabelEncoder()
ParchL=LabelEncoder()

bPclassL=LabelEncoder()
bSexL=LabelEncoder()
bAgeL=LabelEncoder()
bFareL=LabelEncoder()
bSibSpL=LabelEncoder()
bParchL=LabelEncoder()


print("lable done")
inputs["Tclass"]=PclassL.fit_transform(inputs["Pclass"])
inputs["TSex"]=SexL.fit_transform(inputs["Sex"])
inputs["TAge"]=AgeL.fit_transform(inputs["age"])
inputs["TFare"]=FareL.fit_transform(inputs["Fare"])
inputs["TSibSp"]=SibSpL.fit_transform(inputs["SibSp"])
inputs["TParch"]=ParchL.fit_transform(inputs["Parch"])

binputs["Tclass"]=PclassL.fit_transform(binputs["Pclass"])
binputs["TSex"]=SexL.fit_transform(binputs["Sex"])
binputs["TAge"]=AgeL.fit_transform(binputs["age"])
binputs["TFare"]=FareL.fit_transform(binputs["Fare"])
binputs["TSibSp"]=SibSpL.fit_transform(binputs["SibSp"])
binputs["TParch"]=ParchL.fit_transform(binputs["Parch"])

inputs_n=inputs.drop(["Pclass","Parch","SibSp","PassengerId","Sex","age","Fare",],axis="columns")
binputs_n=binputs.drop(["Pclass","Parch","SibSp","PassengerId","Sex","age","Fare"],axis="columns")

print(inputs_n.head())
print("======================here test data=========================")

print(binputs_n.head())
"============classifier============================"
"=================================="

#from sklearn import RandomForestRegressor
from sklearn.svm import SVC
model=SVC()
model.fit(inputs_n,target)
sc=model.score(inputs_n,target)
print("===========here is your accuracy score=========================")
print(sc)
"""print("===========here is your prediction=========================")
Tclass=input("Enter Tclass (0-2) : ")
TSex=input("Enter TSex 1-male/0-female: ")
TAge=input("Enter TAge 0- 551 : ")
TFare=input("Enter TFare 0-247 : ")
TSibSp=input("Enter TSibSp 0-6 : ")
TParch=input("Enter TParch 0-6 : ")

pre=model.predict([[Tclass,TSex,TAge,TFare,TSibSp,TParch]])
#pre=model.predict([[1,1,0,55,200,1]])max(inputs_n["TSex"])
print("predict",pre)

#import pickle
#filename="finalmodel.pkl"   
#pickle.dump(model, open(filename,"test"))

"""
binputs_n
ypred=model.predict(binputs_n)
ypred
for i in ypred:
    if i>0.5:
        i=1
    else:
        i=0
ypred
fpred=pd.DataFrame(ypred)
idt=pd.read_csv("test.csv")
datasets=pd.concat([idt["PassengerId"],fpred],axis=1)
datasets.columns=["PassengerId","Survived"]
datasets.to_csv("upload30dec19SVM.csv",index=False)






"------------------------=========================="
"""inputs_n.shape
binputs_n.shape


finaldf=pd.concat([inputs_n,binputs_n],axis=0)
finaldf.shape
inputs_n.head()

finaldf.head()"""
