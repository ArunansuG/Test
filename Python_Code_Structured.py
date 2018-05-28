# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:32:47 2017

@author: arunansu.gorai
"""

#How to save a dataset in py format
#libraries

import numpy as np
import pandas as pd
from numpy.random import randn
from pandas import Series, DataFrame, Index
from IPython.display import Image
import os
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from scipy.stats import mode
import re
import matplotlib
import ctypes

#Read a dataset
file = pd.read_excel("C:/Users/arunansu.gorai/AG/Projects/KC_Data/Q&I_report/Dir+ Changes 140101_170630.xlsx")
data = pd.read_csv(r'C:\Users\arunansu.gorai\AG\Projects\Sas_code_data_for_python\dummydata.csv', delimiter=',', skipinitialspace=True)
optek=pd.read_sas(r"C:\Users\arunansu.gorai\AG\Projects\Sas_code_data_for_python\optek_final_data.sas7bdat", format='SAS7BDAT')

print(optek.iloc[0])
print(optek.iloc[0:2])
print(optek[0:3])


#read a particular sheet in a dataset
filepath=r"C:\Users\arunansu.gorai\AG\Projects\reporting_team_migration_to_python"
filename="Global Team Leader Changes for ALF 8-14-17.xlsx"
oldsheet="08-07-17"

olddata=pd.read_excel(os.path.join(filepath,filename),sheetname=oldsheet)

file.describe()
#Export
tips.to_csv('tips2.csv')

filepath=r"C:\Users\arunansu.gorai\AG\Projects\reporting_team_migration_to_python"
tl_changes.to_excel(os.path.join(filepath,'TL Changes.xlsx'),index=False)

#Data understanding
print(optek.size, optek.shape, optek.ndim)

optek.info()

optek.describe()
optek.describe(percentiles=[.05,.1,.25])
optek.describe(include='all')

optek.head()
optek.tail()

optek[['OPTEK_Headcount', 'calculated_headcount']].head(4)

optek.geo_unit.unique()
#or
list(optek['geo_unit'].unique())


optek.career_level_seg.unique()

optek.mean()
optek.sum()
optek.quantile()


#missing values
for col_name in optek.columns:
   print (col_name, end="---->")
   print (sum(optek[col_name].isnull()))
#or
out=optek.isnull().sum()
#or
optek.info()

#checking frequency
optek.calculated_HEADCOUNT.value_counts() #sorted by freq, by default
optek.career_level_seg.value_counts().sort_index() # sorted by index
optek.Pay_Scale_Group1.value_counts(dropna=False) # including missing values

#Find type of the data/variable etc
type(optek)
type(optek.calculated_headcount)

#Renaming a variable
optek=optek.rename(columns={'calculated_HEADCOUNT':'calculated_headcount'})
#Remember while renaming a variable, if the variable name is wrong or doesn't exist python will not throw any error

# gives total count,gives uniqe count of the mentioned variable
optek.career_level_seg.describe()
optek.Pay_Scale_Group1.describe()


#Calculation of mode and Imputation
#Imputation by mean mode median etc
#import a function to determine the mode
from scipy.stats import mode
mode(df['year'])
mode(data['year']).mode[0] #Incase of multiple mode taking the first one
  
df['year'].fillna(mode(df['year']).mode[0], inplace=True)


#sorting data

#removing duplicates and creating a new data with duplicate values

#join

#group by or pivot table


#Writing in console: interactive
name=input("what is your name: ")
print(name)

#For loop


for country in ["Ind", "SL", "China"]:
    print(country)
    print(len(country))


#Missing Values


#reverse

a=[1,2,8,9]
a[::-1]
[x for x in reversed(a)]
a.reverse()

ra=[]
for i in reversed(a):
    ra.append(i)
ra

cnt=len(a)
ra=[]
for i in range(cnt):
    ra.append(a[cnt-1-i])
ra

cnt=len(a)
ra=[]
while cnt !=0:
	  ra.append(a[cnt-1])
	  cnt=cnt-1
ra

#reading & writing a file

file = open("newfile.txt", "w")
file.write("This has been written to a file")
file.close()

file = open("newfile.txt", "r")
print(file.read())
file.close()

[i**3 for i in range(5)]
[i**2 for i in range(10) if i**2 % 2 == 0]

print("{0}{1}{0}".format("abra", "cad"))


#Basic of Python : Skillsoft course

def main():
    choices=dict(alex="A", Jane="B", tim="C")
    print (choices["alex"])
main()

def x():
    data="i want to break"
    for char in data:
 
        if char=='b': 
            break
        print(char)
    
x()

squares=[y for y in(y**2 for y in range(10) if y%3==0)]

s1 = "String"
print(s1.upper())

s2 = s1.lower()
print(s2)

olddata['User Name']=olddata['User Name'].str.upper()

#Strings are Immutable 

s1 = "I want Apple"
s2= 'u' + s1[2:]
print(s1)
print(s2)


import urllib.request
page = urllib.request.urlopen("http://www.google.com")
print (page.read())

import webbrowser
ie = webbrowser.get(webbrowser.iexplore)
ie.open('https://ext.connect.kcc.com/vpn/index.html')

ie.open('https://google.com')

import subprocess
subprocess.call([r'C:\Program Files (x86)\CRYPTOCard\CRYPTOCard Software Tools\bin\Authenticator.exe'])


import os
os.startfile(r'C:\Users\arunansu.gorai\AG\ID card Belt Clip .jpg')

user_name
Password


DataFrame.drop_duplicates(subset=None, keep='first', inplace=False)
#first,last, False


raw_data = {'first_name': ['Jason', 'Jason', 'Jason', 'Tina', 'Jake', 'Jake'],
        'last_name': ['Miller', 'Miller', 'Statham', 'Ali', 'Milner', 'Cooze'],
        'age': [42, 42, 36, 24, 73, 78],
        'preTestScore': [4, 4, 31, 2, 3, 4],
        'postTestScore': [25, 25, 25, 62, 70, 23]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'preTestScore', 'postTestScore'])

df1 = df.drop_duplicates()
df1
df2 = df.drop_duplicates(['first_name','postTestScore'], keep='last')
df2

#group by and/or take max 
df.sort_values('first_name', ascending=False).drop_duplicates('postTestScore').sort_index()

df.groupby('postTestScore', as_index=False).max()


# Create a new variable - multiple condition - conditional variable
optek['calc_hc_grp']=["<20" if x < 20 else 
                      "20 - 30" if  20< x <30  else 
                      ">30" 
                      for x in optek['calculated_HEADCOUNT']]  

file['marital_grp']=['Married' if x in('married','divorced') else 
                     'Single'  if x=='single'  else 
                     'Others' 
                      for x in file['marital']]

df["Management Level"].unique()
df["Management Level"].replace(np.NaN, 'NULL', inplace=True)
df["Management Level"].unique()

df["Management_order"] = df["Management Level"].map(lambda x: 
    0 if x=="NULL" else
    1 if "S" in x else 
    2 if x=="P1" else 
    3 if x in ("P2","M1") else
    4 if x in ("P3","M2") else
    5 if x in("P4","M3") else
    6 if x in("P5","M4") else
    7 if x=="M5" else
    8 if x=="M6" else
    9 if x=="M7" else
    10) 

#To keep specific rows based on a condition
file1 = file[(file['marital']=='single') & ((file['job']!='services'))]

#To drop specific rows based on a condition
file1=file.drop(file[(file.marital=='single')].index)

#So how would you calculate the number of columns having categorical values?
(file.dtypes == object).sum()

#distinct values
file.marital.unique()
#missing values
(dir_140101_170630.isnull().sum()).Demotion 
#drop blank demotions
new_dataframe = dir_140101_170630[~dir_140101_170630.Demotion.isnull()] 

#drop / remove rows with a column having missing / blank values
df.dropna(subset=[1])


#If a row contains more than 5 missing values, 
#you decide to drop them and store remaining data in DataFrame "temp". 
temp = dir_140101_170630.dropna(axis=0, how='any', thresh=dir_140101_170630.shape[1] - 5)

#combine "married" and "divorced" in a new category "Married" . You also decide to rename "single" to "Single"

turn_dict = {'married': 'Married', 'divorced': 'Married', 'single': 'Single'}
file1.loc[:, 'marital'] = file1.marital.replace(turn_dict)

#percentage of married males in the data
(file.loc[(file.job == 'blue-collar') & (file.marital == 'married')].shape[1] / float(file.shape[0]))*100 

#How to find which cols are present in test but not in train? Assume data has already been read in DataFrames "train" & "test" respectively.

file1=file
file2=file1
file2.drop(['education','housing','loan'], inplace=True, axis=1, errors='ignore')
#??????????????????????
set(file.columns).difference(set(dir_140101_170630.columns)) 
set(file.columns.tolist()) - set(dir_140101_170630.columns.tolist()) 
set(file.columns.tolist()).difference(set(dir_140101_170630.columns.tolist()))


#categorical "Gender" values to numerical values (i.e. change M to 1 and F to 0). 
#Which of the commands would do that?

file.ix[:, 'y'] = file.y.map({'yes':1,'no':0}).astype(int) 

#How would you check if all values of "Product_ID" in test DataFrame are available in train DataFrame dataset?

set(file.marital.unique()).issubset(set(file1.marital.unique())) 

set(file1.marital.unique()).issubset(set(file.marital.unique())) 

#You decide to replace the Categorical column 'Age' by a numeric column by replacing the range with its average 
#(Example: 0-17 and 17-25 should be replaced by their averages 8.5 and 21 respectively)

raw_data = {'first_name': ['Jason', 'Jason', 'Jason', 'Tina', 'Jake', 'Jake'],
        'last_name': ['Miller', 'Miller', 'Statham', 'Ali', 'Milner', 'Cooze'],
        'age': ['42-47', '42-47', '36-41', '24-29', '73-78', '78-81'],
        'preTestScore': [4, 4, 31, 2, 3, 4],
        'postTestScore': [25, 25, 25, 62, 70, 23]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'preTestScore', 'postTestScore'])

df['age'] = df.age.apply(lambda x: np.array(x.split('-'), dtype=int).mean())

#For example, in "Ticket", the values are represented as one or two blocks separated with spaces. 
#Each block has numerical values in it, but only the first block has characters combined with numbers. 
#(eg. "ABC0 3000"). 

#Which of the following code return only the last block of numeric values? 
#(You can assume that numeric values are always present in the last block of this column)

raw_data = {'first_name': ['Jason', 'Jason', 'Jason', 'Tina', 'Jake', 'Jake'],
        'last_name': ['Miller', 'Miller', 'Statham', 'Ali', 'Milner', 'Cooze'],
        'age': ['42-47', '42-47', '36-41', '24-29', '73-78', '78-81'],
        'preTestScore': [4, 4, 31, 2, 3, 4],
        'postTestScore': ['a 25','b 25','25','hdh 62',' 70','hs 238999']}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'preTestScore', 'postTestScore'])

df.postTestScore.str.split(' ').str[-1] 

#You decide to fill missing "Age" values by mean of all other passengers of the same gender.
# Which of the following code will fill missing values for all passengers by the above logic?

raw_data = {'first_name': ['Mr. Jason', 'Jason', 'Mr. Jason', 'Tina', 'Mr. Jake', 'Jake'],
        'last_name': ['Miller', 'Miller', 'Statham', 'Ali', 'Milner', 'Cooze'],
        'age': [42, 42, 36,73, np.NaN,np.NaN],
        'age_group': ['42-47', '42-47', '36-41', '24-29', '73-78', '78-81'],
        'gender':['M','F','F','F','M','F'],
        'preTestScore': [4, 4, 31, 2, 3, 4],
        'postTestScore': ['a 25','b 25','25','hdh 62',' 70','hs 238999'],
        'location': ['S','S','S','M','M','S']}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age','age_group','gender', 'preTestScore', 'postTestScore','location'])

df['age'] = df.groupby('gender').transform(lambda x: x.fillna(x.mean())).age 

#how many females embarked from location 'S'?

df.loc[(df.location == 'S') & (df.gender == 'F')].shape[0] 

#calculate how many values in column "Name" have "Mr." contained in them
(df.first_name.str.find('Mr.')==False).sum() 

#create a new column "Missing_age" and put the right values in it (i.e. if "age_missing" then 1 else 0)

df['Missing_age'] = df.age.isnull().astype(int)
#or
df['Missing_age'] = df.age.isnull() == False

#the dataset does not contain headers. Inspite of this, you know what are the appropriate column names.
# How would you read the the dataframe by specifying the column names

pd.read_csv(r"C:\Users\arunansu.gorai\AG\Knowledge Base\Python\Training_Data.csv", header=None, names=['index' ,'name' ,'age', 'location'])

#change the datatype of "Item_Fat_Content" column from "object" to "category"
df['Missing_age'] = df['Item_Fat_Content'].astype('category') 

#find all values in "Item_Identifier" that starts with "F"

df.first_name.str.startswith('M') 

#convert the float values in column "Item_MRP" to integer values
df['preTestScore'] = df.preTestScore.astype(int)

#corr
df.age.corr(df.preTestScore, method='pearson') 
df.age.corr(df.preTestScore) 

#create a pivot table of 'Marital.Status' vs 'Occupation' and put the values.
#Create the pivot table as mentioned above, with the aggregating function as "sum"

df.pivot_table(index='gender', columns='location', values='preTestScore', aggfunc='sum') 

#We want to start reading from the third row of the dataset
train = pd.read_csv(r'C:\Users\arunansu.gorai\AG\Knowledge Base\Python\Training_Data_2.csv', skiprows=2)


#read only the top 5 rows
train = pd.read_csv(r'C:\Users\arunansu.gorai\AG\Knowledge Base\Python\Training_Data_2.csv', nrows=5)

#count of corresponding Relationship status among all individuals, and then divide it by the 
#total data points to get the percentage and map it to the original column

#percentage
df.gender.value_counts()/df.shape[0]

#Map to the original data
df['gender_Percentage'] = df.gender.map(df.gender.value_counts()/df.shape[0]) 

#"Date_time_of_event" column and it is currently read as an object. This will restrict us to perform any date time operation on it. 
#Which command will help to convert the column "Date_time_of_event" to data type "datetime"?

raw_data = {'Date_time_of_event': ['25-08-2012 00:00', '25-08-2012 02:00', '25-08-2012 04:00'],
            'Count': [8,2,3]}
df = pd.DataFrame(raw_data, columns = ['Date_time_of_event', 'Count'])

df['Date_time_of_event'] = pd.to_datetime(df.Date_time_of_event, format="%d-%m-%Y %H:%M") 

#How would you you extract only the dates from the given "Date_time_of_event" column?
df.Date_time_of_event.dt.day 

#name of weekday the date belongs to
df.Date_time_of_event.dt.weekday_name

#no of the week the date belongs to
df.Date_time_of_event.dt.weekday

#convert timestamp to datetime
raw_data = {'TIMESTAMP': ['1408039037', '1408038611', '1408039090'],
            'TAXI_ID': [20000542,20000108,20000370]}
df = pd.DataFrame(raw_data, columns = ['TIMESTAMP', 'TAXI_ID'])

pd.to_datetime(df['TIMESTAMP'],unit='s') 

#difference between current time and column 'Date_time_of_event'.
pd.datetime.now() - df.Date_time_of_event

#replace the "Date_time_of_event" column with the first day of the month
df['Date_time_of_event'] = df.Date_time_of_event.apply(lambda x: x.replace(day=1)) 
 
#or
df['month'] = df.Date_time_of_event.dt.month
df['year'] = df.Date_time_of_event.dt.year 
df['day'] = 1 
df['Date_time_of_event'] = df.apply(lambda x:pd.datetime.strptime("{0} {1} {2}".format(x['year'],x['month'], x['day']), "%Y %m %d"),axis=1) 

#The dataset above provides every day expenses on different necessities 
#(days will be arranged in columns and expenses on necessities is in rows). 
#Provide python code to compute cummulative cost for each day. 

raw_data={'Monday':[1,2,5],'Tuesday':[2,3,6],'Wednesday':[3,4,7]}
df = pd.DataFrame(raw_data, columns = ['Monday', 'Tuesday','Wednesday'], index=['Eating','Sleeping','Singing'])

df.cumsum(axis=1) 

#==============================================================================
# #For three data sets given train,student and internship we need to merge these data sets in such a way 
# #that for train data every row must have the student details from student data and intern details from intern data. 
# #(Consider only those "Student_ID" and "Internship_ID" which are same in both the respective datasets, 
# # i.e. "student" and "internship" in comparison to "train") 
# 
# train=pd.merge(train,internship,on='Internship_ID',how='inner') 
# train=pd.merge(train,student,on='Student_ID',how='inner')
# under
# outer
# right
#==============================================================================

#remove the duplicates by keeping the first occurence only.

raw_data={'id':[1,2,2,3,3,3,4,4,4,4]}
df=pd.DataFrame(raw_data,columns=['id'])

df.drop_duplicates(subset=['id'],keep='first',inplace=False)

#Which of the following will be able to extract an e-mail adress from string of words?
string = "his email address is mailto:tom_42@gmail.com please mail him the documents"
 
match=re.findall(r"[\w._]+@[\w.]+",string) 

#drop a column = axis=1
df.drop("sleep", axis=1)

#plot data
file.job.value_counts().plot(kind='bar')

raw_data={'credit_history':['0','0','1','1'],'loan_status':['N','Y','Y','N'],'count':[82,7,180,90]}
df=pd.DataFrame(raw_data,columns=['credit_history','loan_status','count'])

#divided bar diagram
df.unstack().plot(kind='bar',stacked=True, color=['red','blue'], grid=False) 

#plot
raw_data={'temp':[9.84,9.02,9.02,9.84,9.84],'atemp':[14.395,13.635,13.635,14.395,14.395],'count':[16,40,32,13,1]}
df=pd.DataFrame(raw_data,columns=['temp','atemp','count'])

plt.scatter(df.temp,df.atemp,alpha=1,c='b',s=20) 

plt.scatter(df.temp,df.atemp,alpha=1,c=df['count'],s=20) 

df.hist(column='temp', bins=50)

pd.tools.plotting.autocorrelation_plot(df.temp) 
#join
tl_changes=pd.merge(newdata,olddata[['User Name','Previous Team Leader','Previous Team Leader User Name'
                                     ,'Previous Country','Previous Location']]
                    ,left_on='User Name',right_on='User Name',how='inner')

# sort 
df.sort_values(['a', 'b'], ascending=[True, False])

#Sorting the columns in user defined order - retain
tl_changes=tl_changes[['Employee Name','User Name','Current Team Leader','Current Team Leader User Name'
,'Previous Team Leader','Previous Team Leader User Name','Current Country'	,'Previous Country'
,'Current Location','Previous Location','Management Level']]

#Excel write - modifying excel formats

#Excelwrite
writer = pd.ExcelWriter(os.path.join(filepath,'TL Changes.xlsx'), engine='xlsxwriter')
tl_changes.to_excel(writer, sheet_name='TL Changes',index=False)
workbook  = writer.book
worksheet = writer.sheets['TL Changes']
worksheet.set_column('A:K', 18)
writer.save()

# Turn off the default header and skip one row to allow us to insert a
# user defined header.
writer = pd.ExcelWriter(os.path.join(filepath,'TL Changes.xlsx'), engine='xlsxwriter')
tl_changes.to_excel(writer, sheet_name='Sheet1', startrow=1, header=False)

# Get the xlsxwriter workbook and worksheet objects.
workbook  = writer.book
worksheet = writer.sheets['Sheet1']
worksheet.set_column('A:L', 20)
# Add a header format.
header_format = workbook.add_format({
    'bold': True,
    'text_wrap': False,
    'valign': 'top',
    'fg_color': '#D7E4BC',
    'border': 1})

# Write the column headers with the defined format.
for col_num, value in enumerate(tl_changes.columns.values):
    worksheet.write(0, col_num + 1, value, header_format)

writer.save()

#pop up window / pop up message
ctypes.windll.user32.MessageBoxW(0, "Old data is not unique by User Name", "Check Unique", 1)

import pymsgbox
pymsgbox.alert('This is an alert!', 'Title')
response = pymsgbox.prompt('What is your name?')


#checking the positions for each letter/alphabet 
a = 'Arun'
a=a.lower()
i= 1
for char in a:
    asc = ord(char)-96
    print(char,asc, i, asc*i)
    i = i+1
    
    
#Creating a code to find a score such that each cell have a score 
# = sum of multiplication of postion in the word * position of the letter in alphabet   
raw_data = pd.DataFrame({'first_name': ['jason', 'aravind', 'netaji', 'swamiji', 'ramakrishna', 'jake']})
raw_lower = raw_data['first_name'].apply(lambda x:x.lower()) 
i= 1
for names in raw_lower:
    j= 1
    for char in names:
        asc = ord(char)-96
        print(char ,asc, i, asc*i)
        if j == 1: score_str = asc*j
        else: score_str = score_str+asc*j
        j = j+1
        
    if i == 1: scores = pd.DataFrame({'score_str':score_str}, index=[0])
    else: 
        newval = pd.DataFrame({'score_str':score_str}, index=[0])
        scores = scores.append(newval, ignore_index=True)
    i = i+1

raw_data['scores'] = scores

5. List comprehension

T={ x^2: x is a natural number less than 10 }
S = [x**2 for x in range(10)]
S
v={ 2^x: x is a natural number less than 13 }
V = [2**x for x in range(13)]
V
{ x: x belongs to T and x is divisble by 2}
M = [x for x in S if x % 2 == 0]
M

{ x: x is an alphabet in word ‘MATHEMATICS’, x is a vowel }
[x for x in 'MATHEMATICS' if x in ['A','E','I','O','U']]


noprimes = [j for i in range(2, 8) for j in range(i*2, 50, i)]
primes = [x for x in range(2, 50) if x not in noprimes]
          
words = 'The quick brown fox jumps over the lazy dog'.split()

stuff = [[w.upper(), w.lower(), len(w)] for w in words]
for i in stuff:
    print i

stuff = map(lambda w: [w.upper(), w.lower(), len(w)], words)
for i in stuff:
    print i

Comparison with for loop:
    for (set of values to iterate):
        if (conditional filtering): 
            output_expression()
            
[output_expression() for(set of values to iterate) if(conditional filtering)]
 
matrix = [range(0,5), range(5,10), range(10,15)]
def eg1_for(matrix):
    flat = []
    for row in matrix:
        for x in row:
            flat.append(x)
    return flat

def eg1_lc(matrix):
    return [x for row in matrix for x in row ]

print ("Original Matrix: " + str(matrix))
print ("FOR-loop result: " + str(eg1_for(matrix)))
print ("LC result      : " + str(eg1_lc(matrix)))



def eg2_for(sentence):
    vowels = 'aeiou'
    filtered_list = []
    for l in sentence:
        if l not in vowels:
            filtered_list.append(l)
    return ''.join(filtered_list)

def eg2_lc(sentence):
    vowels = 'aeiou'
    return ''.join([ l for l in sentence if l not in vowels])
sentence = 'My address in balkum naka, ashoknagar!'
print "FOR-loop result: " + eg2_for(sentence)
print "LC result      : " + eg2_lc(sentence)



def eg3_for(keys, values):
    dic = {}
    for i in range(len(keys)):
        dic[keys[i]] = values[i]
    return dic

def eg3_lc(keys, values):
    return { keys[i] : values[i] for i in range(len(keys)) }

country = ['India', 'Pakistan', 'Nepal', 'Bhutan', 'China', 'Bangladesh']
capital = ['New Delhi', 'Islamabad','Kathmandu', 'Thimphu', 'Beijing', 'Dhaka']
print "FOR-loop result: " + str(eg3_for(country, capital))
print "LC result      : " + str(eg3_lc(country, capital))

#Subset the data wheneevr some cells of a particular column is blank
outs=pd.merge(olddata,newdata1[['KC Logon ID New']],left_on='KC Logon ID',right_on='KC Logon ID New',how='left')
outs=outs[(outs['KC Logon ID New'].isnull())] 
          
#  drop / dropping a particular column froma  dataframe 
del df['column_name']
#The best way to do this in pandas is to use drop:

df = df.drop('column_name', 1)
where 1 is the axis number (0 for rows and 1 for columns.)

#To delete the column without having to reassign df you can do:

df.drop('column_name', axis=1, inplace=True)
#Finally, to drop by column number instead of by column label, try this to delete, e.g. the 1st, 2nd and 4th columns:

df.drop(df.columns[[0, 1, 3]], axis=1)  # df.columns is zero-based pd.Index 

#Populated rows /non null rows
outs=outs[(outs['KC Logon ID New'].notnull())]
          
          
#keeping a python copy of the dataset as the file reading time willbe too much
pickle.dump(comm, open(os.path.join(path,'candidate_comm'),"wb"))
#comm = pickle.load(open(os.path.join(path,'candidate_comm'),"rb"))

pickle.dump(mcr, open(os.path.join(path,'master_candidate_report'),"wb"))
#mcr = pickle.load(open(os.path.join(path,'master_candidate_report'),"rb"))
#delete a dataset in pyhton
#del load

start_time = time.clock()
print(time.clock() - start_time, "seconds")




c=pd.DataFrame({'Gender':['M','M','F','M','F','F','O','F','M','F'],'Age':[11,11,12,11,12,13,12,12,13,14]
               ,'Weight':[12,17,19,36,24,35,65,13,35,7]
               ,'Height':[4,5,7,3,8,9,4,6,7,7]})
# No of unique values
c['Gender'].count()
c['Age'].max()
x=c['Gender'].value_counts()
c['Gender'].unique()
c['Gender'].nunique()

c.groupby(['Gender']).groups.keys()
x=len(c.groupby(['Gender']).groups['M'])

# Get the first entry for variable gender
x=c.groupby('Gender').first()
 
# Get the sum of the durations per month
x=c.groupby('Gender',as_index=False)['Age'].sum()
x=c.groupby('Gender')[['Age']].sum()
c.groupby(['Gender','Age'])['Weight'].sum()
c.groupby(['Gender','Age']).count()
c.groupby(['Gender','Age'])['Weight'].mean()
c[c['Gender'] == 'M'].groupby('Age')['Weight'].sum()
 
x=c.groupby(['Gender','Age'])['Weight'].sum() # produces Pandas Series
x=c.groupby(['Gender','Age'])[['Weight']].sum() # Produces Pandas DataFrame
x=c.groupby(['Gender','Age'], as_index=False).agg({"Weight": "sum"}) #To avoid setting this index, pass “as_index=False” to the groupby operation.

# Group the data frame by gender and age and extract a number of stats from each group
x=c.groupby(['Gender', 'Age']).agg({'Weight':'sum',     
                                    'Height': "mean"})
x=c.groupby(['Gender', 'Age']).agg({'Weight':['sum','mean']})

aggregations = {
    'Weight':'sum',
    'Height': lambda x: max(x) - 1
}
c.groupby('Gender').agg(aggregations)

# Group the data frame by month and item and extract a number of stats from each group
x=c.groupby(['Gender', 'Age'],as_index=False).agg({'Height': [min, max, sum],    
                                  'Weight': ['count',min, 'first', 'nunique']})   
#renaming group by
grouped = c.groupby('Gender').agg({"Age": [min, max, 'mean']})
grouped.columns = grouped.columns.droplevel(level=0)
grouped=grouped.rename(columns={"min": "min_age", "max": "max_age", "mean": "mean_age"})


#ravel approach of renaming
grouped = c.groupby('Gender').agg({"Age": [min, max, 'mean']}) 
# Using ravel, and a string join, we can create better names for the columns:
grouped.columns = ["_".join(x) for x in grouped.columns.ravel()]


index = pd.date_range('10/1/1999', periods=1100)
ts = pd.Series(np.random.normal(0.5, 2, 1100), index)
ts = ts.rolling(window=100,min_periods=100).mean().dropna()

key = lambda x: x.year
zscore = lambda x: (x - x.mean()) / x.std()
transformed = ts.groupby(key).transform(zscore)

compare = pd.DataFrame({'Original': ts, 'Transformed': transformed})
compare.plot()

#opening and reading a file
f = open("crime_rates.csv", "r")
data=f.read()
print(data)
rows=data.split('\n')
ten_rows = rows[0:10]
for i in ten_rows:
    print(i)
    
three_rows = ["Albuquerque,749", "Anaheim,371", "Anchorage,828"]
final_list = []
for row in three_rows:
    split_list = row.split(',')
    final_list.append(split_list)
print(final_list)


#we can use the split() method to turn a string object into a list of strings
sample = "john,plastic,joe"
split_list = sample.split(",")
# split_list is a list of _strings_: ["john", "plastic", "joe"]

three_elements=[['A',10],['B',20],['C',40]]
list=[]
for i in three_elements:
    list.append(i[0])
print(list)


f = open('crime_rates.csv', 'r')
data = f.read()
rows = data.split('\n')
print(rows[0:5])

int_crime_rates=[]
for i in rows:
    j=i.split(',')
    k=int(j[1])
    int_crime_rates.append(k)
print(int_crime_rates)
#['Albuquerque,749', 'Anaheim,371', 'Anchorage,828', 'Arlington,503', 'Atlanta,1379']
#[749, 371, 828, 503, 1379, 425, 408, 542,.....,...]

#Boolean
a=(5==5)
b=(5!=5)


#https://fivethirtyeight.com/features/there-are-922-unisex-names-in-america-is-yours-one-of-them/
#http://fivethirtyeight.com/
#https://github.com/fivethirtyeight/data/blob/master/unisex-names/unisex_names_table.csv
f = open('dq_unisex_names.csv', 'r')
names = f.read()
names_list = names.split('\n')
nested_list=[]
for c in names_list:
    comma_list=c.split(',')
    nested_list.append(comma_list)
numerical_list=[]
for i in nested_list:
    x=i[0]
    y=float(i[1])
    z=[x,y]
    numerical_list.append(z)
print(numerical_list[0:5])

#change the condition of minimum range from 100 to 1000
numerical_list[len(numerical_list)-1]

thousand_or_greater=[]
for i in numerical_list:
    if i[1]>=1000:
        thousand_or_greater.append(i)
print(thousand_or_greater[0:10])


animals = ["cat", "dog", "rabbit"]
if "cat" in animals:
    print("Cat found")
    
#or
animals = ["cat", "dog", "rabbit"]
cat_found = "cat" in animals

planet_numbers = {"mercury": 1, "venus": 2, "earth": 3, "mars": 4}
jupiter_found='jupiter' in planet_numbers
earth_found='earth' in planet_numbers



pantry = ["apple", "orange", "grape", "apple", "orange", "apple", "tomato", "potato", "grape"]
pantry_counts={}
for i in pantry:
    if i in pantry_counts:
        pantry_counts[i]+=1
    else:
        pantry_counts[i]=1
print(pantry_counts)

#IMDb movie data cleaning
f=open('movie_metadata.csv','r').read().split('\n')
movie_data=[]
for i in f:
    movie_data.append(i.split(','))
    
def feature_counter(input_lst,index,input_str,header_row=False):
    num=0
    if header_row == True:
        input_lst = input_lst[1:len(input_lst)]
    for each in input_lst:
        if each[index]==input_str:
            num+=1
    return num
num_of_us_movies=feature_counter(movie_data,6,"USA",True)

#The two main types of errors are:
#Syntax errors
#Runtime errors


def read_csv(filename):
    string_list=open(filename,'r').read().split('\n')[1:]
    final_list=[]
    for i in string_list:
        string_fields=[int(z) for z in i.split(',')]
        final_list.append(string_fields)
    return final_list
cdc_list=read_csv("US_births_1994-2003_CDC_NCHS.csv")

def calc_counts(input_lst,input_column):
    no_of_births={}
    for i in input_lst:
        grp=i[input_column]
        births=i[4]
        if grp in no_of_births:
            no_of_births[grp]+=births
        else:
            no_of_births[grp]=births
    return no_of_births
cdc_year_births=calc_counts(cdc_list,0)
cdc_month_births=calc_counts(cdc_list,1)
cdc_dom_births=calc_counts(cdc_list,2)
cdc_dow_births=calc_counts(cdc_list,3)

def max_min(input_dict):
    x=[min(input_dict),max(input_dict)]
    return x
max_min(cdc_year_births)


def calc_counts1(input_lst,index_column1,index_column2,index2_val):
    no_of_births={}
    for i in input_lst:
        if i[index_column2]==index2_val:
            grp=i[index_column1]
            births=i[4]
            if grp in no_of_births:
                no_of_births[grp]+=births
            else:
                no_of_births[grp]=births
    x=max_min(cdc_year_births)
    diff={}
    for j in range(x[0],x[1]):
        diff[j+1]=(no_of_births[j+1]>no_of_births[j])
    return diff    
x=calc_counts1(cdc_list,0,3,7)
print(x)

cdc_common=[]
ssa_common=[]
for i in cdc_list:
    if i[0] in (2000,2001,2002,2003):
        cdc_common.append(i)
for i in ssa_list:
    if i[0] in (2000,2001,2002,2003):
        ssa_common.append(i)
        
final_common=[]
for i in range(0,len(cdc_common)):
    if cdc_common[i][4]>ssa_common[i][4]:
        final_common.append(cdc_common[i])
    else:
        final_common.append(ssa_common[i])
print(final_common)


import csv
f=open('nfl.csv')
readcsv=csv.reader(f)
nfl=list(readcsv)

class Dataset:
    def __init__(self):
        self.type='csv'
dataset=Dataset()
print(dataset.type)

class Dataset:
    def __init__(self,data):
        self.data = data
nfl_data=open('nfl.csv','r')
nfl_data=list(csv.reader(nfl_data))
nfl_dataset=Dataset(nfl_data)
dataset_data=nfl_dataset.data

class Dataset:
    def __init__(self, data):
        self.data = data       
    # Your method goes here
    def print_data(self,num_rows):
        print(self.data[:num_rows])
nfl_dataset=Dataset(nfl_data)
nfl_dataset.print_data(5)

class Dataset:
    def __init__(self, data):
        self.header = data[0]
        self.data = data[1:]
    
    def column(self, label):
        if label not in self.header:
            return None
        
        index = 0
        for idx, element in enumerate(self.header):
            if label == element:
                index = idx
        
        column = []
        for row in self.data:
            column.append(row[index])
        return column
    
    # Add your count unique method here
    def count_unique(self,label):
        unique_results=set(self.column(label))
        count=len(unique_results)
        return count

nfl_dataset = Dataset(nfl_data)
total_years = nfl_dataset.count_unique('year')

#data and functions associated with specific class then it is called attribute and methods. 
#Method - A function associated with a class

class Employee:
    pass
#difference betweena class and an instance of a class
#class is a blueprint of creating instances
emp_1=Employee() # instances
emp_2=Employee() # instances
print(emp_1)
print(emp_2)
#manual allocation of instances
emp_1.fname='A'
emp_1.lname='G'

emp_2.fname='G'
emp_2.lname='A'

class Employee:
    def __init__(self,fname,lname,pay):
        self.fname=fname
        self.lname=lname
        self.pay=pay
        self.email=fname+'.'+lname+'@gmail.com'
    def fullname(self):
        return '{} {}'.format(self.fname,self.lname)
    
emp_1=Employee('a','g',100000)
emp_2=Employee('g','a',200000)
print(emp_1)
print(emp_2)

print(emp_1.email)
print(emp_2.pay)
print(emp_1.fullname)
print(emp_1.fullname())

print(Employee.fullname(emp_1))

class Employee:
    def __init__(self,fname,lname,pay):
        self.fname=fname
        self.lname=lname
        self.pay=pay
        self.email=fname+'.'+lname+'@gmail.com'
    def fullname(self):
        return '{} {}'.format(self.fname,self.lname)
    def apply_raise(self):
        self.pay=int(self.pay*1.05)
emp_1=Employee('a','g',100000)
emp_2=Employee('g','a',200000)

print(emp_1.pay)
emp_1.apply_raise()
print(emp_1.pay)

class Employee:
    raise_amount=1.05
    def __init__(self,fname,lname,pay):
        self.fname=fname
        self.lname=lname
        self.pay=pay
        self.email=fname+'.'+lname+'@gmail.com'
    def fullname(self):
        return '{} {}'.format(self.fname,self.lname)
    def apply_raise(self):
        #self.pay=int(self.pay*raise_amount) #will not work
        self.pay=int(self.pay*self.raise_amount) #will work - assign individual values of raise_mount
        #self.pay=int(self.pay*Employee.raise_amount) #will work - assign the class(Employee) level raise amount
emp_1=Employee('a','g',100000)
emp_2=Employee('g','a',200000)

print(emp_1.pay)
emp_1.apply_raise()
print(emp_1.pay)

print(emp_1.raise_amount)
print(emp_2.raise_amount)
print(Employee.raise_amount)

print(emp_1.__dict__)
print(Employee.__dict__)

Employee.raise_amount=1.1
print(emp_1.raise_amount)
print(emp_2.raise_amount)
print(Employee.raise_amount)

emp_1.raise_amount=1.1
print(emp_1.raise_amount)
print(emp_2.raise_amount)
print(Employee.raise_amount)

print(emp_1.__dict__)
print(Employee.__dict__)

#One example where class level value is sufficient than instance level value(self)
class Employee:
    raise_amount=1.05
    no_of_emp=0
    def __init__(self,fname,lname,pay):
        self.fname=fname
        self.lname=lname
        self.pay=pay
        self.email=fname+'.'+lname+'@gmail.com'
        Employee.no_of_emp+=1
    def fullname(self):
        return '{} {}'.format(self.fname,self.lname)
    def apply_raise(self):
        self.pay=int(self.pay*self.raise_amount) #will work - assign individual values of raise_mount
emp_1=Employee('a','g',100000)
emp_2=Employee('g','a',200000)

print(Employee.no_of_emp)


#Numpy
world_alcohol=numpy.genfromtxt('world_alcohol.csv',delimiter=',',dtype='U75',skip_header=1)

is_1986_canada=(world_alcohol[:,0]=='1986') & (world_alcohol[:,2]=='Canada')
canada_1986 =world_alcohol[is_1986_canada,:]
is_empty=canada_1986[:,4]==''
canada_1986[is_empty,4]='0'
canada_alcohol=canada_1986[:,4].astype(float)
total_canadian_drinking=canada_alcohol.sum()

column_names = food_info.columns
# Returns the tuple (8618,36) and assigns to `dimensions`.
dimensions = food_info.shape
# The number of rows, 8618.
num_rows = dimensions[0]
# The number of columns, 36.
num_cols = dimensions[1]
# Series object representing the row at index 0.
food_info.loc[0]
print(food_info.dtypes)
# DataFrame containing the rows at index 3, 4, 5, and 6 returned.
food_info.loc[3:6]

# DataFrame containing the rows at index 2, 5, and 10 returned. Either of the following work.
# Method 1
two_five_ten = [2,5,10] 
food_info.loc[two_five_ten]

# Method 2
food_info.loc[[2,5,10]]
columns = ["Zinc_(mg)", "Copper_(mg)"]
zinc_copper = food_info[columns]

# Skipping the assignment.
zinc_copper = food_info[["Zinc_(mg)", "Copper_(mg)"]]
#creting a global variable useful for automatic data creation
sas=r"C:\Users\arunansu.gorai\AG\Projects\KC_Data\Sas"
list=[170831,170731,170630,170531,170430,170331,170228,170131,161231,161130,161031,160930,160831,160731,160630
      ,160531,160430,160331,160228,160131]
for i in list:
    name='w_'+str(i)
    file_name=sas+'\w_'+str(i)+'.sas7bdat'
    globals()['w_'+str(i)]=pd.read_sas(file_name,format='SAS7BDAT') 
    
#Extracting date
#last date of a month 1st weekday of  month etc
import calendar
for i in range(2016,2018):
    for j in range(1,13):
        if j<10:
            x=str(i)+'0'+str(j)+str(calendar.monthrange(i,j)[1])
            x=x[2:]
            print(x)
        else:
            x=str(i)+str(j)+str(calendar.monthrange(i,j)[1])
            x=x[2:]
            print(x)
#Plotting and visualizing            
import matplotlib.pyplot as plt
first_twelve=unrate[0:12]
plt.plot(first_twelve['DATE'],first_twelve['VALUE'])
plt.show

plt.plot(first_twelve['DATE'], first_twelve['VALUE'])
plt.xticks(rotation=90)
plt.show()

#Plotting
plt.plot(first_twelve['DATE'], first_twelve['VALUE'])
plt.xticks(rotation=90)
plt.xlabel('Month')
plt.ylabel('Unemployment Rate')
plt.title('Monthly Unemployment Trends, 1948')
plt.show()

plt.plot(women_degrees['Year'], women_degrees['Biology'], c='blue', label='Women')
plt.plot(women_degrees['Year'], 100-women_degrees['Biology'], c='green', label='Men')
plt.legend(loc='upper right')
plt.title('Percentage of Biology Degrees Awarded By Gender')
plt.show()

fig, ax = plt.subplots()
ax.plot(women_degrees['Year'], women_degrees['Biology'], label='Women')
ax.plot(women_degrees['Year'], 100-women_degrees['Biology'], label='Men')

ax.tick_params(bottom="off", top="off", left="off", right="off")
ax.set_title('Percentage of Biology Degrees Awarded By Gender')
ax.legend(loc="upper right")

plt.show()

import pandas as pd
train=pd.read_csv('http://bit.ly/kaggletrain')
train.head()
train['sex_map']=train.Sex.map({'female':0,'male':1}) # Series Method
train.loc[0:4,['Sex','sex_map']]
#map(aFunction, aSequence)
#Way 1
items = [1, 2, 3, 4, 5]
squared = []
for x in items:
    squared.append(x ** 2)
print(squared)
#Way 2
squared=[i**2 for i in items]
print(squared)
#Way 3
def sqr(x): 
    return x ** 2
squared=list(map(sqr, items))
print(squared)
#way 3
squared=list(map(lambda x: x**2, items))
print(squared)
list(map(pow, [2, 3, 4], [10, 11, 12]))
x = [1,2,3]
y = [4,5,6]
from operator import add
print(map(add, x, y))
print(list(map(add, x, y)))
#Apply is both a series method and a df method
# Series Method
train['name_len']=train.Name.apply(len)
train.loc[0:4,['Name','name_len']]
train['fare_ceil']=train.Fare.apply(np.ceil)
train.loc[0:4,['Fare','fare_ceil']]
name_split_list=train.Name.str.split(',')
name_split_list.head() # it will generate a list separated by a comma
train.Name.str.split(',').apply(lambda x:x[0]).head()# It will generate the first value before comma

#Apply as a dataframe method
drink=pd.read_csv('http://bit.ly/drinksbycountry')
drink.head()
drink.loc[:,'beer_servings':'wine_servings'].apply(max,axis=0)# maximum value in a column
drink.loc[:10,'beer_servings':'wine_servings'].apply(max,axis=1) # maximum value in a row
drink.loc[:10,'beer_servings':'wine_servings'].apply(np.argmax,axis=1) # gives the column name with the max value.
# but in case of ties gives the first column with the highest which is by default

#applymap - a df method - it applies to every element of a df
drink.loc[:10,'beer_servings':'wine_servings'].applymap(float)
drink.loc[:10,'beer_servings':'wine_servings'].apply(sum,axis=1)
drink.loc[:10,'beer_servings':'wine_servings'].apply(sum,axis=0)

# How many uniqu Parch survived or drowned for each group in survive
df.groupby('Survived').agg({'Parch': ['count','nunique']})
df.groupby('Survived').agg({'Age': ['count','nunique']})

df.groupby('Survived').Parch.nunique()
df.groupby('Survived').Age.nunique()

#Accesing the index values
list(desc.index.values)
desc.index.values.tolist()

#creating a new column with the index values
desc['var']=desc.index
desc = desc.reset_index(drop=True)

'''
hello
excuse me
'''

quote="\"you are unique"
quote="'you are unique"
quote='\"you are unique'
print(quote)

string=''' hello,
i'm great
'''
print(string)

new=quote+string
print(new)

print("%s %s %s" % ('i like the quote', quote, string))

i=0
while (i<=20):
    if (i%2==0):
    elif(i==17):
        break
   else:
        i+=1
        continue
    i+=1
print("%c is my %s letter and number %d is %.5f" % ("A","favourute",1,.14))
        
#dopping columns 1-7 simultaneously
df = df.drop(np.arange(1,7), axis=1)

#make all the colnames of a dataframe lower or upper

df=df.rename(columns=str.lower)

#how to list the data types/ column types and group by the columns using different column types

g = df.columns.to_series().
(df.dtypes).groups
g
{dtype('int64'): ['A', 'E'], dtype('float64'): ['B'], dtype('O'): ['C', 'D']}
{k.name: v for k, v in g.items()}
{'object': ['C', 'D'], 'int64': ['A', 'E'], 'float64': ['B']}

#bivariate frequency table and one way frequency tabel
region_dist = pd.crosstab(index=req2["Region_new"], columns=req2["Region"])
region_dist = pd.crosstab(index=req2["Region_new"], columns=req2["Region"],margins=True)
region_dist/region_dist.ix["All","All"]
region_dist/region_dist.ix["All"]
region_dist.div(region_dist["All"],axis=0)
region_dist = pd.crosstab(index=req2["Region_new"], columns=[req2["Region"],req2["Tier of Service"]],margins=True) 

#lowercase or uppercase all the column names
df.columns = map(str.lower, df.columns)

#Plotting with Plotly
import plotly.plotly as py
import plotly.graph_objs as go

from datetime import datetime
import pandas_datareader.data as web

df = web.DataReader("aapl", 'yahoo',
                    datetime(2015, 1, 1),
                    datetime(2016, 7, 1))

data = [go.Scatter(x=df.index, y=df.High)]

py.iplot(data)

#creating a dataframe directly from the data

raw_data = pd.DataFrame({'first_name': ['Jason', 'Aravind', 'Netaji', 'Swamiji', 'Ramakrishna', 'Jake']})

#Creating condisitional new column need to check
train['Embarked_New'].replace([1,2,3],['C','S','Q'],inplace=True) 


#calculating rank and then creating decile wise max, mean count etc - binning - bin creation
prob_10_11=pd.read_csv(r"C:\Users\arunansu.gorai\AG\Projects\TSP_Modeling_XGBoost\prob_10_11.csv")
prob_10_11['decile'] = pd.qcut(prob_10_11['Hire'], 10,labels=False,duplicates='drop')
x= prob_10_11.groupby('decile').agg({"Hire": ['count',min, max, 'mean']})

#use of datetime function, reading a string datetime from a particular format
Hire['yearmonth']=pd.to_datetime(Hire['yearmonth'],format='%Y%m',errors='ignore')

#reset the index as a column - create a column from index
Hire['yearmonth']=Hire.index
Hire=Hire.reset_index(drop=True)

#Fitting a basic time series ARIMA model in python
#Starting with time series model

plt.plot(Hire['HireYearMonth'])
plt.plot(Hire)

from statsmodels.tsa.stattools import adfuller

#Please use the pandas.tseries module instead. from pandas.core import datetools

#Determing rolling statistics
rolmean= Hire.rolling(window=12,center=False).mean()
rolstd = Hire.rolling(window=12,center=False).std()

#Plot rolling statistics:
orig = plt.plot(Hire, color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)
    
#Perform Dickey-Fuller test:
print('Results of Dickey-Fuller Test:')
dftest = adfuller(Hire['HireYearMonth'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
#Estimating & Eliminating Trend
#    Taking log

ts_log = np.log(Hire)
plt.plot(ts_log)
#Moving average
moving_avg = ts_log.rolling(window=12,center=False).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
#differencing the moving avg
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)

ts_log_moving_avg_diff.dropna(inplace=True)

def test_stationarity(tsdata):
    rolmean= tsdata.rolling(window=12,center=False).mean()
    rolstd = tsdata.rolling(window=12,center=False).std()
    
    orig = plt.plot(tsdata, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(Hire['HireYearMonth'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)
test_stationarity(ts_log_moving_avg_diff)

#exponentially weighted moving average
expwighted_avg =ts_log.ewm(halflife=12,min_periods=0,adjust=True,ignore_na=False).mean()
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')

#Above the parameter half life denotes the parameter for exponential decay
ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)

#Eliminating Trend and Seasonality
#Differencing
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

#Decomposing
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

#the trend, seasonality are separated out from data and we can model the residuals
# check stationarity of residuals
ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)



#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

from statsmodels.tsa.arima_model import ARIMA
#AR Model

model = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((pd.DataFrame(results_AR.fittedvalues)[0]-ts_log_diff['HireYearMonth'])**2))

#combined model
model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((pd.DataFrame(results_ARIMA.fittedvalues)[0]-ts_log_diff['HireYearMonth'])**2))

#Taking it back to original scale
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

#The way to convert the differencing to log scale is to add these differences consecutively to the base number. 
#An easy way to do it is to first determine the cumulative sum at index and then add it to the base number
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

#create a series with all values as base number
predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()
#predictions_ARIMA_log[0]=ts_log.iloc[0]
#predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(Hire)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA[0]-Hire['HireYearMonth'])**2)/len(Hire)))

#converting an integer and float column to a string column
US_eco['Yearqtr']=US_eco['Year'].apply(str)+"Q"+US_eco['Qtr'].apply(str)


#Runnng a linear regression model using statsmodels
from sklearn import datasets
data = datasets.load_boston()
print (data.DESCR)

import numpy as np
import pandas as pd

# define the data/predictors as the pre-set feature names  
df = pd.DataFrame(data.data, columns=data.feature_names)
# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])

## Without a constant
import statsmodels.api as sm
X = df["RM"]
y = target["MEDV"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model
# Print out the statistics
model.summary()

## With a constant
X = df["RM"] ## X usually means our input variables (or independent variables)
y = target["MEDV"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
# Print out the statistics
model.summary()

#More than one variable
X = df[['RM', 'LSTAT']]
X = sm.add_constant(X)
y = target['MEDV']
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
model.summary()

#Runnng a linear regression model using sklearn
from sklearn import linear_model
from sklearn import datasets
data = datasets.load_boston()
# define the data/predictors as the pre-set feature names  
df = pd.DataFrame(data.data, columns=data.feature_names)
# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])

X = df
y = target['MEDV']
lm = linear_model.LinearRegression()
model = lm.fit(X,y)
predictions = lm.predict(X)
print(predictions[0:5])
lm.score(X,y) #R - Score
lm.coef_
lm.intercept_


#creating a dataframe column with consecutive integer values
x=pd.DataFrame({'int':range(1, US_eco.shape[0] + 1 ,1)})

df_t=df_t[df_t['Management Level'].isin(["P1","P2","P3","P4","P5","M1","M2","M3","M4","M5"])]


#Use of enumerate
some_list=['x','y','z']
for counter, value in enumerate(some_list):
    print(counter, value)

for c, value in enumerate(some_list, 1):
    print(c, value)
    
my_list = ['apple', 'banana', 'grapes', 'pear']
counter_list = list(enumerate(my_list, 1))
print(counter_list)
# Output: [(1, 'apple'), (2, 'banana'), (3, 'grapes'), (4, 'pear')]

# usage of *args and **kwargs
def test_var_args(f_arg, *argv):
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)

test_var_args('x', 'python', 'eggs', 'test')

def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))

greet_me(name="xyz",place="mum")


def test_args_kwargs(arg1, arg2, arg3):
    print("arg1:", arg1)
    print("arg2:", arg2)
    print("arg3:", arg3)
    
    
# first with *args
args = ("two", 3, 5)
test_args_kwargs(*args)

# now with **kwargs:
kwargs = {"arg3": 3, "arg2": "two", "arg1": 5}
test_args_kwargs(**kwargs)

#some_func(fargs, *args, **kwargs)

#Python debugger debug
#You can set break points in the script itself so that you can inspect the variables and stuff at particular points

import pdb

def make_bread():
    pdb.set_trace()
    return "I don't have time"

print(make_bread())


#object oriented programmimg class, iterables , generators etc - use of global variable
balance = 0

def deposit(amount):
    global balance
    balance += amount
    return balance
def withdraw(amount):
    global balance
    balance -= amount
    return balance
deposit(100)
withdraw(25)

#A very good example of defining a funtion
def make_account():
    return {'balance': 0}
def deposit(account, amount):
    account['balance'] += amount
    return account['balance']
def withdraw(account, amount):
    account['balance'] -= amount
    return account['balance']

a=make_account()
b=make_account()
deposit(a,200)
withdraw(a,25)
deposit(b,125)
withdraw(b,50)


class BankAccount:
    def __init__(self):
        self.balance = 0

    def withdraw(self, amount):
        self.balance -= amount
        return self.balance

    def deposit(self, amount):
        self.balance += amount
        return self.balance

a = BankAccount()
b = BankAccount()
a.deposit(200)
b.deposit(25)
b.withdraw(125)
a.withdraw(50)

#Inheritance

class MinimumBalanceAccount(BankAccount):
    def __init__(self, minimum_balance):
        BankAccount.__init__(self)
        self.minimum_balance = minimum_balance

    def withdraw(self, amount):
        if self.balance - amount < self.minimum_balance:
            print 'Sorry, minimum balance must be maintained.'
        else:
            BankAccount.withdraw(self, amount)


#Extract a number from dataframe to anew column using regular expressions
df['ticket']=df['Comments'].str.extract('(\d+)')

#How to replace a nan value using fillna and then create new variables using index find contains etc

IS_Demand["CHANNEL"].replace(np.NaN, 'NULL', inplace=True)
IS_Demand["CHANNEL_CATEGORY"] = IS_Demand["CHANNEL"].map(lambda x: 
    "Internal" if "Internal" in x else 
    "Referral" if "Referral" in x else 
    "Agency" if "Agency" in x else 
    "Social Media" if "Social Media" in x else 
    "Job Boards" if "Job Boards" in x else 
    "University" if "University" in x else "Others") 

# coalesce in python
raw_data = {'first_name': ['Jason', 'Jason', 'Jason', 'Tina', 'Jake', 'Jake'],
        'last_name': ['Miller', 'Miller', 'Statham', 'Ali', 'Milner', 'Cooze'],
        'age': [42, 42, np.nan, 24, 73, 78],
        'preTestScore': [4, 4, 31, 2, 3, 4],
        'postTestScore': [25, 25, 25, 62, 70, 23]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'preTestScore', 'postTestScore'])

df['x']=df['age'].fillna(df['preTestScore']).fillna(df['postTestScore'])

#rename all the columns in a particular format - renaming all columns
w_16=w_16.rename(columns=lambda x: x+"_old", inplace=False)


#removing by id from another dataset in df
bad = df.loc[df.Comment.str.contains('Bad Process'), 'ID Name']
df_good = df.loc[~df['ID Name'].isin(bad)]

Tenured = w_17_f.loc[~w_17_f['EEID'].isin(temp['EEID'])]

# Creating a new variable based on more than one 
al variable 
def x(row):
    if joined['Manageent_Level_P_M_old']=="P" & joined['Manageent_Level_P_M']=="M":
        return 1
    elif joined['Manageent_Level_P_M_old']=="M" & joined['Manageent_Level_P_M']=="P":
        return -1
    else:
        return 0
joined['Progression']=joined.apply(lambda row: x(row),axis=1)
joined['Progression']=joined.apply(x,axis=1)


# concatenation/ append  two datasets
concat = pd.concat([df,df0])
data = data_java.append([data_angular_js_h,data_axtech_h,data_crmtech_h,data_j2ee_arch_h])

#creation of a new variable base on two variables conditional
scored_java['Model_final_score'] = np.where((scored_java['predicted_hire_prob']>.5) & (scored_java['predicted_non_hire_prob']>.5),1-scored_java['predicted_non_hire_prob'],scored_java['predicted_hire_prob'])

def func(row):
    if (row['predicted_hire_prob']>.5) and (row['predicted_non_hire_prob']>.5):
        return 1-row['predicted_non_hire_prob']
    else:
        return row['predicted_hire_prob']
scored_java['Model_final_score'] = scored_java.apply(func,axis=1)  

def func(row):
    if (row['Model_final_decision']==1):
        return "High"
    elif (row['Model_final_decision']==0) and (row['predicted_hire']==1):
        return 'Med'
    else:
        return 'Low'
scored_java['Segment'] = scored_java.apply(func,axis=1)  



#Compress / remove a particular letter
temp_dataframe['PPI'].replace('PPI/','',regex=True,inplace=True)

temp_dataframe['PPI'].str.replace('PPI/','')

#using sql functions in pyhton
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
summary=pysqldf("SELECT count(*) as cnt,sum(Email_id_match_tag) as Email_id_match_tag,sum(mobile_no_match_tag) as mobile_no_match_tag,sum(abacus_duplicate_tag) as abacus_duplicate_tag FROM cv_map_data;")

#how to keep only a particular kind of variables dtypes in a dataset
temp=File1_MSBA.select_dtypes(include=['object'])

####################################PATH TO SAVE FILES###########################################
os.chdir(r"C:\Users\arunansu.gorai\AG\Projects\TSP\Anup")

#making all the object / character columns to upcase uppercase lowcase lowercase in python
def upcase(df):
    x=pd.DataFrame(df.dtypes)
    x1=x[x[0]=='object']
    x2=x[x[0]!='object']
    y1=[z for z in x1.index]
    y2=[z for z in x2.index]
    df1=pd.concat([df[col].astype(str).str.upper() for col in y1], axis=1)
    df2=df[y2]
    df=pd.concat([df1,df2],axis=1)
    return df

#rename all the columns in a python dataframe
df_count=df_count.add_prefix('Input_data_')
# similarly add suffix

