#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


epl=pd.read_csv('/Users/sarahelfeel/Downloads/archive-2/EPL_21_22.csv')


# In[14]:


epl.head()


# In[16]:


epl.info()


# In[17]:


epl.describe()


# In[18]:


epl.isna().sum()


# In[19]:


#create a column for goals per game and assists per game

epl['GoalPerGame']=(epl['goals'])/(epl['mp']).astype(float)
epl['AssistPerGame']=(epl['assists'])/(epl['mp']).astype(float)


epl.head()


# In[20]:


#Total Goals scored in 21/22 season
total_goals = epl['goals'].sum()
print(total_goals)


# In[21]:


#Total penalty goals
total_penalties=epl['pkmade'].sum()
total_penalties_attempted=epl['pkatt'].sum()
pens_success=total_penalties/total_penalties_attempted
print(pens_success*100)


# In[22]:


#pie chart for penalty success
plt.figure(figsize=(13,6))
penalties_comp=np.array([total_penalties, total_penalties_attempted-total_penalties])
labels=np.array(["penalties scored", "penalties missed"])
#plt.pie(penalties_comp)
plt.pie(x=penalties_comp, autopct="%.1f%%", labels=labels, pctdistance=0.5)
plt.title("Penality success rate")


# In[23]:


#all defenders in the league
unique_pos = epl['pos'].unique()
epl[epl['pos']=='DF']


# In[35]:


#all nationalities
nations=np.size((epl['nation'].unique()))
print(nations)


# In[30]:


#countries with most players
most_country=epl.groupby('nation').size().sort_values(ascending=False)
most_country.head(20).plot(kind="bar", figsize=(12,6), color=sns.color_palette("pastel"), title="nations with most players")


# In[31]:


#clubs with most players
most_players=epl['club'].value_counts()
most_players.nlargest(5).plot(kind='bar', title='clubs with biggest squad/most players')


# In[32]:


#age groups
Under20=epl[epl['age']<=20]
Between20_25=epl[(epl['age']<=25) & (epl['age']>20)]
Between25_30=epl[(epl['age']<=30) & (epl['age']>25)]
Between30_35=epl[(epl['age']<=35) & (epl['age']>30)]
Above35=epl[epl['age']>35]

x=np.array([Under20['players'].count(),Between20_25['players'].count(), Between25_30['players'].count(), Between30_35['players'].count(), Above35['players'].count()])
label = (["under 20", "between 20 and 25", "between 25 and 30", "between 30 and 35", "above 35"])
color = (['red', 'blue', 'yellow', 'green', 'grey'])
pie_Chart = plt.pie(x, labels=label, colors=color)
plt.title("age groups in the EPL")
plt.show()


# In[34]:


#club with most young players
club_most_young = Under20['club'].value_counts().idxmax()
print(club_most_young)

Under20['club'].value_counts().nlargest(5).plot(kind='bar', title='clubs with most young players')



# In[41]:


#u20 players in leeds
Under20[Under20['club']=='Leeds United']



# In[40]:


#players between 30-35 in Liverpool
Between30_35[Between30_35['club']=='Liverpool']


# In[38]:


#average age of players in clubs
plt.figure(figsize=(12,6))
avgPl=sns.boxplot(x='club', y='age',data=epl)
plt.xticks(rotation=90)


# In[43]:


#average age of players in clubs
numPlayer=epl.groupby('club').size()
data=epl.groupby('club')['age'].sum()/numPlayer
data.sort_values(ascending=False)


# In[39]:


#Number of Goals scored in each club
goals_club = epl.groupby('club')['goals'].sum()
data_goals=goals_club.sort_values(ascending=False)
data_goals.plot(kind='bar', title='Number of goals in each club')


# In[76]:


#assists per club
goals_by_club=pd.DataFrame(epl.groupby('club', as_index=False)['assists'].sum().sort_values(by='assists'))

ax=sns.barplot(x='club', y='assists', data=goals_by_club, palette='rocket')
plt.xticks(rotation=75)
plt.title('Plot of clubs vs total assists', fontsize=20)


# In[77]:


#10 highest top scorers
top_10_players=epl[['players', 'club', 'goals','mp']].nlargest(n=10, columns='goals')
top_10_players


# In[67]:


#10 players with most goals per game
maximum=epl[['players','club','GoalPerGame']].nlargest(n=5, columns='GoalPerGame')
maximum.sort_values(ascending=False, by='GoalPerGame')


# In[41]:


#adding shot efficiency to the data set
epl['ShotEff']=(epl['shtont'])/(epl['shots']).astype(float)
epl.head()


# In[44]:


#Shot Efficiency of players with at least 10 shots 

shotsEff=epl[['players', 'pos', 'club','shots', 'shtont','ShotEff']].sort_values(ascending=False, by='ShotEff')
updated_shotsEff=shotsEff[shotsEff['shots']>=10].nlargest(n=15, columns='ShotEff')

updated_shotsEff.plot(x='players', y='ShotEff',kind='bar', title='players with highest shot efficiency (atleast 10+ shots)')
#plt.xticks(rotation=75)


# In[137]:


#Highest earners in the league 
earners = epl[['players', 'club', 'weekly']]
earners.nlargest(n=10, columns='weekly')


# In[167]:


#Annual wages of each club
weekly_wages = epl.groupby('club')['weekly'].sum()
annual_wages=weekly_wages.sort_values(ascending=True)*52
titles=['annual wages in billion pounds', 'clubs']
annual_wages.plot(kind='barh')
plt.title('Annual wages of EPL Clubs', fontsize=15)
plt.xlabel("wages in billion GBP")
plt.ylabel("clubs")




# In[48]:


#most shots by players 
maxshots=epl[['players','club','shots','shtont']].nlargest(n=10, columns='shots')
maxshots


# In[60]:


#Regression shots vs shots on target
X=epl['shots']
Y=epl['shtont']
X1=np.array(X)
Y1=np.array(Y)
X2 = X1[:, None] 
model = LinearRegression()
model.fit(X2, Y1)
model.score(X2, Y1)

#r2 value
y_pred = model.predict(X2)
r2 = r2_score(Y1, y_pred)
print('R2 Value: ', r2)

#plot the graph
plt.scatter(X2, Y1, label='Data', s=10)
plt.plot(X2, y_pred, color='red', label='Linear Regression')
plt.xlabel("shots")
plt.ylabel("shots on target")
plt.title("Shots vs Shots on target")
plt.show()


#equation of best fit line
slope = model.coef_[0]
intercept = model.intercept_
equation = f'y = {slope:.3f}x + {intercept:.3f}'
print('Equation of Best-Fit Line:', equation)


# In[46]:


#Regression penalties taken vs penalties scored 
X=epl['pkatt']
Y=epl['pkmade']
X1=np.array(X)
Y1=np.array(Y)
X2 = X1[:, None] 
model = LinearRegression()
model.fit(X2, Y1)
model.score(X2, Y1)

#r2 value
y_pred = model.predict(X2)
r2 = r2_score(Y1, y_pred)
print('R2 value: ',r2)

#plot the graph
plt.scatter(X2, Y1, label='Data', s=10)
plt.plot(X2, y_pred, color='red', label='Linear Regression')
plt.xlabel("penalties taken")
plt.ylabel("penalties scored")
plt.title("penalties taken vs scored")
plt.show()


#equation of best fit line
slope = model.coef_[0]
intercept = model.intercept_
equation = f'y = {slope:.3f}x + {intercept:.3f}'
print('Equation of Best-Fit Line:', equation)


#Percentage of penalty success
print('The percentage of penalty success is: ', slope)


# In[ ]:





# In[ ]:




