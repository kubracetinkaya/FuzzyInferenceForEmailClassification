import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import statistics
import skfuzzy.membership as mf
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# read the dataset
df=pd.read_csv("dataset/all.us.txt")

# read the dataset column
list=[]
df_open=df['Open']
df_high=df['High']
df_low=df['Low']
df_close=df['Close']
Lavg=[]

date=[]
for x in range(1,219):
    date.append(x);
df_date=pd.DataFrame(date)

# calculate avarage of columns and append to Lavg array 
for i in range(df_open.size):
    avg=(df_open[i]+df_close[i]+df_low[i]+df_high[i])/4
    Lavg.append(avg)
        
df_avg=pd.DataFrame(Lavg)

# calculate standart deviation with lavg's percentage
for x in range(df_open.size-1):
    percentage=(Lavg[x+1]*100)/Lavg[x]
    list.append(percentage-100)
s=statistics.stdev(list)
print("Standard Deviation:",s*40)


# split matrices into random train and test subsets with train_test_split 
x_train, x_test,y_train,y_test = train_test_split(df_date, df_avg, test_size=0.005, shuffle=False)

# show linear regression with predicted train and test subsets
lr = LinearRegression()
lr.fit(x_train,y_train)
prediction = lr.predict(x_test)
plt.scatter(df_date,df_avg,color='red')
plt.plot(df_date,lr.predict(df_date), color = 'blue')
plt.show()

# calculate slope with predicted data 
slope=prediction[1]-prediction[0]
print('Slope:',slope)

# formula for linear regression slope in stock market
slope=slope+1
slope100=(100*slope)/2
#print(slope100)


# computes standart deviation values using a triangular membership function
x_std=np.arange(0,101,1)
std_low=mf.trimf(x_std,[0,28,56])
std_med=mf.trimf(x_std,[28,56,84])
std_high=mf.trimf(x_std,[56,84,100000])

# computes slope values using a triangular membership function
x_slope=np.arange(0,101,1)
slope_inc=mf.trimf(x_slope,[0,100,100])
slope_dec=mf.trimf(x_slope,[0,0,100])


# computes risk values using a triangular membership function
x_risk=np.arange(0,101,1)
low_risk=mf.trimf(x_risk,[0,0,50])
med_risk=mf.trimf(x_risk,[30,50,70])
high_risk=mf.trimf(x_risk,[50,100,100])

plt.set_title='Slope'
plt.plot(x_slope,slope_inc,'r',linewidth=2,label="Increase")
plt.plot(x_slope,slope_dec,'b',linewidth=2,label="Decrease")
plt.legend()
plt.show()

plt.set_title='Std Dev'
plt.plot(x_std,std_low,'r',linewidth=2,label="Std.Dev Low")
plt.plot(x_std,std_med,'g',linewidth=2,label="Std.Dev Med")
plt.plot(x_std,std_high,'b',linewidth=2,label="Std.Dev High")
plt.legend()
plt.show()

plt.set_title='Risk'
plt.plot(x_risk,low_risk,'r',linewidth=2,label="Low Risk")
plt.plot(x_risk,med_risk,'g',linewidth=2,label="Med Risk")
plt.plot(x_risk,high_risk,'b',linewidth=2,label="High Risk")
plt.legend()
plt.show()

# formula for linear regression slope in stock market
s=s*40
input_std=s
input_slope=slope100

# fuzzy membership formulas for standard deviation
std_fit_low=fuzz.interp_membership(x_std,std_low,input_std)
std_fit_med = fuzz.interp_membership(x_std,std_med,input_std)
std_fit_high=fuzz.interp_membership(x_std,std_high,input_std)

# fuzzy membership formulas for slope
slope_fit_inc=fuzz.interp_membership(x_slope,slope_inc,input_slope)
slope_fit_dec=fuzz.interp_membership(x_slope,slope_dec,input_slope)

# find min values and set fuzzy rules with compared values
rule1=np.fmin(np.fmin(std_fit_low,slope_fit_inc),low_risk)
rule2=np.fmin(np.fmin(std_fit_low,slope_fit_dec),med_risk)
rule3=np.fmin(np.fmin(std_fit_med,slope_fit_inc),med_risk)
rule4=np.fmin(np.fmin(std_fit_med,slope_fit_dec),med_risk)
rule5=np.fmin(np.fmin(std_fit_high,slope_fit_dec),high_risk)
rule6=np.fmin(np.fmin(std_fit_high,slope_fit_inc),high_risk)

# find max values for output rules with compared minima rules 
out_low=rule1
out_med=np.fmax(rule2,rule3,rule4)
out_high=np.fmax(rule5,rule6)

# processes for visualize risk table
risk0=np.zeros_like(x_risk)
fig,bx0=plt.subplots(figsize=(7,4))
bx0.fill_between(x_risk,risk0,out_low,facecolor='r',alpha=0.7)
bx0.plot(x_risk,low_risk,'r',linestyle='--')
bx0.fill_between(x_risk,risk0,out_med,facecolor='g',alpha=0.7)
bx0.plot(x_risk,med_risk,'g',linestyle='--')
bx0.fill_between(x_risk,risk0,out_high,facecolor='b',alpha=0.7)
bx0.plot(x_risk,high_risk,'b',linestyle='--')
bx0.set_title('Risk')

# defuzzification operations for calculated output value
out_risk=np.fmax(out_low,out_med,out_high)
defuzzified=fuzz.defuzz(x_risk,out_risk,'centroid')
result = fuzz.interp_membership(x_risk,out_risk,defuzzified)
print("Calculated output value:",defuzzified)