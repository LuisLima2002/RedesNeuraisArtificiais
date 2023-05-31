def function(x):
    return x*x+x+1
def function1(x):
    return abs(x)

def function2(x,y):
    return max(0,y*x)

file = open("data.csv","w")
file.write("X,Y\n")
for i in range(1000):
    file.write(str(i-40)+","+str(function((i-500)))+"\n")
file.close()

