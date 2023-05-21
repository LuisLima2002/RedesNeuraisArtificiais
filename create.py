def function(x):
    return x*x+x+2
def function1(x):
    return abs(x)
file = open("data.csv","w")
for i in range(80):
    file.write(str(i-40)+","+str(function1(i-40))+"\n")
file.close()

