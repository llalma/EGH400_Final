from os import listdir
from os.path import isfile, join

def creat_dict():
    #can be used to generate a dict from a set of files.
    #just past output into get_dict manually to chaneg teh dictionary for other functions

    mypath = "C:\\Users\\llalma\\Google Drive\\UNI\\QUT\\4th year\\2nd\\egh400\\comparison\\data\\logs"

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    output = {}
    value = 0
    for f in onlyfiles:
        if output.get(f[-1-10:-1-7]) == None:
            output[f[-1-10:-1-7]] = value
            value += 1
        #end
    #end

    print(output)
#end

def get_dict():
    return {'DEN': 0, 'ARN': 1, 'DOH': 2, 'HKG': 3, 'ICN': 4, 'IEV': 5, 'ILR': 6, 'KUL': 7, 'LCY': 8, 'NYC': 9, 'SAO': 10, 'SDJ': 11, 'SFO': 12, 'SGP': 13, 'TPE': 14, 'TYO': 15, 'ZRH': 16}
#end