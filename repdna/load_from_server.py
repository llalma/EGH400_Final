import os
import paramiko 
from os import listdir
import convert

myHostname =
myUsername =
myPassword =

def get_remote_path(sftp):
    files = sftp.listdir()
    output = []

    for f in files:
        if "CSD17" in f and "log" not in f:
            output.append(f)
        #emd
    #end

    return output
#end

def already_converted(name):
    #Checks if the file has already been converted
    files = listdir("vectored_data/")
    
    if name+".txt" in files:
        return True
    #end
    return False
#end

def download():
    #Downloads csd17 files and converts to vector representation

    #Connect to big data server and change working directory
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(myHostname, username=myUsername, password=myPassword)
    sftp = ssh.open_sftp()
    sftp.chdir('../../data5/camdaNonC/')

    #Get files
    paths = get_remote_path(sftp)

    localpath = 'vectored_data/temp.fastq'
    for path in paths:
        if not already_converted(path[-1-12:-1-5]):
            #Download file
            sftp.get(path,localpath)

            #Convert temp file to vectored representation
            convert.convert(path[-1-12:-1-5])

            #Delete temp file
            os.remove("vectored_data/temp.fastq")
            os.remove("vectored_data/temp.fasta")
        #end
    #end

    sftp.close()
    ssh.close()  
#end

if __name__ == '__main__':
    download()
#end