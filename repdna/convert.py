import fastq_reader
from repDNA.nac import RevcKmer

def convert(name):
    print("Now converting "+name)

    #COnvert from fastq to fasta
    fastq2fasta()

    #4mers representation
    revckmer = RevcKmer(k=6)

    #Open file to write to
    f = open("vectored_data/" + name + ".txt", "w")

    temp = revckmer.make_kmer_vec(open("vectored_data/temp.fasta"))
    for val in temp:
        string = ""
        for i,num in enumerate(val):
            string+=str(num) + ","
        #end
        f.write(string[0:-1-1]+"\n")
    #end

    #Close file correctly
    f.close()
#end

def fastq2fasta():
    #Read file
    sequences = fastq_reader.read("vectored_data/temp.fastq")
    
    #Open file to write to
    f = open("vectored_data/temp.fasta", "w")

    for seq in sequences:
        f.write(">"+seq.get("name")+"\n")
        seq = seq.get("sequence")+"\n"
        seq = seq.replace("N","")
        f.write(seq)
    #end

    f.close()
#ends