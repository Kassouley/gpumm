from difflib import SequenceMatcher
import statistics
import sys


def cmp_file(file1, file2): 
    ratio=[]
    while True:
        text1 = file1.readline()
        text2 = file2.readline()
        if not text1 and not text2:
                break
        text1 = text1.split(sep=" ")
        text2 = text2.split(sep=" ")
        for value1, value2 in zip(text1,text2) :
            ratio.append(SequenceMatcher(None, value1, value2).ratio())
    return statistics.mean(ratio)

def main() :
    file1 = open('./output/check_basis.out')
    file2 = open(f"./output/check_{sys.argv[1]}.out")
    avg_ratio = round(cmp_file(file1,file2),3)
    print(f"\033[42mSortie base/{sys.argv[1]} semblable à {avg_ratio*100}%\033[0m")

if __name__ == '__main__':
    main()