import numpy as np
import pandas as pd 

pd.set_option('display.width', 1000)  # 设置字符显示宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行

b = np.loadtxt('embedding.txt')

b=abs(b)

matrix_save=b.mean(axis=1)

np.savetxt("embedding_mean.txt",matrix_save,fmt='%.12e')


test=pd.read_csv("vocab.txt",names=["vocab"])
score = np.loadtxt('embedding_mean.txt')

test['score'] = score[:]
test[['vocab','score']].to_csv('emb_voc.csv',sep='\t',index = None)


lc=pd.DataFrame(pd.read_csv('emb_voc.csv',names=["vocab","score"],sep="\t"))

lc=lc.sort_values(by=["score"])
# for i in range(len(lc)):
# 	print(lc["vocab"][i])
print(lc)
