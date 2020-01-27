import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = [[2618,   61,    0,   0,    0],
 [ 231,  945,   48,    0,    0],
 [  32,  175,  517 ,  15 ,   0],
 [   0  , 14 ,  73 , 108 ,   1],
 [   0 ,   1,    1 ,  12  ,  4]]

df_cm = pd.DataFrame(array, range(5),
                  range(5))
#plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size

plt.show()
