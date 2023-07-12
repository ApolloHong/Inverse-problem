The first generation of this project.
author = 'HELIN GONG', 'Lizhan Hong', 'Junyu Pan', 'Chenxi Yang', 'Haicheng Huang', 'Chenjie Song'

# NRDTML 1.0
## This is the instruction of for POD process in reactorDt第一代设计

+ Overview:
  + The provided code is a Python module with several functions for reading and processing power grid data. 
  It performs operations such as selecting specific columns from the data, reducing the dimensionality of the data, 
  and selecting specific rows and columns from the data. 
  The main purpose of the module appears to be to extract relevant information from the power grid data.

+ Code Explanation:

   + The first two lines of the code import the necessary libraries numpy, pandas, scipy.sparse, and sklearn.utils.extmath. 
   + The read() function is used to read a data file either in text format or Excel format. 
     The function takes two arguments: the absolute path of the data and the type of file (txt or excel). The function returns a numpy ndarray of the data.
   + The extractInpower() function takes the matrix returned by read() and extracts the 2nd to 5th columns. The function then returns the transposed Inpower matrix.
   + The contractIntoOneInFour() function takes the Power matrix and reduces its dimensionality by exploiting symmetry in the data. It only keeps the relevant columns in a given section of the power grid. 
     The function takes four arguments: the Power matrix, the number of grids per section, the number of vertical sections, and the range of useful columns. 
     The function returns a simplified Power matrix.
   + The POD() function performs RandomizedSVD (RSVD) on the simplified Power matrix to reduce its dimensionality.RSVD is a numerical algorithm used to approximate the singular value decomposition of a large matrix, by randomly sampling a subset of its columns, 
     which allows for more efficient and faster computation of the decomposition, especially for matrices that are too large to fit in memory.
     The function takes two arguments: the simplified Power matrix and the number of singular values and vectors to extract. The function returns a reduced order basis with r columns.
   + The selectionMatrix() function selects specific rows and columns from the Power matrix. 
     It takes five arguments: the grids and sections to be selected, the Power matrix, the number of grids in a quarter section, and the number of vertical sections. The function returns the selected matrix.

## Plots
+ Overview:
  + The code provided can be used to plot 3D heatmaps and one-column graphs of a nuclear reactor state. 
    The functions can be customized to suit the needs of the user by adjusting the input parameters.

+ Code Explanation:
  + The code requires the use of various libraries including matplotlib, numpy, pandas, scipy.sparse, and seaborn.
  + The reactorGetSectionIndex() function generates the index of the one section data in a grid square. 
    It takes in the path of the control data file as an argument and returns a list of indices.
  + The reactorPlot3D() function plots a 3D heatmap of the entire reactor. 
    It takes in several arguments including the path of the control data file, the path of the reactor data file, 
    the number of the sample, the length and width of one section, the number of grids per section, and the number of vertical sections.
    It also uses the index of the table NRS.
    The function selects a subset of the data from the reactor file based on the input parameters and generates a 3D heatmap using matplotlib's voxels method. The function also adds a color bar and labels for the axes.
  + The reactorPlotCol() function plots a one-column graph of a reactor state. 
    It takes in several arguments including the path of the control data file, 
    the path of the reactor data file, the number of the sample, the length and width of one section, the number of the grid to be plotted, the number of grids per section, and the number of vertical sections. 
    The function selects a subset of the data from the reactor file based on the input parameters and generates a one-column graph using matplotlib's voxels method. The function also adds a color bar and labels for the axes.


# Apolloid 2.0
## KnnFor.py
## KnnInv.py
## Pro.py : only being used once
## ProSub.py : can be used for a seconde time
## Read.py
## Optimize.py : 
+ In our whole process, the data was normalized.



# Attention :
+ All the input parameters are required for python rule, which are started from 0.
+ The power data is range with numSection in a period and numGrid of periods.

# Appendix:


## Apolloid 1.0

### dimention of matrix in files
+ 'inpower18480.out': (18480, 9),
+ 'inpower18480_4.txt': (18480, 4)
+ 'power8480.txt': (8480, 4956)
+ 'power10000.txt': (1000, 4956)
+ 'InpowerData.txt': (23997, 4)
+ 'Power10.txt': (10, 4956)
+ 'Inpower10.txt': (10,4)

### Return 1
+ 'Power.txt': (23997, 1456)
+ 'alpha.txt': (23997,50)
+ 'q.txt': (50,1456)
+ 'Y.txt': (23997, 84)
+ 'Inpower.txt': (23997, 4)

### Useless
+ 'PowerData.txt': (23997, 4956) # 还未实现
+ 'power18480.out': (1,1) #useless

## NRDTML 2.0

### Input
+ 'inpower18480.out': (18480, 9),
+ 'inpower18480_4.txt': (18480, 4)
+ 'inpower5517.out': (5517, 9) 
+ 'power5517.out': (5517, 4956) 
+ 'powerIAEA5517.txt': (5517, 1456)
+ 'powerIAEA8480.txt': (8480, 1456)
+ 'powerIAEA10000.txt': (10000, 1456)
+ 'powerIAEA18480coef.txt': (18480, 50)
+ 'powerIAEA18480basis.txt': (50,1456)
+ 'alpha5517.txt': (5517,50)
+ 'q5517.txt': (50,4956)
+ 'Y5517.txt': (5517, 84)
+ 'Inpower5517.txt': (5517, 4)
+ 'alpha10.txt': (10,50)
+ 'Y10.txt': (10,84)
+ 'Inpower10.txt': (10,4)
+ 'sensors.txt': (84, 1456)
+ 'knntest_input.pkl': (4620, 4)
+ 'knntest_output.pkl': (4620, 50)
+ 'knntrain_input': (13860, 4)
+ 'knntrain_output': (13860, 50)
+ 'inpowerNor18480_4.txt': (18480, 4) # the normalized data
+ 'scalingNor.txt': (4, 4) # the scaling factors for the input data.

### data in inverse_problem_predict
+ parameters.shape : (18480, 4)
+ field.shape : (18480, 1456) 
+ observations.shape : (18480, 84)  
+ sensors.shape : (84, 1456) 
+ r : the dimension of modes (50 in our case)   nc is the iterable dimension for finding the optimal r.


# Results
+ 'ForwardKnn Handwritten vs Sklearn' : 
+ 'InverseKnn Handwritten vs Sklearn' : 

## The distinction between some variables:
+ The variables begin with 'n' denote the number we choose into our 
visualization, while those begin with 'num' denote the upper bound of our range number.
For instance:
{
    nGrid: 'The number of gird we want (ranging from 0 to 176).'
    numGrid: 'The number of grids per section  (In our case, numGrid = 177).'
    }


## The table of one nuclear reactor section  (NRS).

|     |     |     |     |     |     |     |     |     |     |     |     |     |     |      |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
|     |     |     |     |     | 1   | 2   | 3   | 4   | 5   |     |     |     |     |      |
|     |     |     | 6   | 7   | 8   | 9   | 10  | 11  | 12  | 13  | 14  |     |     |      |
|     |     | 15  | 16  | 17  | 18  | 19  | 20  | 21  | 22  | 23  | 24  | 25  |     |      |
|     | 26  | 27  | 28  | 29  | 30  | 31  | 32  | 33  | 34  | 35  | 36  | 37  | 38  |      |
|     | 39  | 40  | 41  | 42  | 43  | 44  | 45  | 46  | 47  | 48  | 49  | 50  | 51  |      |
| 52  | 53  | 54  | 55  | 56  | 57  | 58  | 59  | 60  | 61  | 62  | 63  | 64  | 65  | 66   |
| 67  | 68  | 69  | 70  | 71  | 72  | 73  | 74  | 75  | 76  | 77  | 78  | 79  | 80  | 81   |
| 82  | 83  | 84  | 85  | 86  | 87  | 88  | 89  | 90  | 91  | 92  | 93  | 94  | 95  | 96   |
| 97  | 98  | 99  | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111  |
| 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 | 120 | 121 | 122 | 123 | 124 | 125 | 126  |
|     | 127 | 128 | 129 | 130 | 131 | 132 | 133 | 134 | 135 | 136 | 137 | 138 | 139 |      |
|     | 140 | 141 | 142 | 143 | 144 | 145 | 146 | 147 | 148 | 149 | 150 | 151 | 152 |      |
|     |     | 153 | 154 | 155 | 156 | 157 | 158 | 159 | 160 | 161 | 162 | 163 |     |      |
|     |     |     | 164 | 165 | 166 | 167 | 168 | 169 | 170 | 171 | 172 |     |     |      |
|     |     |     |     |     | 173 | 174 | 175 | 176 | 177 |     |     |     |     |      |






