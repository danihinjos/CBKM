# Cooperative Bisecting K-Means (CBKM) Algorithm Implementation

Cooperative Bisecting K-Means is presented in [1] as an algorithm based on the principles of Bisecting K-Means, including the idea of splitting clusters at each level of a generated hierarchical tree (from l=2...k ). However, it includes the notion of intermediate cooperation between KM and BKM. Speci cally, by using cooperative contingency and cooperative merging matrices at each level, resulting from the intersections of KM and BKM and the subclusters merging cohesiveness factors, respectively, CBKM eventually obtains better clustering results than the ones provided by both methods separately.

The aim of this project is to understand, summarize, implement, reproduce and test this algorithm with different artifcial and real datasets.

[1] R. Kashef and M. Kamel, "Enhanced bisecting k-means clustering using intermediate cooperation."

---------------------------------------------------------------------

This project has been developed using Python v3.6 as programming language and PyCharm as IDE. In order to execute the project, simply open the project with PyCharm
and run "main.py". You will be greeted with a menu that will provide you the opportunity of selecting between several options:

• "1": Processing DS1 dataset.

• "2": Processing DS2 dataset.

• "3": Processing Heart-C dataset.

• "4": Processing Breast Cancer dataset.

• "5": Processing CMC dataset.

• "6": Process another dataset.

• "7": Exit.

Before anything else, note that the PyCharm project is composed by five .py  les and two folders: "datasets" and "results". In order for the adjusted execution to work properly, the dataset files should be located inside a folder called "datasets" within the root path. The folder called "results" should also exist previously to running.

Once the project is properly set, by selecting "1", "2", "3", "4" or "5" you will start the processing of the studied datasets with CBKM, KM, BKM and SL. By selecting option number "4" you will be able to process another dataset of your choice. For that matter, you will have to provide the filename of your dataset, and verify that said file is located inside a folder "datasets" as just mentioned. In addition, only .csv or .arff files will be properly loaded. Last option is just exiting the execution.

Select a studied dataset and, if it's real, some basic relevant information about it will be displayed in a first instance. Afterwards, the CBKM will start executing for the specified number of iterations and with a fixed k value that can be changed within the main file. After the n runs of CBKM, n runs of KM and BKM will be executed as well. Finally, only one execution of SL will be performed. When the whole process is finished, the relevant information about the metrics for the different algorithms will be stored in the corresponding .txt file inside the folder "results".

Moreover, if you want only the relevant classes and methods of the algorithm in order to integrate it in your library, you must extract the files "CBKM.py", "BKM.py" and "utils.py" from the project. The first two contain the classes for CBKM and BKM, whereas the third one comprises relevant methods used by the main and by both classes. Therefore, be careful to include the important methods of "utils.py", which can be easily integrated into the classes if desired. Mostly all relevant imports are in "utils.py" as well, so make sure to correctly reference them.

In a separate way, if you want to test the algorithm's performance with the metrics introduced, you should also consider all the methods of the "metrics.py" file.
