# NEM_clustering
This repository contains python code for implementing Neighbourhood Expectation Maximisation (NEM) clustering. 

NEM is a modifed version ofthe EM algorithm with a spatial constraint. Such an algorithm can be usefull for image segmentation or for any problem related to decomposing data into gaussian components. It can also be applied to lognormally distributed data. Taking the logarithm of such data yields gaussian distributed data.

## NEM in a nutshell:
A new criterion U is defined that add a spatial term. The spatial term is dependant on a neighbour matrix V.
NEM uses a version of the EM algorithm to maximise the criterion U, given gaussian mixture parameters. It can be used for classification of spatial data.

## Relevant litterature:

First paper on NEM: 
C. Ambroise and V. M. Dang, “Clustering of spatial databythe EM algorithm,” geoENV I-Geostatistics for Environmental Applications, vol. 9, Dec. 6, 1997.

Convergence and matheamtical theorems for NEM: C. Ambroise and G. Govaert, “Convergence of an EM-type algorithm for spatial cluster- ing,” Pattern Recognition Letters, vol. 19, no. 10, pp. 919–927, Aug. 1998, issn: 01678655. doi: 10.1016/S0167-8655(98)00076-2. [Online]. Available: https://linkinghub.elsevier. com/retrieve/pii/S0167865598000762

Initialisation:
T. Hu, J. Ouyang, C. Qu, and C. Liu, “Initialization of the neighborhood EM algorithm for spatial clustering,” in Advanced Data Mining and Applications, R. Huang, Q. Yang, J. Pei, J. Gama, X. Meng, and X. Li, Eds., vol. 5678, Series Title: Lecture Notes in Computer Science, Berlin, Heidelberg: Springer Berlin Heidelberg, 2009, pp. 487–495, isbn: 978-3- 642-03347-6 978-3-642-03348-3. doi: 10.1007/978-3-642-03348-3_48. [Online]. Available: http://link.springer.com/10.1007/978-3-642-03348-3_48 
