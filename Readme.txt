This file is for the information about code and files for the work done in paper titled "Accelerated discovery of perovskite-based materials guided by machine learning techniques."

Information about the files: 

The supplementary materials folder contains the following files:

	1. Seven excel files for total 246820 prototype structures with material names and features for each material. (1-ABX3, 2-AB2X5, 3-A2BX6, 4-A3B2X9, 5-AB2X4, 6-A2BCX6, 7-A4BCX12)

	2. Bandgap_Training_Dataset.csv : Contains bandgap and Features for 14633 Compounds from the ICSD database. 

	3. Stability_Training_Dataset.csv: Contains stable and unstable compounds with features (stable compounds: 1 and unstable compounds: 0)

	4. Validation_Dataset1_with_Features.csv: Contains dataset for 54461 compounds with features from the Materials Project Database.

	5. Validation_Dataset2_with_Features.csv: Contains 253 perovskite materials with features from the Materials Project Database.

	6. Bandgap Model Training Code.py: python file for training the Random Forest Regressor for the ICSD dataset for Bandgap.

	7. Stability Model Training Code.py: python file for training the Random Forest Classifier for the ICSD dataset for Stability.

	8. Validation Code for Bandgap.py: python file used for the validations datasets 1 and 2.

	9. Tolerence Factor for ABX3.py: python file for calculating tolerence factor for single perovskite structures.

	10. Tolerence Factor for ABCX.py: python file for calculating tolerence factor for double perovskite structures.

	11. FinalRegModel.sav: pickle file for regression model to predict bandgap.

	12. FinalStabilityModel.sav: pickle file for classification model to predict stable compounds.

	14. Predict Bandgap.py: python file for predicting bandgap for the prototype structures.

	15. Predict Stability.py: python file for predicting the stable compounds.

All other files in the folder have a self explainatory filename.  
	

Usage: 

	1. Each file can be run using any python IDE. 
	2. all codes are self explainatory and need no change to predict and/or train. 
	3. In the python files for tolerence factor and bandgap prediction, filename needs to be changed to predict any of the seven prototype structures files. 
     

          
 