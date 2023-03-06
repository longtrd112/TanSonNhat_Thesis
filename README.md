# TanSonNhat_Thesis

1. Raw extracted data is located in /data/Extract/date/flight

2. METAR data is located in /data/

3.  Step 1: Run /dataSorting/data_sorting.py -> generate sorted data located in /data/sortedData/date/flight

    Step 2: Run /featureExtraction/feature_extraction.py -> generate 'extracted_features.csv' in the same directory
	
    Step 3: Run /featureExtraction_3points/feature_extraction_3points.py -> generate 'extracted_features_3points.csv' in the same directory
	
    Step 4: Build machine learning model:  
	4.1 Random Forest  
		4.1.1 Run /randomForest_sklearn/utils/add_metar_features.py and /randomForest_sklearn/utils/add_metar_features_3points.py  
                      -> Generate 'final_data.csv' and 'final_data_3points.csv'  
                4.1.2 Run /randomForest_sklearn/result.py with value of variable file_name as 'final_data.csv' and 'final_data_3points.csv' respectively  
                      -> Generate 'result_output.txt', 'result_output_3.txt' (average metrics values) and 'loss_history.csv', 'loss_history_3.csv'  
	4.2 Neural Network  
		4.2.1 Copy-paste 'final_data.csv' and 'final_data_3points.csv' generated earlier, or re-run the same code in /neuralNetwork_keras/utils/  
		4.2.2 Run /neuralNetwork_keras/result.py with value of variable file_name as 'final_data.csv' and 'final_data_3points.csv' respectively  
                      -> Generate 'result_output.txt', 'result_output_3.txt' (average metrics values) and 'loss_history.csv', 'loss_history_3.csv'  
