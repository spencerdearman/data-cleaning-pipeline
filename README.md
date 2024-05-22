# Ethics Final Project: CMSC 25910

## To Run

### Flask Server Setup
1. cd into 'flask-server'
2. run 'source myenv/bin/activate'
3. run 'python server.py'

### React App Setup
1. cd into 'react/client'
2. run 'npm start'

## Using the Data Cleaner
1. Upload your file using the button at the top. See 'ReferenceImage1'

2. On the left is the sidebar, where you can control which cleaning functions you want to run. In order to add it to the cleaning pipeline, just drag and drop it into the gray box that says 'Drop cleaning options here'. If you want to clean everything, then just press the purple 'Total Clean' button, which will add all of the functions to the pipeline.

3. Once you are satisfied with the options, then click 'Upload and Clean' which might take a few seconds. Then, you should see the pipeline turn green and the progress bar should move to 100%. You can then scroll down on the sidebar and select the option 'Download Cleaned File'. See 'ReferenceImage2'

### Note 
If you want to see the outcome with test data, I would recommend looking at the audible folder under the test-data section. 'audible_uncleaned' is the uncleaned version, and 'cleaned_audible_uncleaned' is the outcome after being run through the complete pipeline. I DID NOT EDIT THE OUTCOME FILE.

### Main Files

#### Server.py 
This is where all of the actual functions for cleaning the data are. This is the python file that is in the flask-server folder, and it also handles a lot of the pipeline processing. 

#### App.js
Main file for the React App

#### PipelineVisualization.js
The pipeline structure and formatting

#### Sidebar.js
The sidebar structure and formatting

#### utils.js
Just contains the cleaningOptionsMap



