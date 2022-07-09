Dataset = imageDatastore('dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[Training_Dataset, Validation_Dataset] = splitEachLabel(Dataset,7.0);

net = googlenet;
analyzeNetwork(net)

Input_Layer_Size = net.Layers(1).InputSize(1:2);
Resized_Training_Images = augmentedImageDatastore(Input_Layer_Size,Training_Dataset);
Resized_Validation_Images = augmentedImageDatastore(Input_Layer_Size,Validation_Dataset);

Feature_Learner = net.Layers(142);
Output_Classifier = net.Layers(144);

Number_of_Classes = numel(categories(Training_Dataset.Labels));

New_Feature_Learner = fullyConnectedLayer(Number_of_Classes, ...
    'Name', 'Animal Feature Learner', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);
New_Classifier_Learner = classificationLayer('Name', 'Animal Classifier');

Layer_Graph =layerGraph(net);

New_Layer_Graph = replaceLayer(Layer_Graph, Feature_Learner.Name, New_Feature_Learner);
New_Layer_Graph = replaceLayer(New_Layer_Graph, Output_Classifier.Name, New_Classifier_Learner);

analyzeNetwork(New_Layer_Graph)

size_of_MiniBatch = 5;
Validation_Frequency = floor(numel(Resized_Training_Images.Files)/size_of_MiniBatch);
Training_Options = trainingOptions('sgdm', ...
    'MiniBatchSize', size_of_MiniBatch, ...
    'MaxEpochs', 6, ...
    'InitialLearnRate', 3e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', Resized_Validation_Images, ...
    'ValidationFrequency', Validation_Frequency, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(Resized_Training_Images, New_Layer_Graph, Training_Options);