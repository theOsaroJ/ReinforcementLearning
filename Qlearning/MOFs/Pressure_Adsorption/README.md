The RL codes depending on the metric of interest are provided. 
Prior.csv refers to the initial training dataset which was set to be the bounds of the data (1 and 1E7 Pa) and the corresponding adsorption. Test.csv refers to the ground-truth data to be explored/exploited by the RL agent.

Hyperparameter tuning was also done and the best parameters are chosen based on the best metric value (lowest MRE or highest R.square), users can turn off the Timer for the Q-learning hyperparameter search if they so choose.

![RL Schematic](https://github.com/theOsaroJ/ReinforcementLearning/assets/64130121/cca6d18f-afac-4501-a117-75f149c283b0)
