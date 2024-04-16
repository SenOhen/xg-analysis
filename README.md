# Analysis of Expected Goals (xG) as a metric in football.
Data Science Capstone project exploring the xG metric in football.

#Authors:
Senyo Ohene

# Abstract
This project propose a machine-learning-based method of determining the expected goals metric, which is the expectation from 0 to 1 that a goal will be scored from a particular shot. I examine full seasons from five different country football leagues and train five basic models to output probabilities that a shot will score based on its characteristics. From this, I obtain the importance of each feature.

# Introduction
Football is a game of goals. With goals coming so infrequently, there is incentive for football teams and players to 
Expected goals were first proposed by ____ in 2014?? 
Some papers that tackle similar problems are [1], [2], [3], and [4].

# Methods
## Data
This project uses data from the 2015/16 seasons of the Premier League, La Liga, Bundesliga, Ligue 1 and Serie A soccer leagues obtained from Statsbomb (cite). For each league, all shots taken in the first 3/4 of the league were added to a dataframe with the following attributes
- location
- shot type
- shot technique
- shot body part
- under pressure
- minute
- player position
- shot first time

It is important to note that more data feature could theoretically be added, but only subject to availability. Some features that were not added (due to difficulty of access), were goalkeeper position, stronger foot of the player, number of defenders between the ball and the goal at the moment the ball was struck, amond others.

I calculated the following features using the location of the shot.
- distance
- angle

I had to take some preprocessing steps, such as transforming the goal outcome to a binary attribute. I am only interested in whether a shot resulted in a goal or not. 
I also had to convert sparse matrices into boolean values.
I separated my data into nominal (features with Boolean or descriptive values) and numerical features.

Finally, I plot all the shots displaying their statsbomb xg and their proposed model xg for five different classifiers output using predict_proba().



## Model
The project constructs an expected goals model comparable to the benchmark (Statsbomb), using five different classifiers. I used Logistic Regression, SGDClassifier, GradientBoostingClassifier, RandomForestClassifier, and DecisionTreeClassifier.

This project struck a curious chord between classification and regression. It would not be appropriate to see this as a purely regressional task, because then we would only be measuring our proposed model's ability to mimic the results of Statsbomb's xG model. That said, we do note that Statsbomb's xG model is a good benchmark.
At the same time, this is not a purely classificational task either: we do not aim to predict whether a particular shot results in a goal or not.
Instead, we are interested in the probabilities.



## Model selection
The shots were used to train five classifiers, and the results were displayed to show their correlation with the Statsbomb xG. The model selection was based on the following characteristics:

The desired model has a few characteristics:
- it has a wide enough range of values to be realistic. The sample size of shots means that there should be a reasonable spread of goal expectation values.
- it should have a relatively high correlation with the industry benchmark. While the hope is to compare performance with the benchmark, and possibly improve it, it is unlikely that our proposed model's performance will be a vast improvement on the benchmark. I measured correlation using $R^2$ and the trendline of the graph.

Based on these characteristics, the Gradient Boosting classifier was chosen, for all 5 leagues.

# Evaluation
## Permutation Feature Importance
I tested my model on the test set and obtained the following results:

This seeks to determine which of the features are most important when it comes to assessing the probability of a goal being scored.
Here are the graphs I obtained for all five leagues:

Perhaps, unsurprisingly the distance is the most important, across the board.


Goal
The project seeks to answer the question: is it better to measure the quality of shots as they pertain to goals. More specifically the projects asks: over the course of a season, is it better to have fewer high quality shots, or more low quality ones?

# Discussion
There are many improvements that can be made to this project. For instance, it is always useful to train the model on more data, and also add a discussion on conceding goals instead of just scoring goals. What type of goals are conceded by the teams who concede the least? Or the most? Insights like this can be used to implement training ground strategies, and then results can be verified from game play.
Hyperparameter tuning can also be used to make improvements to the models instead of relying on default parameters.

# Statement of contributions

# Conclusion
A fairly functional expected goals model can be developed using some of the basic characteristics of any shot, although the performance and usefulness of the model increases the number of attributes it is trained on.

# References
CITE CODE!!!
McKay Johns
Professor Lee Jeongkyu
TA Ting Tang 
CITE PAPERS!!!
# Appendix