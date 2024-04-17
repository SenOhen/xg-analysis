# Analysis of Expected Goals (xG) as a metric in football.
Data Science Capstone project exploring the xG metric in football.

#Authors:
Senyo Ohene

# Abstract
This project proposes a machine-learning-based method of determining the expected goals metric, which is the expectation from 0 to 1 that a goal will be scored from a particular shot. I examine full seasons from five different country football leagues and train five basic models to output probabilities that a shot will score based on its characteristics. I find that a reasonable model can be built based on the distance, angle, shot type and other features of a shot. Furthermore, I find that distance of a shot is the most important feature in determining the expectation of a goal.

# Introduction
Football is a game of goals, in which individual moments matter disproportionately as compared with other sports. At the most recent FIFA World Cup in 2022, there were 172 goals scored in 64 matches, giving an average of 2.69 goals per match. Since goals occur so infrequently, an individual goal is worth very much, and there is incentive for football teams and players to invest in creating chances that maximize their goal scoring expectations.

Hewitt et al. use machine learning to build an expected goals model that adjusts for players and player positions in [1]. Rathke proposes a simple method based on simple ratios of goals scored to shots taken for different sections of the football field in [2]. Brechot et al. propose that examining teams' performance using expected goals is better than using the number of points they have at a particular point in the season in [3]. Lastly, Spearman use spatiotemporal data to quantify the probability of a player scoring at any point during a match [4].
Fans of football who have watched the sport for any period of time have some idea of how likely it is that a shot will result in a goal intuitively, but it is a more involving process to quantify the quality of a shot, or the expectation that such a shot will result in a goal.
Current industry leaders in the sport analytics world have expected goals models that vary slightly from each other in value, which suggests that, perhaps obviously, they use different algorithms to generate their expectations.
The goal of this project is to use machine learning to develop a simple model that gives the expectation that a goal will be scored from a shot, and get an idea of which factors contribute the most to high quality chances.

# Methods
## Data
This project uses data from the 2015/16 seasons of the Premier League, La Liga, Bundesliga, Ligue 1 and Serie A soccer leagues obtained from Statsbomb (https://github.com/statsbomb). For each league, all shots taken in the first 75% of matchweeks of the league were added to a dataframe with the following attributes
- location (where the shot was taken)
- shot type (whether the shot was from open play, or a penalty, corner kick or free kick)
- shot technique (whether the shot was kicked facing the goal, over the head, with the back of the heel, lobbed over the keeper, etc.)
- shot body part (whether the shot was played with the left foot, right foot, head or other body)
- under pressure (whether the shot was taken under pressure from an opponent or not)
- minute (what minute the shot was taken)
- player position (the position of the player taking the shot i.e. defender, midfielder, striker etc.)
- shot first time (whether the shot was taken at once by the player, or the player took prior touches)

This resulted in a separate dataframe for each league, each with at least 5,500 shots.

The shot outcome (whether a goal was scored from the shot or not) was also added to the dataframe to eventually be used as a target attribute to train the models. 

It is important to note that more data feature could theoretically be added, but only subject to availability. Some features that were not added (due to difficulty of access), were goalkeeper position, stronger foot of the player, number of defenders between the ball and the goal at the moment the ball was struck, among others.

The following features are calculated using the location of the shot and added to the dataframe:
- distance (the distance from the shot location to the center of the goal)
- angle (the angle between the line from the shot location to the center of the goal and the line perpendicular to the goal line passing through the center of the goal)

I apply some preprocessing steps, such as transforming the shot outcome to a binary attribute. I am only interested in whether a shot resulted in a goal or not. My model takes no note of whether a shot that did not result in a goal missed the target, was saved, was blocked, hit the crossbar, just that it did not score.
I remove any shots from corner kicks, since their rarity makes them a problem for the models.
I convert boolean features from having values of True and nan (not a number) to True and False.
I separate my data into nominal (features with Boolean or descriptive values) and numerical features. Nominal features are imputed, one-hot encoded and scaled. Numerical features are imputed and scaled. 

## Models
Finally, the following models are trained on the features, using the shot outcome as a target attribute:
The project constructs an expected goals model comparable to the benchmark (Statsbomb), using five different classifiers. Default implementations of Logistic Regression, SGDClassifier, GradientBoostingClassifier, RandomForestClassifier, and DecisionTreeClassifier from sklearn are used.

Statsbomb's own xG values are used as a benchmark to compare our models to.
I plot all the shots displaying their statsbomb xg and their proposed model xg for five different classifiers output using the function predict_proba, which is a function of each classifier which provides the probability that a shot resulted in a goal.


## Model selection
The shots were used to train five classifiers, and the results were displayed to show their correlation with the Statsbomb xG. The model selection was based on the following characteristics:

The desired model has a few characteristics:
- Range of values: it has a wide enough range of values to be realistic. The sample size of shots means that there should be a reasonable spread of goal expectation values.
- it should have a relatively high correlation with the industry benchmark. While the hope is to compare performance with the benchmark, and possibly improve it, it is unlikely that our proposed model's performance will be a vast improvement on the benchmark. I measured correlation using $R^2$ and the trendline of the graph.

It is important to note that this project strikes a curious chord between classification and regression. It would not be appropriate to see this as a purely regressional task, because then we would only be measuring our proposed model's ability to mimic the results of Statsbomb's xG model. That said, we do note that Statsbomb's xG model is a good benchmark.
At the same time, this is not a purely classificational task either: we do not aim to predict whether a particular shot results in a goal or not.
Instead, we are interested in the probabilities.

Based on these characteristics, the Gradient Boosting classifier was chosen, for all 5 leagues. In all cases, the Gradient Boosting Classifier had the highest $R^2$ value with the statsbomb xG value, as well as the closest gradient to 1.

# Evaluation
## Performance of best model on test set
I tested my model on the test set and obtained the following results:

## Permutation Feature Importance

The permutation feature importance of a feature in a model is a measure of how different the output of the model is if the feature in question is randomly permuted and all other features remain the same. If outputs remain relatively unchanged despite permutation, then the feature is not particularly crucial in determining the output of the model. On the other hand, if the outputs differ significantly when a feature is permuted, then the feature is very important in determining the output.
This seeks to determine which of the features are most important when it comes to assessing the probability of a goal being scored.
Here are the graphs I obtained for all five leagues:

Perhaps, unsurprisingly the distance is the most important, across the board. This means that shots taken closer to the goal are more likely to score than those taken from further away.
In all cases, the second most important feature was the shot type, although in all cases, this feature


# Discussion
The main takeaway from the project is that of the basic features of a shot, distance is by far the greatest indicator of goal expectation. This means that teams should prioritize patterns of play that enable them to take shots as close to goal as possible.

There are many improvements that can be made to this project. For instance, it is always useful to train the model on more data, from individual shots to data on full matches and league seasons. For individual shots, the model would undoubtedly be improved by including data such as goalkeeper position and number of opposition players in the path between the ball and the goal. Over the course of a match, data on the stamina levels of the players, or even data on previous shots within the match could improve the model as well. The models can also be trained on more league seasons, instead of just one season for each league. Also, the model can be improved by calibrating the model based on the opponent. What type of shots does the opposing team tend to score or concede from? In addition, the model can be improved by making it player-specific: do particular players more likely to score particular types of chances?

Another useful component of analysis has to do with league performance of teams. and also add a discussion on conceding goals instead of just scoring goals. What type of goals are conceded by the teams who concede the least? Or the most? Insights like this can be used to implement training ground strategies, and then results can be verified from game play.

Hyperparameter tuning can also be used to make improvements to the models instead of relying on default parameters.

Another question that can be posed is: over the course of a season, is it better to have fewer high quality shots, or more low quality ones?

In addition, there is the question of xG overperformance. Some players and teams are known for being able to score chances which when analysed with a model, have a low expectation of scoring. How can the performance of these players and teams be incorporated into the model? How sustainable is it to play football that relies on scoring low-probability chances?

# Statement of contributions
Senyo Ohene is the author of the project. He conducted all the data collection, model training, model testing and model evaluation.

# Conclusion
A fairly functional expected goals model can be developed using some of the basic characteristics of any shot, although the performance and usefulness of the model increases the number of attributes it is trained on. The default Gradient Boosting Regressor provides a good model for expected goals, and teams can use it to inform training methods by making use of the feature importance of the characteristics of the shot. Across the board, the model identifies distance as the highest indicator of the quality of a shot, followed by other characteristics.

# References
CITE CODE!!!
[1] J. H. Hewitt and O. Karakuş, "A machine learning approach for player and position adjusted expected goals in football (soccer)," __Franklin Open__, 4, 100034, August 2023.
[2] A. Rathke, "An examination of expected goals and shot efficiency in soccer," __Journal of Human Sport and Exercise__, 12(2proc), S514-S529. doi:https://doi.org/10.14198/jhse.2017.12.Proc2.05 , 2017.
[3] M. Brechot and R. Flepp, "Dealing With Randomness in Match Outcomes: How to Rethink Performance Evaluation in European Club Football Using Expected Goals," __Journal of Sports Economics__ 2020, Vol. 21(4) pp. 335-362, 2020.
[4] W. Spearman, "Beyond expected goals," in __12th MIT sloan sports analytics conference__, pp. 1-17, Feb 2018.
McKay Johns
Professor Lee Jeongkyu
TA Ting Tang 
CITE PAPERS!!!
# Appendix