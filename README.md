# DomainAdaptation
In emotion classification texts of different granularities will be mapped to a set of predefined emotions (Schuff et al., 2017), e.g. to the Ekman's basic emotions: anger, disgust, fear, joy, sadness, and surprise. 
Automatically emotion classification gains more popularity in different areas, as it helps to improve several applications in human-computer interaction.
Emotion detection in social media can be deployed, e.g., in the industry to recognise areas that are going well or problematic (Bougie et al., 2003). 
Emotion recognition in dialogue systems, like tutoring systems, helps to improve learning (Litman and Forbes-Riley, 2004).
The software with emotion recognition feature could increase the value of the hardware. 
For example Siri (Apple software for recognition and processing of the speech with the function like a personal assistant), which fails in dealing with emotions.

In real-world scenario, we have a little amount of well-annotated data on which the statistical model is trained, and we want to apply this model to real-world data.
Statistical models known from text classification require a significant amount of labelled data for accurate prediction.
However, data labelling is one of the challenges, as there are considerable costs to annotate data for each new area.
Aue and Gamon (2005) combined data from different domains to cover the lack of labelled data. 
Blitzer et al. (2007) conclude: “Adding more labeled data always helps, but diversifying training data does not”.
Performance of the model trained on one domain drops if we use it directly to the data from the other domain. 
The main reason is that many features from the domain on which the model is trained are no longer useful for the domain on which the model is applied. 
To capture that, we need some step before we train and apply the model, e.g. some method like “this feature from training domain behaves similar to this feature from testing domain and both can be aligned”. 
One possible solution is domain adaptation. 

Domain adaptation aims to adapt the knowledge learned in a previous domain providing labelled training data (so-called source domain) to the new domain (so-called target domain) on which the classifier will be tested to perform the same task (Pan et al., 2010). 
The advantage of domain adaptation is that no or little labelled data in the target domain are required. 
There are different approaches to perform domain adaptation, e.g. instance-based (Jiang and Zhai, 2007; Tsakalidis et al., 2014), model-based (Wang et al., 2016), and feature-based domain adaptation. 
Blitzer et al. (2006), Pan et al. (2010), Glorot et al. (2011) investigated the adaptation method for features, as it is more robust to domain shift.
While domain adaptation was successfully applied to different tasks (e.g. part of speech tagging or sentiment analysis), there are only a few works in emotion classification of written texts. 
We want to contribute to this topic with feature-based domain adaptation using Structural Correspondence Learning (SCL) (Ando et al., 2005; Blitzer et al., 2006; Blitzer et al., 2007).

To demonstrate an example for SCL, we consider two sentences from different domains, both expressing the same emotion “joy”:
1. The first sentence is from emotional events domain: 
“I felt very happy when I heard I had passed the examination to move up to the second year course.” 
2. The second sentence is from fairy tales domain: 
“And the rain fell upon the burdock-leaves, to play the drum for them, and the sun shone to paint colors on the burdock forest for them, and they were very happy; the whole family were entirely and perfectly happy.”

The words like “passed the examination” from personal events or “burdock forest” from fairy tales are domain-specific. 
We call them non-pivot features. 
There is also word “happy” which occur in both sentences and have in each domain the same meaning. 
We call them pivot features. 
If these both non-pivot features have a high correlation with the emotion “joy” and low correlation, for example, with emotion “sad” than they can be aligned. 

SCL align the pivot and non-pivot features extracted from the unlabelled source and target data using the weight vectors. 
For example, the pivot feature “happy” belongs to the class “joy”. If it often occur with particular non-pivot “burdock forest”, then it is highly likely that they both correspond with each other and therefore they get higher weight. 
After adding the weight vectors to the features in training and testing sets, we use labelled data from source domain, which help to build a good classifier for the target.

We follow the setting of Blitzer et al. (2007) and use four different domains for our experiments: 
emotional events descriptions, 
conversations, 
tales, 
and tweets. 
We take ISEAR (Scherer and Wallbott, 1994) for emotional events descriptions, DailyDialogs (Li et al., 2017) for conversations, Tales (Alm et al., 2005) for tales domain, TEC and CrowdFlower for tweets.

The criterion for our data set selection is enough instances for each label according to Ekman’s model to train and test the system. 
We choose domains for which preferably 613 instances per label and at least four labels from Ekman’s emotions set are available per dataset. 
The number of instances was limited by the number of minimum available examples. 

First, we perform in-domain emotion classification (training and testing set drawn from the same domain) on each of the five datasets. 
Then, we do cross-domain classification, one for the model adapted with SCL algorithm, and one for the baseline model (training on one domain and testing on the other without adaptation step). 
There are 20 domain-pairs for each cross-domain setting additionally to in-domain models. 
We use in-domain models as a gold standard model for each target domain in cross-domain settings
