This is the subset of the description from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data 

**UPDATE (Nov 18, 2019):** The following files have been added post-competition close to facilitate ongoing research. See the File Description section for details.

* `test_public_expanded.csv`
* `test_private_expanded.csv`
* `toxicity_individual_annotations.csv`
* `identity_individual_annotations.csv`

# Background

At the end of 2017 the Civil Comments platform shut down and chose make their ~2m public comments from their platform available in a lasting open archive so that researchers could understand and improve civility in online conversations for years to come. Jigsaw sponsored this effort and extended annotation of this data by human raters for various toxic conversational attributes.

In the data supplied for this competition, the text of the individual comment is found in the `comment_text` column. Each comment in Train has a toxicity label (`target`), and models should predict the target toxicity for the Test data. This attribute (and all others) are fractional values which represent the fraction of human raters who believed the attribute applied to the given comment. For evaluation, test set examples with `target >= 0.5` will be considered to be in the positive class (toxic). 

The data also has several additional toxicity subtype attributes. Models do not need to predict these attributes for the competition, they are included as an additional avenue for research. Subtype attributes are:

* `severe_toxicity`
* `obscene`
* `threat`
* `insult`
* `identity_attack`
* `sexual_explicit` 

Note that the data contains different comments that can have the exact same text. Different comments that have the same text may have been labeled with different targets or subgroups.

In addition to the labels described above, the dataset also provides metadata from Jigsaw's annotation: toxicity_annotator_count and identity_annotator_count, and metadata from Civil Comments: created_date, publication_id, parent_id, article_id, rating, funny, wow, sad, likes, disagree. Civil Comments' label rating is the civility rating Civil Comments users gave the comment.

# Labelling Schema

To obtain the toxicity labels, each comment was shown to up to 10 annotators*. Annotators were asked to: "Rate the toxicity of this comment"

* Very Toxic (a very hateful, aggressive, or disrespectful comment that is very likely to make you leave a discussion or give up on sharing your perspective)
* Toxic (a rude, disrespectful, or unreasonable comment that is somewhat likely to make you leave a discussion or give up on sharing your perspective)
* Hard to Say
* Not Toxic

These ratings were then aggregated with the `target` value representing the fraction of annotations that annotations fell within the former two categories.

To collect the identity labels, annotators were asked to indicate all identities that were mentioned in the comment. An example question that was asked as part of this annotation effort was: "What genders are mentioned in the comment?"

* Male
* Female
* Transgender
* Other gender
* No gender mentioned

Again, these were aggregated into fractional values representing the fraction of raters who said the identity was mentioned in the comment.

The distributions of labels and subgroup between Train and Test can be assumed to be similar, but not exact.

*Note: Some comments were seen by many more than 10 annotators (up to thousands), due to sampling and strategies used to enforce rater accuracy.

# File descriptions

* `train.csv` - the training set, which includes toxicity labels and subgroups
* `test.csv` - the test set, which does not include toxicity labels or subgroups
* `sample_submission.csv` - a sample submission file in the correct format

The following files were added post-competition close, to use for additional research. Learn more here.

* `test_public_expanded.csv` - The public leaderboard test set, including toxicity labels and subgroups. The competition target was a binarized version of the toxicity column, which can be easily reconstructed using a >=0.5 threshold.
* `test_private_expanded.csv` - The private leaderboard test set, including toxicity labels and subgroups. The competition target was a binarized version of the toxicity column, which can be easily reconstructed using a >=0.5 threshold.
* `toxicity_individual_annotations.csv` - The individual rater decisions for toxicity questions. Columns are:
    - `id` - The comment id. Corresponds to id field in train.csv, test_public_labeled.csv, or test_private_labeled.csv.
    - `worker` - The id of the individual annotator. These worker ids are shared between toxicity_individual_annotations.csv and identity_individual_annotations.csv.
    - `toxic` - 1 if the worker said the comment was toxic, 0 otherwise.
    - `severe_toxic` - 1 if the worker said the comment was severely toxic, 0 otherwise. Note that any comment that was considered severely toxic was also considered toxic.
    - `identity_attack`, `insult`, `obscene`, `sexual_explicit`, `threat` - Toxicity subtype attributes. 1 if the worker said the comment exhibited each of these traits, 0 otherwise.
* `identity_individual_annoations.csv` - The individual rater decisions for identity questions. Columns are:
    - `id` - The comment id. Corresponds to id field in train.csv, test_public_labeled.csv, or test_private_labeled.csv.
    - `worker` - The id of the individual annotator. These worker ids are shared between toxicity_individual_annotations.csv and toxicity_individual_annotations.csv.
    - `disability`, `gender`, `race_or_ethnicity`, `religion`, `sexual_orientation` - The list of identities within this category that the rater noticed in the comment. Formatted a space-separated strings.


