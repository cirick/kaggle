__author__ = 'cririck'

import os
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import metrics
from sklearn.cross_validation import train_test_split

import tensorflow as tf

import matplotlib.pyplot as plt

data_file = os.path.join('data','data.csv')
submission_file = os.path.join('data','submission.csv')

df = pd.read_csv(data_file)

# add any new fields or data manipulations
df['away'] = df.matchup.str.contains('@')
df['home'] = df.matchup.str.contains('vs')

# test data where labels are NaN
df_nan_i = pd.isnull(df).any(1).nonzero()[0]
X_pred = df.loc[df_nan_i]
X_pred = X_pred.drop(['shot_made_flag'], axis=1)

# training data whre labels non NaN
X = df.dropna()
# labels
y = X['shot_made_flag']

# all other features
X = X.drop(['shot_made_flag'], axis=1)

def enum_label_data(df):
    labels_to_enum = ['action_type', 'combined_shot_type', 'season', 'shot_type',
                      'shot_zone_area', 'shot_zone_basic', 'shot_zone_range',
                      'team_id', 'team_name', 'game_date', 'matchup', 'away', 'home', 'opponent']

    for label in labels_to_enum:
        s = pd.Series(df[label], dtype="category")
        e = zip(*list(enumerate(s.cat.categories)))
        s = s.cat.rename_categories(e[0])
        df[label] = s

    return df

# enumerate data with labels
X_pred = enum_label_data(X_pred)
X = enum_label_data(X)

'''
# compare probability distributions for made/miss vs feature
label = 'shot_distance'
prob_made = dict()
prob_miss = dict()
for i, row in X.iterrows():
    if y[i] == 1:
        if X[label][i] in prob_made:
            prob_made[X[label][i]] += 1
        else:
            prob_made[X[label][i]] = 1
    else:
        if X[label][i] in prob_miss:
            prob_miss[X[label][i]] += 1
        else:
            prob_miss[X[label][i]] = 1

plt.subplot(211)
plt.bar(prob_made.keys(),prob_made.values())
plt.subplot(212)
plt.bar(prob_miss.keys(),prob_miss.values())
plt.show()

seasonal_data = list()
for i in X['season'].cat.categories:
    t = X['season'][X['season'] == i]
    seasonal_data.append(X[:][t])
'''


# feature selection
selector = SelectPercentile(f_classif)
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
scores[np.isnan(scores)] = 0
p = scores / scores.max()
feat_dis_i = np.where(p <= 0.2)[0]
feat_keep_i = np.where(p > 0.2)[0]
n_classes = len(feat_keep_i)

# display feature selection results
idx = np.zeros(len(p))
idx[feat_keep_i] = 1
feat_keep = p * idx
feet_dis = p * (np.logical_not(idx))

plt.figure()
plt.bar(np.arange(len(idx)), feet_dis, color='r', label='Features Discarded')
plt.bar(np.arange(len(idx)), feat_keep, color='g', label='Features Kept')
plt.xlabel('Feature Number')
plt.ylabel('P value')
plt.title('Feature Selection')
plt.xticks(np.arange(len(idx)))

# select best features
# X_pred = X_pred[:, feat_keep_i]
# X = X[:, feat_keep_i]
feat_keep = ['action_type', 'shot_distance', 'away', 'shot_zone_basic', 'shot_zone_range']
X = X[feat_keep]
# normalize
X = preprocessing.scale(X)

y = y.to_frame()
y['make'] = (y['shot_made_flag'] == 1)*1
y['miss'] = (y['shot_made_flag'] == 0)*1
y = y[['make', 'miss']].as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

'''
def hyper_param_selection():
    itr, scores = list(range(1, 20, 1)), []
    for i in itr:
        model = RandomForestClassifier(n_jobs=-1, n_estimators=70, max_depth=i)
        model.fit(X_train, y_train)
        x = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=10)
        print x
        scores.append(x)
    x = [i for i in itr for j in range(10)]
    plt.figure()
    sns.boxplot(x, np.array(scores).flatten())



# classify
model = RandomForestClassifier(n_jobs=-1, n_estimators=70, max_depth=7)
model.fit(X_train, y_train)

print cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=20).mean()

print("Classification Report:\n%s\n" % (
    metrics.classification_report(
        y_test,
        model.predict(X_test))))

fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict(X_test))
plt.figure()
plt.title('ROC')
plt.plot(fpr, tpr, 'b')
plt.show()

#submit
pred = model.predict(X_pred[feat_keep])
pd.DataFrame({'shot_id': X_pred.shot_id, 'shot_made_flag': pred}).to_csv(submission_file, index=False)
'''

'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

m_X_train = mnist.train.images
m_y_train = mnist.train.labels
m_X_test = mnist.test.images
m_y_test = mnist.test.labels
'''

num_features = X_train.shape[1]
num_examples = X_train.shape[0]
num_output = y_train.shape[1]

num_hidden_nodes = 200

num_lr_steps = 10
init_lr = 0.1
lr_drop = 1.5

l2_rate = 1e-2
num_epochs = 10
batch_size = 25

x = tf.placeholder(tf.float32, shape=[None, num_features])
y_ = tf.placeholder(tf.float32, shape=[None, num_output])

w1 = tf.Variable(tf.random_normal(shape=[num_features, num_hidden_nodes], dtype=tf.float32))
b1 = tf.Variable(tf.random_normal(shape=[num_hidden_nodes], dtype=tf.float32))
l1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal(shape=[num_hidden_nodes, num_output], dtype=tf.float32))
b2 = tf.Variable(tf.random_normal(shape=[num_output], dtype=tf.float32))
y = tf.nn.softmax(tf.matmul(l1, w2) + b2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))

# evaluate
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    learning_rate = init_lr
    for lr in xrange(num_lr_steps):
        print 'LR: %f' % learning_rate
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        learning_rate /= lr_drop

        for epoch in xrange(num_epochs):
            avg_cost = 0
            batch = num_examples // batch_size
            for i in xrange(batch):
                start_idx = i*batch_size % num_examples
                train_dict = {x: X_train[start_idx:start_idx+batch_size, :],
                              y_: y_train[start_idx:start_idx+batch_size]}

                sess.run(train_step, feed_dict=train_dict)
                avg_cost += sess.run(cross_entropy, feed_dict=train_dict)/batch

            print 'Epoch: %d' % (epoch+1), 'Cost: %f' % avg_cost

        print 'Cost: %f' % accuracy.eval(feed_dict={x: X_test, y_: y_test} )



