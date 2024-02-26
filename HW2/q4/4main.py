import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import RandomForestClassifier

"""
Open .npy files
"""
train = np.load('train.npy')
train_labels = np.load('trainlabels.npy')

test = np.load('test.npy')
test_labels = np.load('testlabels.npy')

print(train_labels)
print(train.shape)
print(test.shape)

# Select five random data-points
indices = np.random.choice(train.shape[0], 5, replace=False)

# Plot the data-points and their labels
for i, idx in enumerate(indices):
    plt.subplot(1, 5, i + 1)
    plt.imshow(train[idx])
    plt.title(f'Label: {train_labels[idx]}')
    plt.axis('off')

plt.show()

# Reshape the data from (60000, 28, 28) to (60000, 784)
train = train.reshape(train.shape[0], -1)
test = test.reshape(test.shape[0], -1)

"""
(ii)
"""

max_leaf_nodes = [2**i for i in range(1,25)]
test_loss = []
train_loss = []
num_leaves = []
count = 0

# Train a series of decision trees
for n in max_leaf_nodes:
    count += 1
    print("count:", count)
    # Create a decision tree classifier
    my_tree = tree.DecisionTreeClassifier(max_leaf_nodes=n)
    # Train the classifier on the training data
    my_tree.fit(train, train_labels)
    # Predicted labels
    y_tag_train = my_tree.predict(train)
    y_tag_test = my_tree.predict(test)
    # Compute loss
    train_loss.append(zero_one_loss(train_labels, y_tag_train))
    test_loss.append(zero_one_loss(test_labels, y_tag_test))
    num_leaves.append(my_tree.get_n_leaves())


print("max_leaf_nodes:",max_leaf_nodes)
print("num_leaves:",num_leaves) #num leaves actually used (for iii)
print("min test_loss:", min(test_loss))
min_index = test_loss.index(min(test_loss))
print("min_index",min_index)
print("train_errors", train_loss[min_index])

print(("test_loss" , min(test_loss)))
min_index = test_loss.index(min(test_loss))
print("min_index",min_index)
print("train_errors", train_loss[min_index])


plt.figure(figsize=(8,6))
plt.plot(max_leaf_nodes, train_loss, label='train loss')
plt.plot(max_leaf_nodes, test_loss, label='test loss')
plt.xscale('log')
plt.xlabel('Maximum Leaf Nodes')
plt.ylabel('Zero-One Loss')
plt.legend()
plt.show()

"""
(v).a

Random forest classifier:
with a fixed number of estimators (default works just fine) but with varying number of
maximum allowed tree leaves for individual estimators.
"""
max_leaf_nodes = [2**i for i in range(1,13)]
test_loss = []
train_loss = []
count = 0
for n in max_leaf_nodes:
    count += 1
    print("count:", count)
    my_random_forest = RandomForestClassifier(n_estimators=100, max_leaf_nodes=n, random_state=0)
    my_random_forest.fit(train, train_labels)
    y_tag_train = my_random_forest.predict(train)
    y_tag_test = my_random_forest.predict(test)
    train_loss.append(zero_one_loss(train_labels, y_tag_train))
    test_loss.append(zero_one_loss(test_labels, y_tag_test))

print(("test_loss:", min(test_loss)))
min_index = test_loss.index(min(test_loss))
print("min_index:", min_index)
print("train_loss:", train_loss[min_index])

# total parameter count is the product of max_leaf_nodes * n_estimators
total_params = np.array(max_leaf_nodes) * 100
print("total_params:", total_params)
plt.plot(total_params, train_loss, label='Train')
plt.plot(total_params, test_loss, label='Test')
plt.xlabel('total_params:')
plt.ylabel('0-1 loss')
plt.xscale('log',base=2)
plt.legend()
plt.show()

"""
(v).b

Random forest classifier:
with a fixed maximum number of leaves but varying number of estimators.
"""

max_estimators = [2**i for i in range(1,11)]
test_loss = []
train_loss = []
count = 0
for n in max_estimators:
    count += 1
    print("count:", count)
    my_random_forest = RandomForestClassifier(n_estimators=n, max_leaf_nodes=100)
    my_random_forest.fit(train, train_labels)
    y_tag_train = my_random_forest.predict(train)
    y_tag_test = my_random_forest.predict(test)
    train_loss.append(zero_one_loss(train_labels, y_tag_train))
    test_loss.append(zero_one_loss(test_labels, y_tag_test))

print(("test_loss:", min(test_loss)))
min_index = test_loss.index(min(test_loss))
print("min_index:", min_index)
print("train_loss:", train_loss[min_index])

# total parameter count is the product of max_leaf_nodes * n_estimators
total_params = np.array(max_estimators) * 100
print("Total Params:", total_params)
plt.plot(total_params, train_loss, label='Train')
plt.plot(total_params, test_loss, label='Test')
plt.xlabel('Total Params')
plt.ylabel('0-1 loss')
plt.xscale('log',base=2)
plt.legend()
plt.show()

"""
(vi)

Random forest classifier:
with a fixed maximum number of leaves but varying number of estimators.
"""
max_leaf_nodes = [2**i for i in range(1,11)]
max_estimators = [2**i for i in range(1,11)]
test_loss = []
train_loss = []
num_leaves = []
count = 0
for n in max_leaf_nodes:
    count += 1
    print("count:", count)
    # Create a decision tree classifier
    my_tree = tree.DecisionTreeClassifier(max_leaf_nodes=n)
    # Train the classifier on the training data
    my_tree.fit(train, train_labels)
    # Predicted labels
    y_tag_train = my_tree.predict(train)
    y_tag_test = my_tree.predict(test)
    # Compute loss
    train_loss.append(zero_one_loss(train_labels, y_tag_train))
    test_loss.append(zero_one_loss(test_labels, y_tag_test))
    num_leaves.append(my_tree.get_n_leaves())

max_estimators = [2*(2 ** i) for i in range(1,11)]
for n in max_estimators:
    count += 1
    print("count:", count)
    my_random_forest = RandomForestClassifier(n_estimators=n, max_leaf_nodes=100)
    my_random_forest.fit(train, train_labels)
    y_tag_train = my_random_forest.predict(train)
    y_tag_test = my_random_forest.predict(test)
    train_loss.append(zero_one_loss(train_labels, y_tag_train))
    test_loss.append(zero_one_loss(test_labels, y_tag_test))


total_params = np.array(max_leaf_nodes +
                        [(5000 * e) for e in max_estimators])
plt.plot(total_params, train_loss, label='Train')
plt.plot(total_params, test_loss, label='Test')
plt.xlabel('Total Params')
plt.ylabel('0-1 loss')
plt.xscale('log')
plt.legend()
plt.show()