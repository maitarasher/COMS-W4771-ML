import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_train = pd.read_csv(r'~/Documents/ML/HW2/q1/propublicaTrain.csv')
df_test = pd.read_csv(r'~/Documents/ML/HW2/q1/propublicaTest.csv')

print(df_train)

df_train_0 = df_train.loc[df_train['two_year_recid'] == 0]
df_train_1 = df_train.loc[df_train['two_year_recid'] == 1]

df_test_0 = df_test.loc[df_test['two_year_recid'] == 0]
df_test_1 = df_test.loc[df_test['two_year_recid'] == 1]

"""
Normalize n x d matrix to be used for MLE classifier and K_nn
where n is the number of data points, and d is the number of features

NOTE: I did no see any significant different in the result so I will not normalize the matrices  
"""
def norm(matrix):
    m = np.zeros(matrix.shape)
    m[:,0] = matrix[:,0]
    m.astype(float)
    for d in range(1, m.shape[1]):
        if (np.sum(matrix[:,d]) != 0):
            m[:,d] = np.divide(matrix[:,d], np.sum(matrix[:,d]))
    return m

m_train = df_train.to_numpy()
m_test = df_test.to_numpy()

nm_train = norm(m_train)
nm_test = norm(m_test)

m_train_0 = df_train_0.to_numpy()
m_train_1 = df_train_1.to_numpy()

nm_train_0 = norm(m_train_0)
nm_train_1 = norm(m_train_1)

m_test_0 = df_test_0.to_numpy()
m_test_1 = df_test_1.to_numpy()


"""
(1) MLE classifier

class priors:
print("mle_p_0:", mle_p_0)
print("mle_p_1:", mle_p_1)
mle_p_0_2 =  (m_train_0.shape[0] / (m_train_0.shape[0] + m_train_1.shape[0])) gives same result
"""
mle_p_0 =  (df_train['two_year_recid'].value_counts()[0]) / (df_train.shape[0])
mle_p_1 =  (df_train['two_year_recid'].value_counts()[1]) / (df_train.shape[0])
class_priors = [mle_p_0, mle_p_1]

"""
To compute mu - find the mean value of each feature 
"""
def compute_mu_sigma(matrix, c):
    """
    param:
        matrix: r x (d+1) matrix
        c: some constant to add to the covariance matrix
    return:
        mu - 1 x d vector
        sigma - d x d matrix

    Notes:
    the 0 index in matrix corresponds to the y label
    r=2000, d=9
    """
    r, d1 = matrix.shape
    d = d1-1

    mu = np.zeros(d) # 1 X d array
    sigma = np.zeros([d,d])

    for i in range(1,d+1):
        mu[i-1] = (np.sum(matrix[:,i]) / r)

    for i in range(r):
        x_i = matrix[i,1:]
        np.add(sigma, (np.subtract(x_i,mu).T) @ (np.subtract(x_i,mu)))

    sigma = sigma * (1/r)

    return mu, np.add(sigma, (c) * np.identity(d))



def multivariate_gaussian_pdf(data,mu,sigma):
    """
    param:
        data: 1 x (d+1) vector (some row of data)
    return:
        prob: p{mu,sigma}(data)
    """
    x = data[1:] #0 index is the label, so now x is in shape 1 x d
    d = x.shape[0]
    sigma_inv = np.linalg.inv(sigma)
    denominator = np.sqrt((2 * np.pi) ** d * np.linalg.det(sigma))
    #expo = -(1 / 2) * np.matmul(np.matmul((x - mu).T, sigma_inv), (x - mu))
    expo = -(1/2) * ((x - mu) @ sigma_inv @ (x - mu).T)
    prob = float((1.0 / denominator) * np.exp(expo))
    return prob

def mle_classifier(data, pdf, mu_0, mu_1, sigma_0, sigma_1, class_priors):
    """
    :param data: 1 x (d+1) - some raw of data
    :param pdf:
    :param mu_0:
    :param mu_1:
    :param sigma_0:
    :param sigma_1:
    :param class_priors:
    :return: argmax over y of P(X=x|Y=y) * P(Y=y)
    """
    probs = np.zeros(2)
    probs[0] = pdf(data, mu_0, sigma_0) * class_priors[0]
    probs[1] = pdf(data, mu_1, sigma_1) * class_priors[1]
    return np.argmax(probs)


mu_train_0, sigma_train_0 = compute_mu_sigma(m_train_0, 1)
mu_train_1, sigma_train_1 = compute_mu_sigma(m_train_1, 1)

"""
print("mu_test_0: \n", mu_train_0, "\nsigma_test_0: \n", sigma_train_0)
print("mu_test_1: \n", mu_train_1, "\nsigma_test_1: \n", sigma_train_1)
"""


mle_correct = 0
mle_positive_predictions = 0
mle_num_dp_0 = 0
mle_num_dp_1 = 0
mle_numerator_eo_pp_0 = 0 #true positive (Y^=1, Y=1)
mle_numerator_eo_pp_1 = 1 #true positive (Y^=1, Y=1)
mle_num_eo_pp_tn_0 = 0 #true negative (Y^=0, Y=0) A=0
mle_num_eo_pp_tn_1 = 0 #true negative (Y^=0, Y=0) A=1




for i in range(m_test.shape[0]):
    y = m_test[i][0]
    y_tag = mle_classifier(m_test[i,:], multivariate_gaussian_pdf, mu_train_0, mu_train_1, sigma_train_0, sigma_train_1, class_priors)
    if (y_tag == y):
        mle_correct += 1
        # To calculate EO for numerator count how many times I predicted correctly y^=y=1 + my sensitive attribute equaled 0
        if (y == 1 and m_test[i,3] == 0):
            mle_numerator_eo_pp_0 += 1
        # To calculate EO for numerator count how many times I predicted correctly y^=y=1 + my sensitive attribute equal 1
        if (y == 1 and m_test[i,3] == 1):
            mle_numerator_eo_pp_1 += 1
        # To calculate EO for numerator count how many times I predicted correctly y^=y=0 + my sensitive attribute equaled 0
        if (y == 0 and m_test[i, 3] == 0):
            mle_num_eo_pp_tn_0 += 1
            # To calculate EO for numerator count how many times I predicted correctly y^=y=0 + my sensitive attribute equal 1
        if (y == 0 and m_test[i, 3] == 1):
            mle_num_eo_pp_tn_1 += 1
    if (y_tag == 1):
        mle_positive_predictions += 1
        # To calculate DP for numerator count of how many times I predicted the positive class + my sensitive attribute equal 0
        # (use this count for PP denominator)
        if (m_test[i,3] == 0):
            mle_num_dp_0 += 1
        # To calculate DP for numerator count of how many times I predicted the positive class + my sensitive attribute equaled 0
        if (m_test[i,3] == 1):
            mle_num_dp_1 += 1


mle_acc = (mle_correct) / (m_test.shape[0])
print("mle_acc:", mle_acc)
Y = [mle_acc]

# P(Y^=1,A=0)/P(A=0)
mle_dp_0 = mle_num_dp_0 / (np.count_nonzero(m_test[:,3] == 0))
# P(Y^=1,A=1)/P(A=1)
mle_dp_1 = mle_num_dp_1 / (np.count_nonzero(m_test[:,3] == 1))
print("mle_dp_0:", mle_dp_0)
print("mle_dp_1:", mle_dp_1)
DP_0 = [mle_dp_0]
DP_1 = [mle_dp_1]

mle_correct_eo_0 = mle_numerator_eo_pp_0 / (np.count_nonzero(m_test_1[:,3] == 0))
mle_correct_eo_1 = mle_numerator_eo_pp_1 / (np.count_nonzero(m_test_1[:,3] == 1))
print("mle_correct_eo_0:", mle_correct_eo_0)
print("mle_correct_eo_1:", mle_correct_eo_1)
EO_0 = [mle_correct_eo_0]
EO_1 = [mle_correct_eo_1]

mle_eo_0_tn = mle_num_eo_pp_tn_0 / (np.count_nonzero(m_test_0[:,3] == 0))
mle_eo_1_tn = mle_num_eo_pp_tn_1 / (np.count_nonzero(m_test_0[:,3] == 1))
EO_0_tn = [mle_eo_0_tn]
EO_1_tn = [mle_eo_0_tn]

mle_pp_0 = mle_numerator_eo_pp_0 / mle_num_dp_0
mle_pp_1 = mle_numerator_eo_pp_1 / mle_num_dp_1
print("mle_pp_0", mle_pp_0)
print("mle_pp_1:", mle_pp_1)
PP_0 = [mle_pp_0]
PP_1 = [mle_pp_1]

"""
(2) Nearst neighbor classifier
"""
def k_NN_classifier(k, L, train, test):
    """
    :param k: number of nearst neighbors
    :param L: distance metric, accepts either 1,2, or infinity
    :param train: train set as metrix
    :param test: test set as metrix
    :return: the predicted class of every data in the testing set
    """

    y_tag = np.zeros(test.shape[0])

    """
    for each data point in the TEST data:
      (1) calculate the distance between one testing point to each of the training points (square diff if needed)
      (2) find k nearst neighbors
      (3) choose majority neighbors label and return as the predicted label   
    """
    for r in range(test.shape[0]):
        test_point = test[r,:]

        # create a matrix, where each row is the test_point and # of rows is as # of rows in the train set
        a_tp = []
        a_tp.append(test_point)
        m_tp = np.repeat(a_tp, repeats=train.shape[0], axis=0)

        m_diff = np.subtract(train,  m_tp)
        m_diff = np.absolute(m_diff)

        if (L == 2):
            m_diff = np.square(m_diff)

        m_dist = []
        for i in range(m_diff.shape[0]):
            row = m_diff[i,:]
            # each item in the list is storing [distance, train label, test label = true label]
            if (L == float('inf')):
                max_dist = np.max(row[1:])
                m_dist.append([max_dist, train[i, 0], test_point[0]])
            else:
                m_dist.append([np.sum(row[1:]), train[i,0], test_point[0]])

        # Now m_dist is a list (of list) of all distances from a spesific test point to any training point
        m_dist.sort(key=lambda x: x[0])
        k_nn = m_dist[:k]

        train_0_count = 0
        for sublist in k_nn:
            if (sublist[1] == 0):
                train_0_count += 1

        if (train_0_count > int(k/2)):
            majority_label = 0
        else:
            majority_label = 1

        #update test_list to include the k_nn for a spesific test point
        y_tag[r] = majority_label
    return y_tag


k_NN_correct = 0
k_NN_num_dp_0 = 0
k_NN_num_dp_1 = 0
k_NN_numerator_eo_pp_0 = 0
k_NN_numerator_eo_pp_1 = 0



y_tag = k_NN_classifier(5, 1, m_train, m_test)
for i in range(m_test.shape[0]):
    y = m_test[i][0]
    if (y_tag[i] == y):
        k_NN_correct += 1
        # To calculate EO for numerator count how many times I predicted correctly y^=y=1 + my sensitive attribute equal 0
        if (y == 1 and m_test[i,3] == 0):
            k_NN_numerator_eo_pp_0 += 1
        # To calculate EO for numerator count how many times I predicted correctly y^=y=1 + my sensitive attribute equal 1
        if (y == 1 and m_test[i, 3] == 1):
            k_NN_numerator_eo_pp_1 += 1
    # to calculate DP: for numerator count of how many times f predicted true and sensitive attribute equal 0
    if (y_tag[i] == 1 and m_test[i,3] == 0):
        k_NN_num_dp_0 += 1
    # to calculate DP: for numerator count of how many times f predicted true and sensitive attribute equal 1
    if (y_tag[i] == 1 and m_test[i,3] == 1):
        k_NN_num_dp_1 += 1

k_NN_acc = (k_NN_correct) / (m_test.shape[0])
print("K_NN_acc:", k_NN_acc)
Y.append(k_NN_acc)

k_NN_dp_0 = k_NN_num_dp_0 / (np.count_nonzero(m_test[:,3] == 0))
k_NN_dp_1 = k_NN_num_dp_1 / (np.count_nonzero(m_test[:,3] == 1))
print("k_NN_dp_0:", k_NN_dp_0)
print("k_NN_dp_1:", k_NN_dp_1)
DP_0.append(k_NN_dp_0)
DP_1.append(k_NN_dp_1)

k_NN_eo_0 = k_NN_numerator_eo_pp_0 / (np.count_nonzero(m_test_1[:,3] == 0))
k_NN_eo_1 = k_NN_numerator_eo_pp_1 / (np.count_nonzero(m_test_1[:,3] == 1))
print("K_nn_eo_0:", k_NN_eo_0)
print("K_nn_eo_1:", k_NN_eo_1)
EO_0.append(k_NN_eo_0)
EO_1.append(k_NN_eo_1)

# PP -- P_0[Y=1|Y^=1] = P[Y^=1,Y=1,A=0|Y^=1,A=0]
k_NN_pp_0 = k_NN_numerator_eo_pp_0 / (k_NN_num_dp_0)
print("k_NN_pp_0:", k_NN_pp_0)
# PP -- P_1[Y=1|Y^=1] = P[Y^=1,Y=1,A=1|Y^=1,A=1]
k_NN_pp_1 = k_NN_numerator_eo_pp_1 / (k_NN_num_dp_1)
print("k_NN_pp_1:", k_NN_pp_1)
PP_0.append(k_NN_pp_0)
PP_1.append(k_NN_pp_1)

"""
K = [3,5]
D = [1,2,float('inf')]

for k in K:
    for d in D:
        k_NN_correct = 0
        y_tag = k_NN_classifier(k,d,m_train, m_test)
        for i in range(m_test.shape[0]):
            y = m_test[i][0]
            if (y_tag[i] == y):
                k_NN_correct += 1
        k_NN_acc = (k_NN_correct) / (m_test.shape[0])
        print("k: ", k, "d:", d, "K_NN_acc:", k_NN_acc)
        Y.append(k_NN_acc)
        
k:  3 d: 1 K_NN_acc: 0.624
k:  3 d: 2 K_NN_acc: 0.6235
k:  3 d: inf K_NN_acc: 0.613
k:  5 d: 1 K_NN_acc: 0.636
k:  5 d: 2 K_NN_acc: 0.633
k:  5 d: inf K_NN_acc: 0.63
"""

"""
(3) Naive-bayes classifier
"""


def naive_bayes_classifier(class_priors, train_0, train_1, test):

    y_tag = np.zeros(test.shape[0])
    train_dps = [train_0.shape[0], train_1.shape[0]]

    """
    for each data point in the TEST data:
        for each class label:
            (1) calculate P(X=x|Y) by multiplying the count of each feature taking some value in the TRAIN set (+1 for smoothing)
                given class label, divided by total number of TRAIN data points
            (2) multipy by class prior (using TRAIN set)
        (3) take argmax of these probs to return predicted label
    """

    for r in range(test.shape[0]):
        probs = np.zeros(2)
        counts = np.ones(2)
        test_point = test[r,:]
        """if (r==0):
            print(test_point)"""
        for d in range(1, test_point.shape[0]):
            """if (r == 0):
                print(test_point[d])"""
            for y in range(2):
                if (y == 0):
                    counts[y] *= (train_0[:,d] == test_point[d]).sum() + 1
                    """if (r == 0):
                        print(train_0[:,d], (train_0[:,d] == test_point[d]).sum())
                        print("counts[y]:", counts[y])"""
                if (y == 1):
                    counts[y] *= (train_1[:, d] == test_point[d]).sum() + 1
                if (d == test_point.shape[0] - 1):
                    probs[y] = (1 / train_dps[y]) * counts[y] * class_priors[y]
                    """if (r == 0):
                        print(counts[y], probs[y])"""
        y_tag[r] = np.argmax(probs)

    return y_tag

naive_bayes_correct = 0
naive_bayes_dp_0_num = 0 #denominator of pp
naive_bayes_dp_1_num = 0 #denominator of pp
naive_bayes_numerator_eo_pp_0 = 0
naive_bayes_numerator_eo_pp_1 = 0

y_tag = naive_bayes_classifier(class_priors, m_train_0, m_train_1, m_test)
for i in range(m_test.shape[0]):
    y = m_test[i][0]
    if (y_tag[i] == y):
        naive_bayes_correct += 1
        # To calculate EO/PP: for numerator count how many times I predicted correctly y^=y=1 + my sensitive attribute equals 0
        if (y == 1 and m_test[i, 3] == 0):
            naive_bayes_numerator_eo_pp_0 += 1
        # To calculate EO/PP: for numerator count how many times I predicted correctly y^=y=1 + my sensitive attribute equals 1
        if (y == 1 and m_test[i, 3] == 1):
            naive_bayes_numerator_eo_pp_1 += 1
    # to calculate DP: for numerator count of how many times f predicted true and sensitive attribute equal 0
    if (y_tag[i] == 1 and m_test[i, 3] == 0):
        naive_bayes_dp_0_num += 1
    # to calculate DP: for numerator count of how many times f predicted true and sensitive attribute equal 1
    if (y_tag[i] == 1 and m_test[i, 3] == 1):
        naive_bayes_dp_1_num += 1


naive_bayes_acc = (naive_bayes_correct) / (m_test.shape[0])
print("naive_bayes_acc:", naive_bayes_acc)
Y.append(naive_bayes_acc)

# DP -- P_0[Y^=1] = P(Y^=1,A=0)/P(A=0), P_1[Y^=1] = P(Y^=1,A=1)/P(A=1)
naive_bayes_dp_0 = naive_bayes_dp_0_num / (np.count_nonzero(m_test[:,3] == 0))
naive_bayes_dp_1 = naive_bayes_dp_1_num / (np.count_nonzero(m_test[:,3] == 1))
print("naive_bayes_dp_0:", naive_bayes_dp_0)
print("naive_bayes_dp_1:", naive_bayes_dp_1)
DP_0.append(naive_bayes_dp_0)
DP_1.append(naive_bayes_dp_1)


# EO -- P_0[Y^=1|Y=1] = P[Y^=1,Y=1,A=0|Y=1,A=0]
naive_bayes_eo_0 = naive_bayes_numerator_eo_pp_0 / (np.count_nonzero(m_test_1[:,3] == 0))
print("naive_bayes_eo_0:", naive_bayes_eo_0)
# EO -- P_1[Y^=1|Y=1] = P[Y^=1,Y=1,A=1|Y=1,A=1]
naive_bayes_eo_1 = naive_bayes_numerator_eo_pp_1 / (np.count_nonzero(m_test_1[:,3] == 1))
print("naive_bayes_eo_1:", naive_bayes_eo_1)
EO_0.append(naive_bayes_eo_0)
EO_1.append(naive_bayes_eo_1)

# PP -- P_0[Y=1|Y^=1] = P[Y^=1,Y=1,A=0|Y^=1,A=0]
naive_bayes_pp_0 = naive_bayes_numerator_eo_pp_0 / (naive_bayes_dp_0_num)
print("naive_bayes_pp_0:", naive_bayes_pp_0)
# PP -- P_1[Y=1|Y^=1] = P[Y^=1,Y=1,A=1|Y^=1,A=1]
naive_bayes_pp_1 = naive_bayes_numerator_eo_pp_1 / (naive_bayes_dp_1_num)
print("naive_bayes_pp_1:", naive_bayes_pp_1)
PP_0.append(naive_bayes_pp_0)
PP_1.append(naive_bayes_pp_1)

X = ["MLE", "K_NN", "Naive Bayes"]
plt.bar(X, Y)
plt.ylabel('Accuracy')
#plt.legend()
plt.show()

"""
(vi)
"""
X_axis = np.arange(len(X))

# DP -- P_0[Y^=1] = P_1[Y^=1]
# P_0[Y^=1] = P(Y^=1,A=0)/P(A=0)
# P_1[Y^=1] = P(Y^=1,A=1)/P(A=1)
plt.bar(X_axis - 0.2, DP_0, 0.4, label='P_0[Y^=1]')
plt.bar(X_axis + 0.2, DP_1, 0.4, label='P_1[Y^=1]')

plt.xticks(X_axis, X)
plt.xlabel("Classifiers")
plt.ylabel("Probability")
plt.title("Demographic Parity (DP)")
plt.legend()
plt.show()

# EO -- P_0[Y^=1|Y=1] = P_1[Y^=1|Y=1]
# P_0[Y^=1|Y=1] = P[Y^=1,Y=1,A=0|Y=1,A=0]
# P_0[Y^=1|Y=1] = P[Y^=1,Y=1,A=1|Y=1,A=1]
plt.bar(X_axis - 0.2, EO_0, 0.4, label='P_0[Y^=1|Y=1]')
plt.bar(X_axis + 0.2, EO_1, 0.4, label='P_1[Y^=1|Y=1]')

plt.xticks(X_axis, X)
plt.xlabel("Classifiers")
plt.ylabel("Probability")
plt.title("Equalized Odds (EO)")
plt.legend()
plt.show()


# PP -- P_0[Y=1|Y^=1] = P_1[Y=1|Y^=1]
plt.bar(X_axis - 0.2, PP_0, 0.4, label='P_0[Y=1|Y^=1]')
plt.bar(X_axis + 0.2, PP_1, 0.4, label='P_1[Y=1|Y^=1]')

plt.xticks(X_axis, X)
plt.xlabel("Classifiers")
plt.ylabel("Probability")
plt.title("Predictive Parity (PP)")
plt.legend()
plt.show()
