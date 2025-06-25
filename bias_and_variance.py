#overfitting ergo Jcv being too high => case of high variance, Jtrain being too high => case of bias
from sklearn.linear_model import LinearRegression, Ridge
import utils


#fixing high bias methods: try adding polynomial degrees, more features and decreasing lambda
#poly degrees
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('Advanced_Learning_Algorithms\data\c2w3_lab2_data1.csv')

print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")

# Preview the first 5 rows
print(f"first 5 rows of the training inputs (1 feature):\n {x_train[:5]}\n")

model = LinearRegression()
utils.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=10, baseline=400)


#more features
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('Advanced_Learning_Algorithms\data\c2w3_lab2_data2.csv')

print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")

# Preview the first 5 rows
print(f"first 5 rows of the training inputs (2 features):\n {x_train[:5]}\n")
model = LinearRegression()
utils.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=6, baseline=250)

#decreasing the regularization parameter
reg_params = [10, 5, 2, 1, 0.5, 0.2, 0.1]
utils.train_plot_reg_params(reg_params, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)


#fixing high variance => decrease poly features, add more training data, increase regularization parameter
reg_params = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

# Define degree of polynomial and train for each value of lambda
utils.train_plot_reg_params(reg_params, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)

#smaller sets of features
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('Advanced_Learning_Algorithms\data\c2w3_lab2_data2.csv')

# Preview the first 5 rows
print(f"first 5 rows of the training set with 2 features:\n {x_train[:5]}\n")

# Prepare dataset with randomID feature
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('Advanced_Learning_Algorithms\data\c2w3_lab2_data3.csv')

# Preview the first 5 rows
print(f"first 5 rows of the training set with 3 features (1st column is a random ID):\n {x_train[:5]}\n")

#comparing 
# Define the model
model = LinearRegression()

# Define properties of the 2 datasets
file1 = {'filename':'Advanced_Learning_Algorithms\data\c2w3_lab2_data3.csv', 'label': '3 features', 'linestyle': 'dotted'}
file2 = {'filename':'Advanced_Learning_Algorithms\data\c2w3_lab2_data2.csv', 'label': '2 features', 'linestyle': 'solid'}
files = [file1, file2]

# Train and plot for each dataset
utils.train_plot_diff_datasets(model, files, max_degree=4, baseline=250)