import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

# f = open('output.txt', 'r')
f = open('./output_stretch.txt', 'r')

hists = []

for line in f:
	newstr = line.replace("[", "")
	newstr = newstr.replace("]", "")
	newstr = newstr.replace(":", ",")
	newstr = newstr.replace(" ", "")
	# newstr = newstr[:-2]
	hist = [float(item) for item in newstr.split(',')]
	hists.append(hist)

data = np.array(hists)

yes = plt.subplot2grid((4, 4), (0, 0), colspan=3)
no = plt.subplot2grid((4, 4), (1, 0), colspan=3)
w = plt.subplot2grid((4, 4), (2, 0), colspan=3)
err = plt.subplot2grid((4, 4), (3, 0), colspan=3)
yes.set_title('yes')
no.set_title('no')

for row in data:
	if row[0] == 0:
		no.plot(row[1:])
	else:
		yes.plot(row[1:])

rows = data.shape[0]
cols = data.shape[1]
L = data[:, 0]
arr = []
for x in range(len(data[0])):
	arr.append(0)

W = np.array(arr)
mu = 1.5
print(W)
W[0] = 10
print (" w0: %f w1: %f w2: %f Error: %f" %(W[0],W[1],W[2],0))

plt.ion()
plt.show()
batch_size = len(data)
acc = []
best_accuracy = 0
best_weights = []
accVsEpoch = []
for j in range(150): # epochs
	training = data[np.random.choice(data.shape[0], batch_size, replace=False), :]
	L = training[:, 0]
	yes = np.sum(L)
	no = batch_size - yes
	accuracy, accuracy_no, accuracy_yes = 0, 0, 0
	for i in range(len(training)):
		charge = float(W[0]) + np.sum(np.multiply(training[i, 1:], W[1:]))
		predict = 1 if charge > 0 else 0
		if predict == L[i]:
			if predict == 1:
				accuracy_yes += 1
			else:
				accuracy_no += 1
			accuracy += 1
		else:
			Error = predict - L[i]
			W_t = W
			X_t = np.concatenate(([1], data[i, 1:]))
			W_t = np.multiply(mu, np.multiply(Error, X_t))
			W = np.subtract(W, W_t)
			# print("Err: %f Chg: %f P: %f L[i]: %f "%(Error, charge, predict, L[i]))

	w.clear()
	w.set_title('weights')
	w.plot(W)

	print(f'Batch size: {batch_size}, total aurora: {yes}, total non: {no}')
	print('Total accuracy: %f, Yes accuracy: %f, No accuracy: %f'%((float(accuracy)) / len(training), (float(accuracy_yes)) / yes, (float(accuracy_no)) / no))
	a = float(accuracy) / len(training)
	accVsEpoch.append(a)
	if a > best_accuracy:
		best_accuracy = a
		best_weights = W
	acc.append(float(accuracy) / len(training))
	err.clear()
	err.set_title('accuracy')
	err.plot(acc)
	plt.pause(0.1)


plt.show()

def test(W):
	test = data[np.random.choice(data.shape[0], 100, replace=False), :]
	L = test[:, 0]
	yes, no = 0, 0
	accuracy = 0
	accuracy_no, accuracy_yes = 0, 0
	for i in range(len(test)):
		charge = float(W[0]) + np.sum(np.multiply(test[i, 1:], W[1:]))
		predict = 1 if charge > 0 else 0
		if L[i] == 1:
			yes+=1
		else:
			no += 1
		if predict == L[i]:
			# print('got it right')
			if predict == 1:
				accuracy_yes += 1
			else:
				accuracy_no += 1
			accuracy += 1
	print('Total accuracy: %f, Yes accuracy: %f, No accuracy: %f'%((float(accuracy)) / len(test), (float(accuracy_yes)) / yes, (float(accuracy_no)) / no))

test(W)
print(f'test accuracy: {best_accuracy}:')
test(best_weights)

out_acc = pd.DataFrame(accVsEpoch)
out_acc.to_csv('results.csv')
