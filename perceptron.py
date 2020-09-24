import numpy as np

sign = lambda i: 1 if i>=0 else -1
sign = np.vectorize(sign)

def desc(w,x,y):
	hyp = sign(w.dot(x.T))
	idx = (hyp!=y).squeeze()
	sum_err = np.multiply(x[idx],y.T[idx]).sum(axis=0)
	return np.array([sum_err,])

def run(w,lr):
	x=np.array([(1,0,0),(1,1,1),(1,2,2)])
	y=np.array([[1,1,-1]])

	repeat = True
	i = 0
	while repeat:
		print("W{}:".format(i),w)
		hyp = sign(w.dot(x.T))
		print("H:", hyp)
		print("y:", y)
		repeat = not np.all(hyp==y)
		w += lr*desc(w,x,y)
		i += 1
	print("CONVERGED")
	return


W=np.array([[-1.,-1.,4.]])
run(W,.1)
print('\n'*10)

W=np.array([[-1.,-1.,2.]])
run(W,.1)
print('\n'*10)

W=np.array([[-1.,-1.,1.9]])
run(W,.1)
print('\n'*10)

W=np.array([[1.,-1.,0]])
run(W,.1)
print('\n'*10)
