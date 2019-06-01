import numpy as np
import test_by_mark

fs = 100e3
t = np.arange(0, 2, 1/fs)
y = np.sin(2*np.pi*150*t)


x = test_by_mark.run_ihc(y, 150, fs)
print(type(x), x.shape)
