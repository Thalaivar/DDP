import numpy as np
from BSOID.cure import CURE

cure = CURE(25, 100, 100, alpha=0.5, n_rep=10)

data = np.random.rand(1000, 40)
cure.process(data)