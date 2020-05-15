import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_ylim(-0.5, 14.5)
ax.set_title('A single plot')
fig.subplots_adjust(bottom=0.15, left=0.2)
min_val, max_val = 15, 15

intersection_matrix = np.random.randint(0, 2, size=(max_val, max_val))

ax.matshow(intersection_matrix, cmap=plt.cm.Blues)

for i in range(15):
    for j in range(15):
        c = intersection_matrix[j,i]
        c = round(c,2)
        ax.text(i, j, str(c), va='center', ha='center')


plt.show()
ax.set_title('A single plot')