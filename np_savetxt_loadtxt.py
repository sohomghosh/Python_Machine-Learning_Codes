import numpy as np
np.savetxt('/data/click_bait_detect/posttext_glove_vectors_average.txt',features_sent_sub_pt)
features_sent_sub_pt = np.loadtxt('/data/click_bait_detect/posttext_glove_vectors_average.txt')
