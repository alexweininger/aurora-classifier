import cv2, os, sys, os.path, csv
import numpy as np
import matplotlib.pyplot as plt
color = ('r','g','b')

f = open('./output_stretch.txt', 'a')

yes = plt.subplot2grid((4, 4), (0, 0), colspan=4)
no = plt.subplot2grid((4, 4), (1, 0), colspan=4)
yes.set_title('yes')
no.set_title('no')

def color_histograms(imgdir, auroraStatus):
	imgs = os.listdir(imgdir)
	for img_path in imgs:
		img = cv2.imread(imgdir + img_path)
		hist = []
		for i,col in enumerate(color):
			series = cv2.calcHist([img],[i],None,[256],[0,256])
			max = np.max(series)
			for x in range(len(series)):
				if series[x] > max / 2:
					series[x] = np.true_divide(series[x], 2)
				else:
					series[x] = series[x] * 2
			series = np.true_divide(series, np.max(series))
			hist.extend(series)
		if auroraStatus == 1:
			yes.plot(hist)
		else:
			no.plot(hist)
		c = ','
		f.write(f'{auroraStatus}: {c.join(map(str, hist))}\n')

color_histograms('./training_images/no/', 0)
color_histograms('./training_images/yes/', 1)

plt.show()
