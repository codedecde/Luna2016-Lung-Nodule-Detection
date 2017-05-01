import numpy as np
import matplotlib.pyplot as plt
import pylab
import joblib
import cPickle as cp
# t = np.arange(0, 5, 0.2)
# t2 = np.arange(0, 5, 0.02)

# def f(t):
# return np.exp(-t)*np.cos(2*np.pi*t)
def dice_np(y_true, y_pred):
    y_true = y_true.reshape(y_true.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)

    #y_true = np.reshape(y_true, -1)
    #y_pred = np.reshape(y_pred, -1)
    # y_true = y_true/np.max(np.max(y_true))
    # y_pred = y_pred/np.max(np.max(y_pred))

    #y_true[y_true > 0.0] = 1.0
    #y_pred[y_pred > 0.0] = 1.0


    print('Shapes  : ', y_true.shape, y_pred.shape)
    intersection = y_true*y_pred

    #print('Int shape ', intersection.shape)
    intersection = np.sum(intersection, axis = 1)
    #print('Int shape new ', intersection.shape)
    dr1 = np.sum(y_true, axis=1)
    dr2 = np.sum(y_pred, axis=1)

    #print('Dr ', dr1, dr2)
    dr = dr1+dr2
    nr = 2*intersection


    x = nr/dr
    return np.mean(x)

results = np.load('data.npz')

print(results.files)

print(results['imgs_test_X'].shape, results['imgs_test_Y'].shape, results['imgs_test_Pred'].shape)

# for i in range(0,results['imgs_test_X'].shape[0]):
# 	plt.figure(1)
# 	#plt.axis('off')
# 	plt.title('%d"'%(i))
# 	plt.subplot(231)
# 	plt.imshow(results['imgs_test_X'][i][0])
# 	plt.subplot(232)
# 	plt.imshow(results['imgs_test_X'][i][0]*results['imgs_test_Y'][i][0], vmin = np.min(results['imgs_test_X'][i][0]), vmax = np.max(results['imgs_test_X'][i][0]))
# 	plt.subplot(233)
# 	plt.imshow(results['imgs_test_X'][i][0]*results['imgs_test_Pred'][i][0], vmin = np.min(results['imgs_test_X'][i][0]), vmax = np.max(results['imgs_test_X'][i][0]))
# 	plt.subplot(235)
# 	plt.imshow(results['imgs_test_Y'][i][0])
# 	plt.subplot(236)
# 	plt.imshow(results['imgs_test_Pred'][i][0])
# 	pylab.show()


#weight = joblib.load(('weights'))
weight = cp.load(open('filters.pkl'))

#weight.reshape(weight.shape[0], -1)

#print('Weights : ', weight.keys())
#plt.figure(3)
#plt.subplot(filters_per_layer)

for key in (weight):
	if key.startswith('conv_') and key.endswith('_2'):
		print(key, len(weight[key]))
		for j in range(0, len(weight[key])):
				#weight[key].reshape(6)
			plt.figure(2)
			plt.subplot(1,2,1)
			plt.imshow(weight[key][0])
			plt.axis('off')
			plt.subplot(1,2,2)
			plt.imshow(weight[key][1])
			plt.axis('off')
			plt.figure(3)
			plt.subplot(1,2,1)
			plt.imshow(weight[key][2])
			plt.axis('off')
			plt.subplot(1,2,2)
			plt.imshow(weight[key][3])
			plt.axis('off')
		pylab.show()


def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


#plots = np.zeros(results['imgs_test_Y'].shape[0],results['imgs_test_Y'].shape[2],results['imgs_test_Y'].shape[3])

#print(plots.shape)

#for i in range(0, results['imgs_test_Y'].shape[0]):


#plot_3d(results['imgs_test_Y'][])


####### PLOTTING
# plot_res = results['imgs_test_X'][0][0][0:200, 300:500]
# #*results['imgs_test_Y'][0][0][300:500]
# print('Shape : ', plot_res.shape)

ind = 19
# plt.figure(1)
# plt.title('Lung CT Scan')
# plt.imshow(results['imgs_test_X'][ind][0])
# plt.axis('off')
# #pylab.show()

# Im1 = results['imgs_test_Y'][ind][0]
# Im2 = results['imgs_test_Pred'][ind][0]

# print(dice_np(Im1, Im2))

# plt.figure(2)
# plt.title('Results')
# ax1 = plt.subplot(131)
# plt.imshow(results['imgs_test_X'][ind][0][150:350, 300:500], vmin = np.min(results['imgs_test_X'][ind][0]), vmax = np.max(results['imgs_test_X'][ind][0]))
# ax1.set_title('Region of Interest')
# plt.axis('off')
# ax1 = plt.subplot(132)
# plt.imshow(results['imgs_test_X'][ind][0][150:350, 300:500]*results['imgs_test_Y'][ind][0][150:350, 300:500], vmin = np.min(results['imgs_test_X'][ind][0]), vmax = np.max(results['imgs_test_X'][ind][0]))
# ax1.set_title('Gold Standard Mask')
# plt.axis('off')
# ax2 = plt.subplot(133)
# plt.imshow(results['imgs_test_X'][ind][0][150:350, 300:500]*results['imgs_test_Pred'][ind][0][150:350, 300:500], vmin = np.min(results['imgs_test_X'][ind][0]), vmax = np.max(results['imgs_test_X'][ind][0]))
# ax2.set_title('Predicted Mask')
# plt.axis('off')
# pylab.show()

###### PLOTTING END






# testMasks = np.load('testMasks.npy')
# trainedMasks = np.load('masksTestPredicted.npy')
# testIm = np.load('testImages.npy')
# trainIm = np.load('trainImages.npy')
# trainMasks = np.load('trainMasks.npy')
# print(testMasks.shape)
# print(trainedMasks.shape)
# print(testIm.shape)
# print(trainIm.shape)
# print(trainMasks.shape)

# for i in range(0,testIm.shape[0]):
# 	plt.figure(1)
# 	plt.subplot(231)
# 	plt.imshow(testIm[i][0])
# 	plt.subplot(232)
# 	plt.imshow(testMasks[i][0])
# 	plt.subplot(234)
# 	plt.imshow(testIm[i][0]*testMasks[i][0])
# 	plt.subplot(233)
# 	plt.imshow(trainedMasks[i][0])
# 	plt.subplot(235)
# 	plt.imshow(testIm[i][0]*trainedMasks[i][0])
# 	pylab.show()

# for i in range(0,trainIm.shape[0]):
# 	plt.figure(1)
# 	plt.subplot(131)
# 	plt.imshow(trainIm[i][0])
# 	plt.subplot(132)
# 	plt.imshow(trainMasks[i][0])
# 	plt.subplot(133)
# 	plt.imshow(trainMasks[i][0]*trainIm[i][0])
# 	pylab.show()

# print(np.sum(trainIm[0][0]), np.sum(trainIm[1][0]))
# for i in range(0,trainIm.shape[0]):
# 	plt.figure(2)
# 	plt.subplot(121)
# 	plt.imshow(trainIm[i][0])
# 	plt.subplot(122)
# 	plt.imshow(trainMasks[i][0]*trainIm[i][0])
# 	pylab.show()
# for i in range(0,trainedMasks.shape[0]):
# 	plt.figure(i+1)
# 	plt.imshow(trainedMasks[i][0])
# 	pylab.show()
# for i in range(0,testMasks.shape[0]):
# 	plt.figure(i+1)
# 	plt.imshow(testMasks[i][0])
# 	pylab.show()
# plt.figure(1)
# plt.subplot(211)
# plt.plot(t, f(t), 'bo', t2, f(t2), 'k')

# plt.subplot(212)
# plt.plot(t2, np.cos(2*np.pi*t2), 'k')

# pylab.show()