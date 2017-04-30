def getImage(model, layer_name,filter_index):
    input_img = model.input
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,filter_index,:,:])
    grads = K.gradients(loss,input_img)
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([input_img, K.learning_phase()], [loss, grads])
    input_img_data = np.random.normal(loc= -0.0001, scale = 0.153, size=(1,1,512,512))
    step = 1.
    NUM_ITERATIONS = 20
    for i in xrange(NUM_ITERATIONS):
        print 'Iteration:',i,'Done'
        loss_value, grads_value = iterate([input_img_data,0])
        input_img_data += grads_value * step
    return input_img_data[0][0]

