#### Shervin Minaee
#### July 12, 2016
#### Similarity Matching Network, without weight sharing, because answer and questions could have different embedding

import pickle, time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


##################################  1- Reading data
train_ques_feat = pickle.load( open( "questionsVecTrain.p", "rb" ) )
train_ans_feat = pickle.load( open( "answersVecTrain.p", "rb" ) )
train_label = pickle.load(open("labelsTrain.p", "rb"))

training_size = len(train_ques_feat) - 16000
indices = list(range(0, len(train_ques_feat) - 1))
shuffled_indices= np.random.permutation(indices)

q_t = []
a_t = []
l_t = []

for index in shuffled_indices:
    q_t.append(train_ques_feat[index])
    a_t.append(train_ans_feat[index])
    l_t.append(train_label[index])

questions = q_t
answers = a_t
labels = l_t

train_ques_feat = np.random.randn( len(questions), 200)
train_ans_feat = np.random.randn( len(questions), 200)
train_label = np.random.randn( len(questions), 1)

i = 0
while i < len(questions):
    train_ques_feat[i, :] = questions[i]
    train_ans_feat[i, :] = answers[i]
    train_label[i, 0] = labels[i]
    i += 1


tmp = np.random.randn( len(questions), 2)
for i in range(len(train_label)):
    if train_label[i] == 1.0:
        tmp[i] = [1, 0]
    else:
        tmp[i] = [0, 1]

train_label = tmp

for i in range( train_label.shape[0]):
    ran_num= (5./100)*np.random.random_sample()  ### a random number in [0,0.05]
    ran_num= round(ran_num,3)
    if ( train_label[i,0]>train_label[i,1] ):
        train_label[i,0]= train_label[i,0]- ran_num
        train_label[i,1]= train_label[i,1]+ ran_num
    else:
        train_label[i,0]= train_label[i,0]+ ran_num
        train_label[i,1]= train_label[i,1]- ran_num


num_train_pairs= train_ques_feat.shape[0]


### we need to shuffle the data, to have both positive and negative samples in each batch of training. Otherwise the first half are positive and the remaining negative
shuffled_indices= np.random.permutation( train_ques_feat.shape[0] )
# question_test_all = pickle.load( open( "question_test_all100.p", "rb" ) )
# question_validation_all = pickle.load( open( "question_validation_all100.p", "rb" ) )
# ques_test_dec2vec = pickle.load( open( "ques_test_dec2vec100.p", "rb" ) )
# ques_valid_dec2vec = pickle.load( open( "ques_valid_dec2vec100.p", "rb" ) )
# ans_dec2vec = pickle.load( open( "ans_dec2vec.p", "rb" ) )



#num_valid_pairs= valid_ques_feat.shape[0]
#num_test_pairs=  test_ques_feat.shape[0]

################################## End of Dara Preparation



##################################  2- Network and Learning Parameters

n_hidden_1 = 100 # 1st layer number of hidden nodes
n_hidden_2 = 50 # 2nd layer number of hidden nodes
n_fc1= 20
n_input = train_ans_feat.shape[1]    # feature dimension of question and asnwer, each 200
n_classes = 2    # either this question and answer match or not. [1,0]= match, and [0.1] means does not match
### number of parameters= 2*(200*100)+ 2*50*20+20*2= 42,040

learning_rate = 0.003  ### 0.0001* 2^1, 10 epoch, batch-size= 100
training_epochs = 100
batch_size = 100
display_step = 1 ### show the results of each epoch
################################## End of parameters





##################################  3- Building the Siamese Network Architecture

# tf Graph input
x1 = tf.placeholder("float", [None, n_input])  ### question features
x2 = tf.placeholder("float", [None, n_input])  ### answer features
y = tf.placeholder("float", [None, n_classes]) ### labels, whether similar or not
keep_prob = tf.placeholder(tf.float32)


# Create model
def Siamese_Network( x1, x2, keep_prob, weights, biases):
    
    #### 1st Hidden layer for the left block with RELU activation
    layer_L1 = tf.add(tf.matmul( x1, weights['h_L1']), biases['b_L1'])  #### the hidden layer on the left
    layer_L1 = tf.nn.relu(layer_L1)
    #### 2nd Hidden layer for the left block with RELU activation 
    layer_L2 = tf.add(tf.matmul(layer_L1, weights['h_L2']), biases['b_L2'])
    layer_L2 = tf.nn.relu(layer_L2)
    
    #### 1st Hidden layer for the right block with RELU activation
    layer_R1 = tf.add(tf.matmul( x2, weights['h_R1']), biases['b_R1'])  #### the hidden layer on the left
    layer_R1 = tf.nn.relu(layer_R1)
    #### 2nd Hidden layer for the right block with RELU activation
    layer_R2 = tf.add(tf.matmul(layer_R1, weights['h_R2']), biases['b_R2'])
    layer_R2 = tf.nn.relu(layer_R2)  
    
    #### combined layer  layer_L2-layer_R2
    combined_layer= tf.concat([ layer_L2, layer_R2], 1)
    
    ### fc1
    layer_fc1= tf.nn.relu( tf.matmul( combined_layer, weights['fc1']) + biases['fc1'] )
    
    #### dropput on the combined layer
    layer_fc1_drop = tf.nn.dropout( layer_fc1, keep_prob )
    
    #### softmax layer with linear activation
    out_layer = tf.nn.softmax( tf.matmul( layer_fc1_drop, weights['out']) + biases['out'] )  ### APPLIES regression on difference of two embedding
    
    return out_layer
######################### End of Siamese Architecture with no weight sharing
    

# Store layers weight & bias
weights = {
    'h_L1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h_L2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h_R1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h_R2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),  
    'fc1':  tf.Variable(tf.random_normal([2*n_hidden_2, n_fc1])),                                       
    'out':  tf.Variable(tf.random_normal([n_fc1, n_classes]))
}
biases = {
    'b_L1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b_L2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b_R1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b_R2': tf.Variable(tf.random_normal([n_hidden_2])), 
    'fc1': tf.Variable(tf.random_normal([n_fc1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Construct model
pred = Siamese_Network( x1, x2, keep_prob, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

## # L2 regularization for the fully connected parameters.
regularizers = (tf.nn.l2_loss(weights['fc1']) + tf.nn.l2_loss(biases['fc1'])+ tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['out']))

# Add the regularization term to the loss.
cost += 2e-3 * regularizers

# Initializing the variables
init = tf.initialize_all_variables()
#######################################  End of Network Architecture


#### saving the model
saver = tf.train.Saver()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int( num_train_pairs/batch_size) ## number of batches in each epoch
        ## Loop over all batches, first shuffle
        for i in range(total_batch):
            batch_x1= train_ques_feat[ shuffled_indices[i*batch_size:(i+1)*batch_size], :]
            batch_x2= train_ans_feat[  shuffled_indices[i*batch_size:(i+1)*batch_size], :]
            batch_y= train_label[  shuffled_indices[i*batch_size:(i+1)*batch_size], :]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x1: batch_x1, x2:batch_x2, keep_prob: 0.7, y: batch_y})
                                                          
            # Compute average loss
            avg_cost += c / total_batch
            if ( i%100==0):
                print("Epoch %d, Batch %d!" % ( epoch+1, i) )
        # Display logs per epoch step
        if epoch % display_step == 0:
            print( "Epoch", '%d' % (epoch+1), ", cost=", \
                "{:.9f}".format(avg_cost) )
    print( "Optimization Finished!" )



    ############################### Finding Recall @k for Validation and Test sets
#     pool_size= 100  ## number of negative answers for each question
#     recall_at_k= 1
#     num_valid_nonsim= len(question_validation_all)*pool_size
#     num_test_nonsim = len(question_test_all)*pool_size
#
#     ######################### Recall @k for Validation, we have 2000 questions in validation
#     recall_validation= np.zeros( (2000, 1) )
#     for i in range(2000):
#         current_GT=   question_validation_all[i]['ground_truth']
#         current_pool= question_validation_all[i]['answer_pool'][0:pool_size]
#         cur_ques_poolsize= pool_size
#         curr_ques_anslist= list()
#         curr_pred_label= np.zeros( (cur_ques_poolsize, 2) )
#         for k in range(pool_size): ## over pool of negative answers
#             curr_pred_label[ k,:]= pred.eval( session=sess, feed_dict={ x1: np.reshape(ques_valid_dec2vec[i],[1,200]), x2: np.reshape(ans_dec2vec[ int(current_pool[k])-1],[1,200]), keep_prob: 1 } )
#
#         prob_positive= curr_pred_label[:,0]  #### the probability of a sample being positive answer for this question
#         sort_descending = prob_positive.argsort()[::-1]  ### the index of sorted answer with highest prob to lowest
#         selected_answer= sort_descending[0:recall_at_k]  ### k "most" related answer
#         selected_answer= selected_answer.tolist()
#         cur_recall= 0
#         for sel_ans_ind in range(len(selected_answer)):
#             for GT_ans_ind in range(len(current_GT)):
#                 if  int( current_pool[selected_answer[sel_ans_ind]] )== int( current_GT[GT_ans_ind]):
#                     cur_recall= 1
#                     #print("found right answer!", "GT_ans_ind=", current_pool[selected_answer[sel_ans_ind]], "selected_ans_ind=", current_GT[GT_ans_ind])
#
#
#         #actual_answer= range(len(current_GT))
#         #intersection_answer= list(set(selected_answer) & set(actual_answer))
#         #cur_recall= float(len(intersection_answer))/float(len(current_GT))
#         recall_validation[i,0]= cur_recall
#         if ( i%100==0 ):
#             print("%d-th sample in validation" % i)
#
#     print("Recall at %d for Validation is: %f" % (recall_at_k, np.mean(recall_validation)) )
#     print("\n")
#
#
#     ######################### Recall @k for Test, we have 2000 questions in test
#
#     recall_test= np.zeros( (2000, 1) )
#     for i in range(2000):
#         current_GT=   question_test_all[i]['ground_truth']
#         current_pool= question_test_all[i]['answer_pool'][0:pool_size]
#         cur_ques_poolsize= pool_size
#         curr_ques_anslist= list()
#         curr_pred_label= np.zeros( (cur_ques_poolsize, 2) )
#         for k in range(pool_size): ## over pool of negative answers
#             curr_pred_label[ k,:]= pred.eval( session=sess, feed_dict={ x1: np.reshape(ques_test_dec2vec[i],[1,200]), x2: np.reshape(ans_dec2vec[ int(current_pool[k])-1],[1,200]), keep_prob: 1 } )
#
#         prob_positive= curr_pred_label[:,0]  #### the probability of a sample being positive answer for this question
#         sort_descending = prob_positive.argsort()[::-1]  ### the index of sorted answer with highest prob to lowest
#         selected_answer= sort_descending[0:recall_at_k]  ### k "most" related answer
#         selected_answer= selected_answer.tolist()
#         cur_recall= 0
#         for sel_ans_ind in range(len(selected_answer)):
#             for GT_ans_ind in range(len(current_GT)):
#                 if  int( current_pool[selected_answer[sel_ans_ind]] )== int( current_GT[GT_ans_ind]):
#                     cur_recall= 1
#                     #print("found right answer!", "GT_ans_ind=", current_pool[selected_answer[sel_ans_ind]], "selected_ans_ind=", current_GT[GT_ans_ind])
#
#
#         #actual_answer= range(len(current_GT))
#         #intersection_answer= list(set(selected_answer) & set(actual_answer))
#         #cur_recall= float(len(intersection_answer))/float(len(current_GT))
#         recall_test[i,0]= cur_recall
#         if ( i%100==0 ):
#             print("%d-th sample in Test" % i)
#
#     print("\nRecall at %d for Test is: %f" % (recall_at_k, np.mean(recall_test)) )
#
#
#
#
#     ## Validation Set Accuracy
#     correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#     # Calculate accuracy
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#     print( "Validation Accuracy:", accuracy.eval({x1: valid_ques_feat, x2: valid_ans_feat, keep_prob: 1, y: valid_label}), "\n" )
#
#
#
#
# #### Save model weights to disk: link: https://goo.gl/s4ScZA
#     model_path = "/home/zliu/sherv_vbox/tensorflow_codes/practice_shervin/Siamese_model.ckpt"
#     save_path = saver.save(sess, model_path)
#     print( "Model saved in file: %s" % save_path )




   
    


