class BNB:
    def __init__(self,X,y,a=1):
        self.a = a # alpha yperparametros
        self.y_count_each = np.unique(y, return_counts=True)[1] # 12500 pos and 12500 negative reviews
        self.categories = 2 # postivie and negative ( 0 or 1)
        self.num_of_words = X.shape[1] # number of words in the vocabulary
        

        return


    def fit(self,X,y):
        
        # propability of P(x==1) and P(x==0)
        # aka propability of being 1 and 0 , since we have fifty fifty 12500 pos
        # and 12500 neg its 0.5 and 0.5 for each one

        self.prop_of_each_class = self.y_count_each / self.y_count_each.sum()
        self.logged_prop_of_each_class = np.expand_dims(np.log(self.prop_of_each_class),axis = 1)

        # self.prop_of_each_class = [12500,12500]
        # so P(1) == P(0) == 0.5



        # propability of P(x|y) , the propability of x given y

        prob_ofX_knowingY = np.zeros([self.categories, self.num_of_words])
        # initializing numpy array with zeros
        # [[0.0.0.0.0 ... 0.0.0.]
        #  [0.0.0.0.0 ... 0.0.0.]   {     2 x (number of words)    }


        # ----------------------------------------------------------------------



        X_filtered = X[y==0,:] # take the rows where class is 0 


        number_of_zeros = np.zeros(self.num_of_words) # sum of 1s and 0s in each collumn
        counter = 0                                                          
        for i in range(0,self.num_of_words):
            counter = 0
            for j in range(0,X_filtered.shape[0]):
                # if (X_filtered[j,i] == 0):
                counter+=X_filtered[j,i]
            number_of_zeros[i] = counter



        n1 = (number_of_zeros + self.a)


        prob_ofX_knowingY[0,:] = n1 / (12500+self.a*2)

        

        # now same thing for class 1

        X_filtered = X[y==1,:] # take the rows where class is 0

        number_of_zeros2 = np.zeros(self.num_of_words) # sum of 1s and 0s in each collumn again 
        counter = 0                                                        
        for i in range(0,self.num_of_words):
            counter = 0
            for j in range(0,X_filtered.shape[0]):
                # if (X_filtered[j,i] == 0):
                counter+=X_filtered[j,i]
            number_of_zeros2[i] = counter
        

        n2 = (number_of_zeros2 + self.a)
        prob_ofX_knowingY[1,:] = n2 / (12500+self.a*2)
     

        
        # ----------------------------------------------------------------------


       # using log() to the entire array of propabilities

        self.logged_positive = np.log(prob_ofX_knowingY) # P(Xn|y=0)
        self.logged_negative = np.log(1 - prob_ofX_knowingY) # P(Xn|y=1)


    def predict(self, X):

        # log P(y|x) is proportional to log P(x|y) + log P(y)
        # each n x 1 column vector is contains a value proportional to P(y|x)
        # for every possible class of y
        
        self.logged_positive2 = np.dot(self.logged_positive,X.T)
        self.logged_negative2 = np.dot(self.logged_negative,X.T)

        
        log_likelihoods = self.logged_positive2 + self.logged_negative2 # n x m matrix , join arrays


        log_joint_likelihoods = log_likelihoods + self.logged_prop_of_each_class 




        preds = np.argmax(log_joint_likelihoods, axis = 0) # 1 x m matrix
        
        #preds = np.array(preds).squeez(e) # m-dimensional vector
        return preds